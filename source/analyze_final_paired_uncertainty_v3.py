"""Qualified final GT/REC comparisons; no training or navigation is launched.

Requires all three paired training seeds, both full official splits, and real
final-checkpoint qualification. Resampling moves all instructions for a route
together and always keeps GT/REC paired. Small-seed uncertainty is descriptive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np

GROUPS = ('gt-joint-boundary', 'rec-joint-boundary')
METRICS = ('success', 'oracle_success', 'distance_to_goal', 'ndtw', 'sdtw')
SEEDS = (10, 11, 12)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_key(episode):
    return (int(episode['scene_id']), tuple(tuple(map(float, row))
                                          for row in episode['reference_path']))


def recipe(config, across_seeds=False):
    ignored = {'group', 'run_root', 'dataset', 'output_dir'}
    if across_seeds:
        ignored |= {'seed', 'sampler_consumed_indices_sha256',
                    'sampler_budget_sha256', 'sampler_budget_relative', 'planned_followup_seeds'}
    result = {key: value for key, value in config.items() if key not in ignored}
    args = list(result['training_args'])
    for flag in ['--output_dir'] + (['--seed', '--data_seed'] if across_seeds else []):
        index = args.index(flag)
        args[index + 1] = '<controlled difference>'
    result['training_args'] = args
    return result


def aggregate(episodes, rows_by_seed):
    """Return episode-weighted sums with a physical-route resampling axis."""
    keys = list(dict.fromkeys(geometry_key(episode) for episode in episodes))
    indices = {key: index for index, key in enumerate(keys)}
    counts = np.zeros(len(keys), dtype=np.int64)
    values = np.zeros((len(SEEDS), 2, len(keys), len(METRICS)), dtype=float)
    expected = {str(episode['episode_id']) for episode in episodes}
    assert len(expected) == len(episodes)
    assert set(rows_by_seed) == set(SEEDS)
    for seed in SEEDS:
        assert set(rows_by_seed[seed]) == set(GROUPS)
        for group in GROUPS:
            assert set(rows_by_seed[seed][group]) == expected, 'Exact episode coverage required'
    for episode in episodes:
        eid = str(episode['episode_id'])
        cluster = indices[geometry_key(episode)]
        counts[cluster] += 1
        for si, seed in enumerate(SEEDS):
            for gi, group in enumerate(GROUPS):
                row = rows_by_seed[seed][group][eid]
                numbers = [float(row[key]) for key in METRICS]
                assert all(math.isfinite(value) for value in numbers)
                assert row['success'] in (0, 1) and row['oracle_success'] in (0, 1)
                values[si, gi, cluster] += numbers
    return counts, values


def summarize(counts, values, repetitions, analysis_seed):
    assert repetitions >= 1000 and values.shape == (3, 2, len(counts), len(METRICS))
    assert (counts > 0).all() and np.isfinite(values).all()
    total = int(counts.sum())
    per_seed = values.sum(axis=2) / total
    differences = values[:, 1] - values[:, 0]
    conditional = np.empty((repetitions, len(METRICS)))
    hierarchical = np.empty_like(conditional)
    rng = np.random.default_rng(analysis_seed)
    for start in range(0, repetitions, 256):
        end = min(start + 256, repetitions)
        size = end - start
        route_draw = rng.integers(0, len(counts), size=(size, len(counts)))
        seed_draw = rng.integers(0, 3, size=(size, 3))
        denominator = counts[route_draw].sum(axis=1)[:, None]
        conditional[start:end] = differences.mean(axis=0)[route_draw].sum(axis=1) / denominator
        paired_seed_means = differences[seed_draw].mean(axis=1)
        selected = np.take_along_axis(paired_seed_means, route_draw[:, :, None], axis=1)
        hierarchical[start:end] = selected.sum(axis=1) / denominator
    result = {}
    for mi, metric in enumerate(METRICS):
        delta = per_seed[:, 1, mi] - per_seed[:, 0, mi]
        result[metric] = {
            'GT_per_training_seed': per_seed[:, 0, mi].tolist(),
            'REC_per_training_seed': per_seed[:, 1, mi].tolist(),
            'REC_minus_GT_per_training_seed': delta.tolist(),
            'GT_mean': float(per_seed[:, 0, mi].mean()),
            'REC_mean': float(per_seed[:, 1, mi].mean()),
            'REC_minus_GT_mean': float(delta.mean()),
            'paired_training_seed_delta_sample_SD': float(delta.std(ddof=1)),
            'fixed_three_seeds_route_cluster_bootstrap_95_CI':
                np.quantile(conditional[:, mi], [.025, .975]).tolist(),
            'paired_seed_and_route_cluster_bootstrap_95_CI':
                np.quantile(hierarchical[:, mi], [.025, .975]).tolist(),
        }
        if metric in ('success', 'oracle_success'):
            result[metric]['mean_delta_percentage_points'] = float(delta.mean() * 100)
            result[metric]['joint_95_CI_percentage_points'] = (
                100 * np.quantile(hierarchical[:, mi], [.025, .975])).tolist()
    return {'episodes': total, 'exact_geometry_route_clusters': len(counts), 'metrics': result}


def load_evaluation(root, folder, official):
    """Reject incomplete/base/historical results before statistical aggregation."""
    from final_policy_checkpoint_contract_v2 import verify_final_training
    from sampler_seed_contract_v2 import verify_sampler_seed
    assert folder.is_relative_to(root / 'eval')
    plan = read(folder / 'plan.json')
    assert plan['kind'] == 'corrected_final_checkpoint_navigation', 'Base or historical model rejected'
    assert set(plan['groups']) == set(GROUPS), 'Use the matched primary joint-boundary pair'
    assert read(folder / 'controller_exit.json')['success'], 'Complete evaluation required'
    assert not (folder / 'failure.json').exists()
    assert plan['temperature'] == .2 and plan['sampling_seed'] == 10 and plan['max_actions'] == 500
    for name, digest in plan['runtime_source_sha256'].items():
        assert sha(root / 'source' / name) == digest
    qualifications, configs, rows = {}, {}, {}
    for group in GROUPS:
        _, _, qualifications[group] = verify_final_training(root, plan, group)
        configs[group] = read(root / plan['groups'][group]['training_config_relative'])
        assert sha(root/'source/sampler_seed_contract_v2.py')==configs[group]['sampler_contract_sha256']
        verify_sampler_seed(root,configs[group])
    assert recipe(configs[GROUPS[0]]) == recipe(configs[GROUPS[1]]), 'Unmatched training recipe'
    seed = configs[GROUPS[0]]['seed']
    assert seed == configs[GROUPS[1]]['seed'] and seed in SEEDS
    outcomes = read(folder / 'all_worker_results.json')
    expected_workers = {(group, split, gpu, shard) for group in GROUPS
                        for split in ('val_seen', 'val_unseen')
                        for shard, gpu in enumerate(plan['GPU_indices'])}
    assert len(outcomes) == len(expected_workers)
    assert {(v['group'], v['split'], v['gpu'], v['shard']) for v in outcomes} == expected_workers
    assert all(v['returncode'] == 0 and v['mode'] == 'shard' for v in outcomes)
    final = read(folder / 'final_metrics.json')
    assert final['checkpoint_qualification'] == plan['groups']
    for split, episodes in official.items():
        assert plan['splits'][split]['sha256'] == sha(episodes['path'])
        expected_ids = {str(ep['episode_id']) for ep in episodes['episodes']}
        assert set(plan['episode_ids'][split]) == expected_ids
        rows[split] = {}
        for group in GROUPS:
            files = sorted((folder / group / 'attempt-001').glob('shard-' + split + '-gpu*-shard*/episodes/*.json'))
            records = {path.stem: read(path) for path in files}
            assert len(files) == len(records) == len(expected_ids) and set(records) == expected_ids
            shards = list((folder / group / 'attempt-001').glob('shard-' + split + '-gpu*-shard*'))
            assert len(shards) == len(plan['GPU_indices'])
            for shard in shards:
                process = read(shard / 'process.json')
                assert process['control_kind'] == plan['kind'] and not process['no_navigation_finetuning']
                assert process['training_qualification'] == qualifications[group]
                assert process['sampling_seed'] == plan['sampling_seed']
                assert process['original_split_sha256'] == plan['splits'][split]['sha256']
                assert read(shard / 'exit.json')['success'] and read(shard / 'validated_metrics.json')['passed']
            summary = final['models'][group][split]
            assert summary['episodes'] == len(records) and summary['exact_official_ID_coverage']
            for key in METRICS:
                assert abs(summary['metrics'][key] - sum(float(v[key]) for v in records.values()) / len(records)) < 1e-12
            rows[split][group] = records
    return seed, rows, configs, plan, qualifications


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, action='append', required=True)
    parser.add_argument('--official-directory', type=Path, default=Path(
        '/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official/eval/official_annotations'))
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--analysis-seed', type=int, default=20261022)
    args = parser.parse_args()
    root = args.root.resolve()
    assert len(args.evaluation_root) == 3, 'Three distinct paired training seeds are required'
    destination = root / 'report/final_paired_policy_uncertainty_v3.json'
    assert not destination.exists(), 'Preserve the existing analysis'
    official = {}
    for split, expected in [('val_seen', 333), ('val_unseen', 531)]:
        path = args.official_directory / (split + '.json')
        episodes = read(path)['episodes']
        assert len(episodes) == len({str(ep['episode_id']) for ep in episodes}) == expected
        official[split] = {'path': path, 'episodes': episodes}
    by_seed, provenance, common_recipe, runtime, gpus = {}, [], None, None, None
    for folder in args.evaluation_root:
        folder = folder.resolve()
        seed, rows, configs, plan, qualifications = load_evaluation(root, folder, official)
        assert seed not in by_seed, 'Duplicate training seed'
        current = recipe(configs[GROUPS[0]], across_seeds=True)
        if common_recipe is None:
            common_recipe, runtime, gpus = current, plan['runtime_source_sha256'], plan['GPU_indices']
        assert current == common_recipe and runtime == plan['runtime_source_sha256'] and gpus == plan['GPU_indices']
        by_seed[seed] = rows
        files = sorted(path for group in GROUPS for path in (folder / group / 'attempt-001').glob('shard-*/episodes/*.json'))
        provenance.append({'training_seed': seed, 'evaluation_root': str(folder),
                           'plan_sha256': sha(folder / 'plan.json'),
                           'final_metrics_sha256': sha(folder / 'final_metrics.json'),
                           'episode_records_sha256': hashlib.sha256(b''.join(
                               str(path.relative_to(folder)).encode() + b'\0' + path.read_bytes() + b'\0' for path in files)).hexdigest(),
                           'training_qualifications': qualifications})
    assert set(by_seed) == set(SEEDS)
    results = {}
    for split, source in official.items():
        counts, values = aggregate(source['episodes'], {seed: by_seed[seed][split] for seed in SEEDS})
        results[split] = summarize(counts, values, args.bootstrap_repetitions, args.analysis_seed)
        results[split]['official_split_sha256'] = sha(source['path'])
    result = {'scope': 'Three qualified final paired GT/REC training seeds; complete official navigation.',
              'primary_split': 'val_unseen', 'primary_metric': 'success',
              'training_seeds': list(SEEDS), 'navigation_sampling_seed': 10,
              'bootstrap_repetitions': args.bootstrap_repetitions, 'analysis_seed': args.analysis_seed,
              'estimand': 'Mean over paired training seeds of episode-weighted REC minus GT metrics.',
              'bootstrap_unit': 'Exact geometry route with all its instructions; GT/REC always paired. Joint interval also resamples paired training seeds. Same route draws span all seed pairs.',
              'limits': ['Three seeds give limited precision for seed variation.',
                         'Only one navigation sampling seed is evaluated; sampling variability is not separately estimated.',
                         'Intervals are descriptive, pointwise and not adjusted for multiple metrics.',
                         'No noninferiority margin or superiority claim is automatically inferred.'],
              'provenance': provenance, 'splits': results, 'source_sha256': sha(Path(__file__)),
              'new_jobs': 0, 'created_at': time.time()}
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(destination)


if __name__ == '__main__':
    main()
