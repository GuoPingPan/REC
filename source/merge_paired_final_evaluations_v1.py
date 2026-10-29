"""Compose completed independent GT/REC evaluations without rerunning episodes.

The composite explicitly records both real controllers and original per-episode
files. Publication requires the same final-model contract as the analysis tool.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

from final_policy_checkpoint_contract_v2 import verify_final_training
from sampler_seed_contract_v2 import personal_path

GROUPS = ('gt-joint-boundary', 'rec-joint-boundary')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gt-evaluation', type=Path, required=True)
    parser.add_argument('--rec-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    sources = dict(zip(GROUPS, [args.gt_evaluation.resolve(), args.rec_evaluation.resolve()]))
    assert output.is_relative_to(root / 'eval') and not output.exists()
    plans, finals, provenance = {}, {}, {}
    results, pilots = [], []
    for group, folder in sources.items():
        assert folder.is_relative_to(root / 'eval')
        plan = read(folder / 'plan.json')
        assert set(plan['groups']) == {group}
        assert read(folder / 'controller_exit.json')['success']
        assert not (folder / 'failure.json').exists()
        verify_final_training(root, plan, group)
        for name, digest in plan['runtime_source_sha256'].items():
            assert sha(root / 'source' / name) == digest
        final = read(folder / 'final_metrics.json')
        assert final['checkpoint_qualification'] == plan['groups']
        outcomes = read(folder / 'all_worker_results.json')
        expected = {(split, gpu, shard) for split in ['val_seen', 'val_unseen']
                    for shard, gpu in enumerate(plan['GPU_indices'])}
        assert len(outcomes) == len(expected)
        assert {(x['split'], x['gpu'], x['shard']) for x in outcomes} == expected
        assert all(x['group'] == group and x['mode'] == 'shard' and x['returncode'] == 0 for x in outcomes)
        for split, count in [('val_seen', 333), ('val_unseen', 531)]:
            expected_ids = set(plan['episode_ids'][split])
            files = list((folder / group / 'attempt-001').glob('shard-%s-gpu*-shard*/episodes/*.json' % split))
            rows = {p.stem: read(p) for p in files}
            assert len(files) == len(rows) == count and set(rows) == expected_ids
            summary = final['models'][group][split]
            assert summary['episodes'] == count and summary['exact_official_ID_coverage']
            for key in ['success', 'oracle_success', 'distance_to_goal', 'ndtw', 'sdtw', 'path_length', 'steps_taken']:
                assert abs(summary['metrics'][key] - sum(float(row[key]) for row in rows.values()) / count) < 1e-12
        plans[group], finals[group] = plan, final
        provenance[group] = {'evaluation_root': str(folder),
                             'source_files_sha256': {name: sha(folder / name) for name in
                                 ['plan.json', 'controller_exit.json', 'final_metrics.json', 'all_worker_results.json']}}
        results.extend(outcomes)
        pilots.extend(read(folder / 'pilot_results.json'))
    first, second = [plans[group] for group in GROUPS]
    for key in ['kind', 'source_manifest_sha256', 'runtime_source_sha256', 'eval_py_sha256',
                'temperature', 'sampling_seed', 'max_actions', 'GPU_indices', 'splits', 'episode_ids']:
        assert first[key] == second[key], ('Evaluation protocol mismatch', key)
    seed = first['groups'][GROUPS[0]]['training_seed']
    assert second['groups'][GROUPS[1]]['training_seed'] == seed
    plan = dict(first)
    plan['groups'] = {group: plans[group]['groups'][group] for group in GROUPS}
    plan.update(scope='Composite of two completed independent final-model evaluations; original episode artifacts are linked without rewriting.',
                composition_source_sha256=sha(Path(__file__)), composed_at=time.time(),
                original_controllers=provenance)
    destination = output
    output = destination.with_name('.compose-' + destination.name + '-' + str(time.time_ns()))
    output.mkdir()
    def save(name, value):
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    save('plan.json', plan)
    for group, folder in sources.items():
        (output / group).symlink_to(folder / group, target_is_directory=True)
    save('all_worker_results.json', results)
    save('pilot_results.json', pilots)
    save('final_metrics.json', {'models': {group: finals[group]['models'][group] for group in GROUPS},
                               'checkpoint_qualification': plan['groups'], 'temperature': .2,
                               'sampling_seed': 10, 'composite_provenance': provenance, 'time': time.time()})
    save('controller_exit.json', {'success': True, 'kind': 'validated_composition_of_completed_controllers',
                                  'original_controllers': provenance, 'time': time.time()})
    # The unchanged statistical loader rechecks rank/checkpoint/episode semantics.
    from analyze_final_paired_uncertainty_v3 import load_evaluation
    common = personal_path(root, '/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
    official = {}
    for split in ['val_seen', 'val_unseen']:
        path = common / 'eval/official_annotations' / (split + '.json')
        official[split] = {'path': path, 'episodes': read(path)['episodes']}
    try:
        load_evaluation(root, output, official)
    except BaseException:
        save('failure.json', {'stage': 'composition_statistical_loader_validation', 'time': time.time()})
        raise
    save('composition_verified.json', {'success': True, 'training_seed': seed,
                                      'source_sha256': sha(Path(__file__)), 'time': time.time()})
    assert not destination.exists()
    output.rename(destination)
    print(json.dumps({'composed': str(destination), 'training_seed': seed, 'original_controllers': provenance}))


if __name__ == '__main__':
    main()
