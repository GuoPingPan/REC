"""Audit one complete official split without stopping the running other split."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
from final_policy_checkpoint_contract_v2 import verify_final_training
from sampler_seed_contract_v2 import personal_path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--evaluation-root', type=Path, required=True)
    ap.add_argument('--group', choices=['gt-joint-boundary', 'rec-joint-boundary'], required=True)
    ap.add_argument('--split', choices=['val_seen', 'val_unseen'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    R, E, output = a.root.resolve(), a.evaluation_root.resolve(), a.output.resolve()
    assert E.is_relative_to(R / 'eval') and output.is_relative_to(R / 'report')
    plan = read(E / 'plan.json')
    assert plan['kind'] == 'corrected_final_checkpoint_navigation' and a.group in plan['groups']
    assert plan['GPU_indices'] == [0, 1, 2, 3, 4]
    expected_count = {'val_seen': 333, 'val_unseen': 531}[a.split]
    folders = [E / a.group / 'attempt-001' / ('shard-%s-gpu%d-shard%d' % (a.split, gpu, shard))
               for shard, gpu in enumerate(plan['GPU_indices'])]
    finished = [E / 'finished' / ('shard-%s-%s-shard%d.json' % (a.group, a.split, shard))
                for shard in range(len(folders))]
    if not all((D / 'exit.json').exists() and F.exists() for D, F in zip(folders, finished)):
        print(json.dumps({'verified': False, 'phase': 'awaiting_all_successful_worker_exits',
                          'split': a.split, 'report_created': False}))
        return 3
    checkpoint, _, qualification = verify_final_training(R, plan, a.group)
    for name, digest in plan['runtime_source_sha256'].items():
        assert sha(R / 'source' / name) == digest, name
    common = personal_path(R, '/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
    official = common / 'eval/official_annotations' / (a.split + '.json')
    assert sha(official) == plan['splits'][a.split]['sha256']
    episodes = read(official)['episodes']
    ids = [str(row['episode_id']) for row in episodes]
    assert len(ids) == len(set(ids)) == expected_count
    assert set(ids) == set(plan['episode_ids'][a.split])
    keys = ['success', 'oracle_success', 'distance_to_goal', 'ndtw', 'sdtw', 'path_length', 'steps_taken']
    records, provenance, shard_proofs = {}, {}, []
    def bind(path):
        provenance[str(path.relative_to(R))] = sha(path)
    bind(E / 'plan.json')
    for shard, (D, F) in enumerate(zip(folders, finished)):
        process, status, metrics, result = [read(p) for p in [D / 'process.json', D / 'exit.json', D / 'validated_metrics.json', F]]
        assert status['success'] and status['client_returncode'] == 0 and status['policy_inference_run']
        assert result['returncode'] == 0 and result['group'] == a.group and result['split'] == a.split and result['mode'] == 'shard'
        assert result['gpu'] == result['shard'] == shard
        assert process['group'] == a.group and process['split'] == a.split and process['mode'] == 'shard'
        assert process['gpu'] == shard and process['sampling_seed'] == 10
        assert process['control_kind'] == plan['kind'] and Path(process['checkpoint']).resolve() == checkpoint.resolve()
        assert process['original_split_sha256'] == sha(official)
        command = process['command']
        for flag, value in [('--inference_sample_temperature', '0.2'), ('--maxAction', '500')]:
            assert command[command.index(flag) + 1] == value
        assigned = set(ids[shard::len(folders)])
        assert set(process['assigned_episode_ids']) == assigned
        paths = list((D / 'episodes').glob('*.json'))
        rows = {p.stem: read(p) for p in paths}
        assert len(paths) == len(rows) == len(assigned) and set(rows) == assigned and not set(records).intersection(rows)
        assert metrics['passed'] and metrics['official_eval_source_unchanged']
        assert metrics['episode_count'] == len(rows) and set(metrics['episode_ids']) == assigned
        assert Path(metrics['checkpoint']).resolve() == checkpoint.resolve()
        for eid, row in rows.items():
            assert all(math.isfinite(float(row[key])) for key in keys), eid
            assert row['success'] in [0, 1] and row['oracle_success'] in [0, 1]
            assert all(float(row[key]) >= 0 for key in keys)
        for key in keys:
            mean = sum(float(row[key]) for row in rows.values()) / len(rows)
            assert math.isclose(mean, float(metrics['metrics'][key]), rel_tol=1e-12, abs_tol=1e-12), (shard, key)
        records.update(rows)
        for p in [D / 'process.json', D / 'exit.json', D / 'validated_metrics.json', F, *paths]:
            bind(p)
        shard_proofs.append({'shard': shard, 'gpu': shard, 'episodes': len(rows), 'worker_success': True, 'client_returncode': 0})
    assert len(records) == expected_count and set(records) == set(ids)
    stable = {'evaluation_relative': str(E.relative_to(R)), 'group': a.group,
              'training_seed': plan['groups'][a.group]['training_seed'], 'split': a.split,
              'episodes': len(records), 'exact_official_ID_coverage': True,
              'official_annotations_sha256': sha(official), 'shards': shard_proofs,
              'success_count': sum(row['success'] for row in records.values()),
              'oracle_success_count': sum(row['oracle_success'] for row in records.values()),
              'metrics': {key: sum(float(row[key]) for row in records.values()) / len(records) for key in keys},
              'temperature': .2, 'sampling_seed': 10, 'max_actions': 500,
              'all_observed_episode_metrics_finite': True,
              'checkpoint': str(checkpoint), 'training_qualification': qualification,
              'provenance_sha256': provenance, 'audit_source_sha256': sha(Path(__file__))}
    if output.exists():
        old = read(output)
        assert all(old[key] == value for key, value in stable.items()), 'Existing immutable audit differs'
        print(json.dumps({'verified': True, 'already_verified': True, 'report': str(output)}))
        return 0
    with output.open('x') as stream:
        json.dump({**stable, 'verified': True, 'verified_at': time.time(),
                   'scope': 'This completed official split only. The other split, paired training seeds, uncertainty analysis and final paper remain separate completion requirements.',
                   'evaluation_protocol_or_model_modified': False, 'jobs_submitted_or_restarted': 0},
                  stream, indent=2, allow_nan=False)
    print(json.dumps({'verified': True, 'report': str(output), 'episodes': len(records),
                      'success_count': stable['success_count'], 'metrics': stable['metrics']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
