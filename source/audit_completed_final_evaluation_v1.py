"""Bind an actually completed single-model evaluation to all original proofs.

Rechecks both split audits and the recorded parser replay's log hashes. This
does not run a policy, alter the protocol, or create completion from counts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

from final_policy_checkpoint_contract_v2 import verify_final_training
from sampler_seed_contract_v2 import personal_path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--evaluation-root', type=Path, required=True)
    cli.add_argument('--group', choices=['gt-joint-boundary', 'rec-joint-boundary'], required=True)
    cli.add_argument('--seen-audit', type=Path, required=True)
    cli.add_argument('--unseen-audit', type=Path, required=True)
    cli.add_argument('--format-audit', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    root, evaluation, output = args.root.resolve(), args.evaluation_root.resolve(), args.output.resolve()
    assert evaluation.is_relative_to(root / 'eval') and output.is_relative_to(root / 'report')
    audits = {'val_seen': args.seen_audit.resolve(), 'val_unseen': args.unseen_audit.resolve()}
    assert all(path.is_relative_to(root / 'report') for path in [*audits.values(), args.format_audit.resolve()])
    if not (evaluation / 'controller_exit.json').exists():
        print(json.dumps({'verified': False, 'phase': 'awaiting_actual_controller_exit', 'report_created': False}))
        return 3
    assert read(evaluation / 'controller_exit.json')['success'], 'Actual controller failed'
    assert not (evaluation / 'failure.json').exists()
    plan = read(evaluation / 'plan.json')
    group = args.group
    assert plan['kind'] == 'corrected_final_checkpoint_navigation' and set(plan['groups']) == {group}
    assert plan['GPU_indices'] == [0, 1, 2, 3, 4]
    checkpoint, _, qualification = verify_final_training(root, plan, group)
    for name, digest in plan['runtime_source_sha256'].items():
        assert sha(root / 'source' / name) == digest, name
    outcomes = read(evaluation / 'all_worker_results.json')
    expected = {(split, shard, shard) for split in audits for shard in range(5)}
    assert len(outcomes) == 10
    assert {(row['split'], row['gpu'], row['shard']) for row in outcomes} == expected
    assert all(row['group'] == group and row['mode'] == 'shard' and row['returncode'] == 0 for row in outcomes)
    pilots = read(evaluation / 'pilot_results.json')
    assert len(pilots) == 1 and pilots[0]['group'] == group
    assert pilots[0]['mode'] == 'pilot' and pilots[0]['returncode'] == 0
    final = read(evaluation / 'final_metrics.json')
    assert final['checkpoint_qualification'] == plan['groups']
    assert final['temperature'] == .2 and final['sampling_seed'] == 10
    splits = {}
    for split, audit in audits.items():
        subprocess.run([sys.executable, str(root / 'source/audit_completed_final_split_v1.py'),
                        '--root', str(root), '--evaluation-root', str(evaluation),
                        '--group', group, '--split', split, '--output', str(audit)], check=True)
        proof = read(audit)
        assert proof['verified'] and proof['evaluation_relative'] == str(evaluation.relative_to(root))
        assert proof['group'] == group and proof['training_seed'] == plan['groups'][group]['training_seed']
        summary = final['models'][group][split]
        assert summary['episodes'] == proof['episodes'] and summary['exact_official_ID_coverage']
        assert summary['success_count'] == proof['success_count']
        assert summary['oracle_success_count'] == proof['oracle_success_count']
        for key, value in proof['metrics'].items():
            assert math.isclose(value, summary['metrics'][key], rel_tol=1e-12, abs_tol=1e-12)
        splits[split] = {'audit': str(audit.relative_to(root)), 'audit_sha256': sha(audit),
                         'episodes': proof['episodes'], 'success_count': proof['success_count'],
                         'oracle_success_count': proof['oracle_success_count'], 'metrics': proof['metrics']}
    assert sum(split['episodes'] for split in splits.values()) == 864
    format_path = args.format_audit.resolve()
    formatting = read(format_path)
    assert personal_path(root, formatting['evaluation_root']) == evaluation
    assert formatting['complete_evaluation_verified'] and formatting['parser_mismatch_count'] == 0
    assert formatting['kind'] == plan['kind'] and formatting['evaluation_plan_sha256'] == sha(evaluation / 'plan.json')
    assert formatting['eval_py_sha256'] == plan['eval_py_sha256']
    assert formatting['source_sha256'] == sha(root / 'source/audit_policy_output_format_v1.py')
    expected_workers = {(split, 'shard-%s-gpu%d-shard%d' % (split, gpu, gpu)) for split in audits for gpu in range(5)}
    workers = formatting['workers']
    assert len(workers) == 10 and {(w['split'], w['worker']) for w in workers} == expected_workers
    generations = recognized = unrecognized = 0
    for worker in workers:
        assert worker['group'] == group
        folder = evaluation / group / 'attempt-001' / worker['worker']
        assert sha(folder / 'client.log') == worker['client_log_snapshot_sha256']
        assert (folder / 'client.log').stat().st_size == worker['client_log_bytes']
        assert sha(folder / 'process.json') == worker['process_sha256']
        counts = worker['counts']
        calls = counts['generation_calls']
        valid = counts.get('recognized_motion', 0) + counts.get('recognized_Stop', 0)
        invalid = counts.get('unmatched_digits_default_right', 0) + counts.get('unmatched_returns_None', 0)
        assert calls == valid + invalid and calls > 0
        generations += calls
        recognized += valid
        unrecognized += invalid
    for split, expected_count in [('val_seen', 333), ('val_unseen', 531)]:
        item = formatting['groups'][group][split]
        assert item['exact_official_ID_coverage'] and item['completed_episodes'] == expected_count
        assert item['counts']['generation_calls'] == sum(w['counts']['generation_calls'] for w in workers if w['split'] == split)
    cfg = read(root / plan['groups'][group]['training_config_relative'])
    run = personal_path(root, cfg['run_root'])
    manifest_path = run / 'debug/final_tensor_checksums.json'
    manifest = read(manifest_path)
    assert manifest['training_config_sha256'] == plan['groups'][group]['training_config_sha256']
    assert manifest['completion_receipt_sha256'] == plan['groups'][group]['completion_receipt_sha256']
    assert personal_path(root, manifest['checkpoint']) == checkpoint
    assert manifest['group'] == group and manifest['training_seed'] == plan['groups'][group]['training_seed']
    bundle_files = {row['relative_path']: row['bytes'] for row in qualification['checkpoint_bundle']['files']}
    assert len(manifest['tensor_files']) == len(bundle_files) == 6
    assert {row['relative_path']: row['bytes'] for row in manifest['tensor_files']} == bundle_files
    assert all(len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']) for row in manifest['tensor_files'])
    for name, digest in manifest['metadata_sha256'].items():
        assert sha(checkpoint / name) == digest
    stable = {'verified': True, 'evaluation_relative': str(evaluation.relative_to(root)),
              'group': group, 'training_seed': plan['groups'][group]['training_seed'],
              'episodes': 864, 'splits': splits, 'all10workers_returned_zero': True,
              'controller_exit_sha256': sha(evaluation / 'controller_exit.json'),
              'all_worker_results_sha256': sha(evaluation / 'all_worker_results.json'),
              'final_metrics_sha256': sha(evaluation / 'final_metrics.json'),
              'evaluation_plan_sha256': sha(evaluation / 'plan.json'),
              'pilot_results_sha256': sha(evaluation / 'pilot_results.json'),
              'final_tensor_manifest_sha256': sha(manifest_path),
              'tensor_hash_scope': 'Bound existing full-byte checksum receipt; current tensor headers/index/sizes and metadata rechecked. Tensor bytes are not rehashed by this evaluation audit.',
              'output_format_audit': str(format_path.relative_to(root)), 'output_format_audit_sha256': sha(format_path),
              'generation_calls': generations, 'recognized_generation_calls': recognized,
              'unrecognized_generation_calls': unrecognized, 'parser_mismatch_count': 0,
              'audit_source_sha256': sha(Path(__file__)),
              'scope': 'One final model on exact Seen333 and Unseen531. Paired comparisons, all training seeds, final uncertainty and final paper are separate requirements.',
              'goal_complete': False, 'protocol_or_model_modified': False, 'jobs_submitted_or_restarted': 0}
    if output.exists():
        previous = read(output)
        assert all(previous[key] == value for key, value in stable.items()), 'Existing immutable audit differs'
        print(json.dumps({'verified': True, 'already_verified': True, 'report': str(output)}))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump({**stable, 'verified_at': time.time()}, stream, indent=2, allow_nan=False)
    print(json.dumps({'verified': True, 'report': str(output), 'episodes': 864, 'splits': splits,
                      'generation_calls': generations, 'unrecognized_generation_calls': unrecognized}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
