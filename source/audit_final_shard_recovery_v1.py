"""Independently bind eight retained and two recovered real successful shards.

Read-only: no evaluation, scoring changes, new controller receipts or retries.
The incomplete gate returns3 and never creates an output.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    R, output = a.root.resolve(), a.output.resolve()
    assert output.is_relative_to(R / 'report')
    E = R / 'eval/final_C4_rec_seed12_002'
    O = R / 'eval/final_C4_rec_seed12_001'
    G = 'rec-joint-boundary'
    C = R / 'report/final_REC_seed12_complete_evaluation_verified_002.json'
    if not all(p.exists() for p in [E / 'controller_exit.json', E / 'recovery_composition_verified.json', C]):
        print(json.dumps({'verified': False, 'phase': 'awaiting_actual_recovery_controller_and_frozen_whole_model_audit',
            'report_created': False, 'no_policy_or_worker_changes': True}))
        return 3
    assert read(E / 'controller_exit.json')['success'] and not (E / 'failure.json').exists()
    proof = read(C)
    assert proof['verified'] and proof['episodes'] == 864 and proof['all10workers_returned_zero']
    assert proof['evaluation_relative'] == str(E.relative_to(R))
    assert proof['controller_exit_sha256'] == sha(E / 'controller_exit.json')
    recovery, composition = read(E / 'recovery_plan.json'), read(E / 'recovery_composition_verified.json')
    oldq, newq = [read(R / f'pipeline/final_navigation_sequence_{n}/plan.json') for n in ['001', '002']]
    assert all(newq['source_sha256'][k] == h == sha(R / 'source' / k) for k, h in oldq['source_sha256'].items())
    assert newq['models'][:5] == oldq['models'][:5]
    assert newq['models'][5]['evaluation_relative'] == str(E.relative_to(R))
    for k, h in newq['source_sha256'].items(): assert sha(R / 'source' / k) == h
    assert recovery['orchestration_source_sha256'] == sha(R / recovery['orchestration_source'])
    assert composition['source_sha256'] == recovery['orchestration_source_sha256']
    assert composition['recovery_plan_sha256'] == sha(E / 'recovery_plan.json')
    assert sha(E / 'plan.json') == sha(O / 'plan.json') == recovery['original_plan_sha256']
    assert recovery['healthy_GPU_lanes_retained'] == [0, 2, 3, 4]
    assert recovery['replay_GPU'] == 1 and recovery['excluded_failed_Seen_partial_records'] == 42
    assert composition['success'] and composition['retained_original_successful_shards'] == 8
    assert composition['new_successful_shards'] == 2 and composition['excluded_failed_Seen_partial_records'] == 42
    assert not read(R / 'pipeline/final_navigation_sequence_001/exit.json')['success']
    assert (O / 'controller_exit.json').exists() and not read(O / 'controller_exit.json')['success']
    failed = O / G / 'attempt-001/shard-val_seen-gpu1-shard1'
    assert not read(failed / 'exit.json')['success'] and read(failed / 'exit.json')['client_returncode'] == -15
    P = R / recovery['preserved_failed_shard_manifest']
    assert sha(P) == recovery['preserved_failed_shard_manifest_sha256']
    saved = read(P)
    assert saved['completed_records_preserved'] == 42
    for rel, h in saved['files_sha256'].items(): assert sha(P.parent / rel) == h
    partial_ids = {p.stem for p in (P.parent / 'episodes').glob('*.json')}
    assert len(partial_ids) == 42 and partial_ids <= set(read(failed / 'process.json')['assigned_episode_ids'])
    assert read(E / 'pilot_results.json') == read(O / 'pilot_results.json')
    pilot = E / G / 'attempt-001/pilot-val_seen-gpu0-shard0'
    assert pilot.is_symlink() and pilot.resolve() == O / G / 'attempt-001' / pilot.name
    assert read(pilot / 'exit.json')['success'] and read(pilot / 'exit.json')['client_returncode'] == 0
    assert sha(pilot / 'process.json') == recovery['original_pilot_process_sha256']
    plan, final = read(E / 'plan.json'), read(E / 'final_metrics.json')
    assert (plan['temperature'], plan['sampling_seed'], plan['max_actions']) == (.2, 10, 500)
    from final_policy_checkpoint_contract_v2 import verify_final_training
    checkpoint, _, qualification = verify_final_training(R, plan, G)
    assert qualification['training_seed'] == 12
    for x in recovery['healthy_initial_sources']:
        d = O / G / 'attempt-001' / f"shard-val_seen-gpu{x['gpu']}-shard{x['gpu']}"
        assert sha(d / 'process.json') == x['original_Seen_process_sha256']
    source_rows = {(x['split'], x['gpu']): x for x in composition['shards']}
    assert set(source_rows) == {(s, g) for s in ['val_seen', 'val_unseen'] for g in range(5)}
    bindings, totals = {}, {}
    def bind(p): bindings[str(p.relative_to(R))] = sha(p)
    for p in [P, C, E / 'recovery_plan.json', E / 'recovery_composition_verified.json', E / 'controller_exit.json',
              E / 'plan.json', E / 'final_metrics.json', E / 'all_worker_results.json', O / 'controller_exit.json',
              R / 'pipeline/final_navigation_sequence_002/plan.json']:
        bind(p)
    keys = ['success', 'oracle_success', 'distance_to_goal', 'ndtw', 'sdtw', 'path_length', 'steps_taken']
    for split, count in [('val_seen', 333), ('val_unseen', 531)]:
        rows = {}
        audit_path = R / proof['splits'][split]['audit']
        assert sha(audit_path) == proof['splits'][split]['audit_sha256']
        split_proof = read(audit_path)
        assert split_proof['exact_official_ID_coverage'] and split_proof['episodes'] == count
        for rel, h in split_proof['provenance_sha256'].items(): assert sha(R / rel) == h
        for gpu in range(5):
            D = E / G / 'attempt-001' / f'shard-{split}-gpu{gpu}-shard{gpu}'
            F = E / 'finished' / f'shard-{G}-{split}-shard{gpu}.json'
            if gpu == 1:
                assert not D.is_symlink() and not F.is_symlink()
                assert D.resolve() == D and D != failed
            else:
                assert D.is_symlink() and D.resolve() == O / G / 'attempt-001' / D.name
                assert F.is_symlink() and F.resolve() == O / 'finished' / F.name
            pr, ex, finish = read(D / 'process.json'), read(D / 'exit.json'), read(F)
            assert ex['success'] and ex['client_returncode'] == finish['returncode'] == 0
            assert pr['gpu'] == gpu and pr['split'] == split and pr['sampling_seed'] == 10
            assert pr['checkpoint'] == str(checkpoint) and pr['training_qualification']['training_seed'] == 12
            assert pr['eval_py_sha256'] == plan['eval_py_sha256']
            assigned = set(plan['episode_ids'][split][gpu::5])
            # The plan preserves original list ordering, and the unchanged
            # frozen worker applies that exact stride before seeding inference.
            assert set(pr['assigned_episode_ids']) == assigned
            paths = list((D / 'episodes').glob('*.json'))
            selected = {p.stem: read(p) for p in paths}
            assert set(selected) == assigned and not set(rows).intersection(selected)
            rows.update(selected)
            src = source_rows[split, gpu]
            assert src['retained_original'] == (gpu != 1)
            assert src['real_source_directory'] == str(D.resolve().relative_to(R))
            assert src['process_sha256'] == sha(D / 'process.json')
            assert src['exit_sha256'] == sha(D / 'exit.json') and src['actual_finished_result_sha256'] == sha(F)
            assert src['episode_sha256'] == {p.name: sha(p) for p in paths}
            for p in [D / 'process.json', D / 'exit.json', D / 'validated_metrics.json', F]: bind(p)
        assert len(rows) == count and set(rows) == set(plan['episode_ids'][split])
        for key in keys:
            mean = sum(float(row[key]) for row in rows.values()) / count
            assert math.isfinite(mean)
            assert math.isclose(mean, final['models'][G][split]['metrics'][key], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(mean, split_proof['metrics'][key], rel_tol=1e-12, abs_tol=1e-12)
        totals[split] = {'exact_original_IDs': count, 'successes': sum(r['success'] for r in rows.values()),
            'retained_original_shards': 4, 'recovered_shards': 1}
        bind(audit_path)
    stable = {'verified': True, 'evaluation_relative': str(E.relative_to(R)), 'episodes': 864,
        'retained_original_successful_shards': 8, 'new_actual_successful_workers': 2,
        'excluded_failed_partial_records': 42, 'original_pilot_retained_and_technically_successful': True,
        'original17_frozen_sources_unchanged': True, 'all10_selected_actual_workers_zero_exit': True,
        'healthy_tasks_restarted': False, 'original_source_artifacts_modified': False,
        'no_partial_prefix_splicing_or_outcome_selection': True,
        'split_summaries': totals, 'bound_files_sha256': bindings, 'source_sha256': sha(Path(__file__)),
        'scope': 'Independent full provenance audit of the explicit recovery composition and all864 actual selected episode files. No new evaluation or bootstrap; no scoring recomputation.',
        'goal_complete': False}
    if output.exists():
        old = read(output)
        assert all(old[k] == v for k, v in stable.items())
    else:
        with output.open('x') as f: json.dump({**stable, 'verified_at': time.time()}, f, indent=2, allow_nan=False)
    print(json.dumps({'verified': True, 'episodes': 864, 'report': str(output)}))
    return 0

if __name__ == '__main__': raise SystemExit(main())
