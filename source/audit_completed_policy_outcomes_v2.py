"""Describe recorded outcomes of one qualified completed final policy.

Outcome groups describe fields, not independently identified failure causes.
In the frozen evaluator action_record contains whole proposed action batches,
including proposals that may never execute; its last entry is not a stop proof.
Final done may also change after scoring during the observation depth check.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import time

from sampler_seed_contract_v2 import personal_path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--evaluation-proof', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    root, proof_path, output = args.root.resolve(), args.evaluation_proof.resolve(), args.output.resolve()
    assert proof_path.is_relative_to(root / 'report') and output.is_relative_to(root / 'report')
    proof = read(proof_path)
    assert proof['verified'] and proof['episodes'] == 864 and proof['all10workers_returned_zero']
    evaluation = (root / proof['evaluation_relative']).resolve()
    assert evaluation.is_relative_to(root / 'eval')
    assert read(evaluation / 'controller_exit.json')['success'] and not (evaluation / 'failure.json').exists()
    for name, key in [('controller_exit.json', 'controller_exit_sha256'),
                      ('all_worker_results.json', 'all_worker_results_sha256'),
                      ('final_metrics.json', 'final_metrics_sha256'), ('plan.json', 'evaluation_plan_sha256')]:
        assert sha(evaluation / name) == proof[key]
    group = proof['group']
    plan = read(evaluation / 'plan.json')
    assert set(plan['groups']) == {group}
    manifest = read(root / 'inventory/strict_release_collator_source.json')
    source = personal_path(root, manifest['source'])
    sources = {}
    for name in ['eval.py', 'AirVLN/src/vlnce_src/env.py']:
        digest = sha(source / name)
        assert digest == manifest['files'][name]
        sources[name] = digest
    summaries, examples = {}, {}
    for split, expected in [('val_seen', 333), ('val_unseen', 531)]:
        audit_path = root / proof['splits'][split]['audit']
        assert sha(audit_path) == proof['splits'][split]['audit_sha256']
        audit = read(audit_path)
        assert audit['verified'] and audit['group'] == group and audit['split'] == split
        assert audit['evaluation_relative'] == proof['evaluation_relative']
        counts = collections.Counter()
        proposed_last = collections.Counter()
        buckets, ids = collections.defaultdict(list), set()
        paths = sorted((evaluation / group / 'attempt-001').glob('shard-%s-gpu*-shard*/episodes/*.json' % split))
        for path in paths:
            assert sha(path) == audit['provenance_sha256'][str(path.relative_to(root))]
            row = read(path)
            eid = str(row['episode_id'])
            assert eid == path.stem and eid not in ids
            ids.add(eid)
            assert row['success'] in [0, 1] and row['oracle_success'] in [0, 1]
            assert isinstance(row['is_collisioned'], bool) and isinstance(row['done'], bool)
            assert isinstance(row['action_record'], list)
            assert all(math.isfinite(float(row[key])) for key in ['distance_to_goal', 'SUCCESS_DISTANCE', 'path_length', 'steps_taken'])
            naive_done_distance_success = int(row['done'] and row['distance_to_goal'] <= row['SUCCESS_DISTANCE'])
            success, oracle, collision = bool(row['success']), bool(row['oracle_success']), row['is_collisioned']
            failed = not success
            flags = {'episodes': True, 'success': success, 'oracle_success': oracle, 'failure': failed,
                     'reported_collision': collision, 'success_and_reported_collision': success and collision,
                     'oracle_success_without_final_success': oracle and failed,
                     'failure_with_reported_collision': failed and collision,
                     'failure_without_reported_collision_with_oracle_success': failed and not collision and oracle,
                     'failure_without_reported_collision_without_oracle_success': failed and not collision and not oracle,
                     'zero_translation_path': row['path_length'] == 0,
                     'success_with_zero_translation_path': success and row['path_length'] == 0,
                     'steps_at_least500': row['steps_taken'] >= 500,
                     'failure_final_planar_distance_above_success_radius': failed and row['distance_to_goal'] > row['SUCCESS_DISTANCE'],
                     'record_done_false': not row['done'],
                     'naive_done_distance_formula_mismatch': row['success'] != naive_done_distance_success,
                     'naive_formula_mismatch_with_reported_collision': row['success'] != naive_done_distance_success and collision,
                     'failure_with_final_distance_within_success_radius': failed and row['distance_to_goal'] <= row['SUCCESS_DISTANCE']}
            last = row['action_record'][-1] if row['action_record'] else None
            proposed_last[str(last)] += 1
            for key, value in flags.items():
                counts[key] += int(value)
                if value:
                    buckets[key].append(row)
        assert len(paths) == len(ids) == expected and ids == set(plan['episode_ids'][split])
        assert counts['success'] == proof['splits'][split]['success_count']
        assert counts['failure'] == counts['failure_with_reported_collision'] + counts['failure_without_reported_collision_with_oracle_success'] + counts['failure_without_reported_collision_without_oracle_success']
        assert math.isclose(counts['oracle_success'] / expected, proof['splits'][split]['metrics']['oracle_success'])
        summaries[split] = {'counts': dict(counts), 'last_proposed_action_counts': dict(proposed_last),
                            'exact_official_ID_coverage': True, 'recorded_success_retained_without_recalculation': True,
                            'split_audit_sha256': sha(audit_path)}
        example_keys = ['failure_with_reported_collision', 'failure_without_reported_collision_with_oracle_success',
                        'failure_without_reported_collision_without_oracle_success', 'zero_translation_path', 'steps_at_least500',
                        'naive_done_distance_formula_mismatch']
        examples[split] = {}
        for key in example_keys:
            selected = sorted(buckets[key], key=lambda row: str(row['episode_id']))[:3]
            examples[split][key] = [{**{name: row[name] for name in ['episode_id', 'success', 'oracle_success',
                                     'is_collisioned', 'done', 'distance_to_goal', 'SUCCESS_DISTANCE', 'path_length', 'steps_taken']},
                                    'last_proposed_actions': row['action_record'][-8:]} for row in selected]
    stable = {'verified': True, 'evaluation_relative': proof['evaluation_relative'], 'group': group,
              'training_seed': proof['training_seed'], 'evaluation_proof': str(proof_path.relative_to(root)),
              'evaluation_proof_sha256': sha(proof_path), 'episodes': 864, 'splits': summaries,
              'examples_first3_lexicographic_episode_IDs_per_recorded_group': examples,
              'frozen_scoring_source_sha256': sources, 'source_sha256': sha(Path(__file__)),
              'interpretation': ['Counts describe recorded outcomes and are not causal failure attribution.',
                 'is_collisioned is the preserved online depth-proxy flag, not independent body-contact or swept-volume certification.',
                 'makeActions updates measurements before eval.py calls get_obs; the latter depth check can set is_collisioned/is_end without recomputing Success. The terminal done field is therefore not a substitute for is_end at scoring time. Preserve recorded scores rather than recomputing Success from final done/distance.',
                 'Oracle success without final success means recorded entry into the planar goal region without final success; it does not establish a stop-prediction error.',
                 'action_record appends complete proposed batches before execution. Last proposed action is not necessarily the last executed action.',
                 'steps_at_least500 is a recorded-step threshold, not a causal claim that budget exhaustion terminated the episode.'],
              'protocol_or_model_modified': False, 'jobs_submitted_or_restarted': 0,
              'paired_comparison_or_full_goal_complete': False}
    if output.exists():
        previous = read(output)
        assert all(previous[key] == value for key, value in stable.items()), 'Existing immutable report differs'
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump({**stable, 'created_at': time.time()}, stream, indent=2, allow_nan=False)
    print(json.dumps({'verified': True, 'report': str(output), 'splits': summaries}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
