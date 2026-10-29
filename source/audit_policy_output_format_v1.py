"""Replay the frozen parser on recorded generations without running a policy.

Format percentages are generation-weighted diagnostics, not navigation scores.
Completed evaluations require exact episode coverage and successful workers.
"""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import math
import re
import time

from sampler_seed_contract_v2 import personal_path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--evaluation-root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--require-complete', action='store_true')
    args = cli.parse_args()
    root, evaluation, output = args.root.resolve(), args.evaluation_root.resolve(), args.output.resolve()
    assert evaluation.is_relative_to(root / 'eval') and output.is_relative_to(root / 'report')
    assert not output.exists()
    plan = read(evaluation / 'plan.json')
    assert plan['kind'] in ['unfinetuned_initialization_control', 'corrected_final_checkpoint_navigation']
    terminal = read(evaluation / 'controller_exit.json') if (evaluation / 'controller_exit.json').exists() else None
    complete = bool(terminal and terminal['success'] and not (evaluation / 'failure.json').exists())
    if args.require_complete:
        assert complete, 'A completed final evaluation is required'
    manifest = read(root / 'inventory/strict_release_collator_source.json')
    source = personal_path(root, manifest['source']) / 'eval.py'
    assert sha(source) == manifest['files']['eval.py']
    syntax = ast.parse(source.read_text())
    function = next(n for n in syntax.body if isinstance(n, ast.FunctionDef) and n.name == 'convert_text_to_action')
    namespace = {'math': math, 're': re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    parse_action = namespace['convert_text_to_action']
    assignment = next(n for n in function.body if isinstance(n, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'patterns' for target in n.targets))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), 'exec'), namespace)
    patterns = namespace['patterns']
    generation = re.compile(r'^step (\d+) action_text: (.*?) actions: (\[[^\]\n]*\])\r?$', re.M | re.S)
    groups = list(plan['groups']) if 'groups' in plan else ['base']
    totals, workers = {}, []
    for group in groups:
        by_split = {split: collections.Counter() for split in plan['episode_ids']}
        coverage = {split: set() for split in plan['episode_ids']}
        for folder in sorted((evaluation / group / 'attempt-001').glob('shard-*')):
            process = read(folder / 'process.json')
            split = process['split']
            raw = (folder / 'client.log').read_bytes()
            log = raw.decode(errors='replace')
            count = collections.Counter()
            examples = {}
            for match in generation.finditer(log):
                step, text, reported = match.groups()
                actions = ast.literal_eval(reported)
                actual = parse_action(text)
                assert actions == actual, ('Parser differs from recorded execution', folder, step)
                matched = next((key for key, regex in patterns.items() if regex.search(text)), None)
                if matched is None:
                    category = 'unmatched_digits_default_right' if re.search(r'\d+', text) else 'unmatched_returns_None'
                elif matched == 0:
                    category = 'recognized_Stop'
                else:
                    category = 'recognized_motion'
                count['generation_calls'] += 1
                count[category] += 1
                if category not in examples:
                    examples[category] = {'step': int(step), 'text': text[:2000], 'actions': actions}
            expected = set(process['assigned_episode_ids'])
            files = list((folder / 'episodes').glob('*.json'))
            rows = {p.stem: read(p) for p in files}
            assert len(files) == len(rows) and set(rows) <= expected
            assert not coverage[split].intersection(rows)
            coverage[split].update(rows)
            if complete:
                assert set(rows) == expected
                assert read(folder / 'exit.json')['success']
                assert read(folder / 'validated_metrics.json')['passed']
                assert count['generation_calls'] > 0
            for row in rows.values():
                for key in ['success', 'oracle_success', 'distance_to_goal', 'path_length', 'steps_taken']:
                    assert math.isfinite(float(row[key]))
                count['completed_episodes'] += 1
                count['episodes_with_translation_path'] += int(row['path_length'] > 0)
                count['successes'] += int(row['success'])
                count['oracle_successes'] += int(row['oracle_success'])
                if 'is_collisioned' in row:
                    count['episodes_with_reported_collision_field'] += 1
                    count['reported_collision_episodes'] += int(row['is_collisioned'])
            by_split[split].update(count)
            workers.append({'group': group, 'split': split, 'worker': folder.name,
                            'counts': dict(count), 'examples': examples,
                            'client_log_bytes': len(raw), 'client_log_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
                            'process_sha256': sha(folder / 'process.json')})
        totals[group] = {}
        for split, count in by_split.items():
            if complete:
                assert coverage[split] == set(plan['episode_ids'][split])
            invalid = count['unmatched_digits_default_right'] + count['unmatched_returns_None']
            total = count['generation_calls']
            totals[group][split] = {'counts': dict(count),
                                    'unrecognized_generation_fraction': invalid / total if total else None,
                                    'completed_episodes': len(coverage[split]),
                                    'expected_episodes': len(plan['episode_ids'][split]),
                                    'exact_official_ID_coverage': coverage[split] == set(plan['episode_ids'][split])}
    result = {'evaluation_root': str(evaluation), 'kind': plan['kind'],
              'complete_evaluation_verified': complete, 'groups': totals, 'workers': workers,
              'parser_function_AST_sha256': hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
              'eval_py_sha256': sha(source), 'evaluation_plan_sha256': sha(evaluation / 'plan.json'),
              'source_sha256': sha(Path(__file__)), 'parser_mismatch_count': 0,
              'interpretation': 'Generation-weighted format diagnostics. A base-versus-finetuned navigation difference can reflect both output-format learning and navigation behavior; it cannot alone identify the value of geometric pseudo-labels.',
              'no_policy_or_protocol_modified': True, 'created_at': time.time()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != 'workers'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
