"""Evaluate qualified final models: one-episode pilots, then exact full splits."""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--evaluation-root', required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
E = Path(a.evaluation_root).resolve()
assert E.is_relative_to(R / 'eval')
lock = (E / 'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (E / 'controller_exit.json').exists()
plan = json.loads((E / 'plan.json').read_text())
assert plan['kind'] == 'corrected_final_checkpoint_navigation'
cards = plan['GPU_indices']
groups = list(plan['groups'])
assert cards and len(cards) == len(set(cards)) and all(0 <= gpu < 5 for gpu in cards)

def save(path, value):
    q = path.with_suffix(path.suffix + '.partial')
    q.write_text(json.dumps(value, indent=2, allow_nan=False))
    q.replace(path)

def run(group, split, gpu, shard, mode):
    tag = '%s-%s-%s-shard%d' % (mode, group, split, shard)
    command = [sys.executable, str(R / 'source/final_policy_worker_v2.py'),
               '--experiment', str(R), '--evaluation-root', str(E), '--group', group,
               '--attempt', 'attempt-001', '--mode', mode, '--split', split,
               '--gpu', str(gpu), '--shard', str(shard), '--shards', str(len(cards))]
    env = dict(os.environ, PAPER_EXPERIMENT_ROOT=str(R), PYTHONPATH=str(R / 'source'))
    with (E / 'logs' / (tag + '.log')).open('x') as stream:
        child = subprocess.Popen(command, cwd='/tmp', env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(E / 'live' / ('gpu-%d.json' % gpu), {'pid': child.pid, 'command': command,
                                                 'group': group, 'split': split, 'mode': mode,
                                                 'shard': shard, 'time': time.time()})
        code = child.wait()
    result = {'group': group, 'split': split, 'gpu': gpu, 'shard': shard,
              'mode': mode, 'returncode': code, 'time': time.time()}
    save(E / 'finished' / (tag + '.json'), result)
    return result

save(E / 'controller_process.json', {'pid': os.getpid(), 'started_at': time.time()})
success = False
try:
    for name, digest in plan['runtime_source_sha256'].items():
        assert hashlib.sha256((R / 'source' / name).read_bytes()).hexdigest() == digest
    from final_policy_checkpoint_contract_v2 import verify_final_training
    for group in groups:
        verify_final_training(R, plan, group)
    save(E / 'phase.json', {'phase': 'single_episode_pilots', 'time': time.time()})
    pilots = [run(group, 'val_seen', cards[0], 0, 'pilot') for group in groups]
    save(E / 'pilot_results.json', pilots)
    assert all(x['returncode'] == 0 for x in pilots), 'Pilot execution failed; preserve navigation failures and inspect technical logs.'
    save(E / 'phase.json', {'phase': 'full_official_navigation', 'time': time.time()})
    def lane(item):
        shard, gpu = item
        values = []
        for split in ['val_seen', 'val_unseen']:
            for group in groups:
                result = run(group, split, gpu, shard, 'shard')
                values.append(result)
                if result['returncode'] != 0:
                    return values
        return values
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards)) as pool:
        lanes = list(pool.map(lane, enumerate(cards)))
    outcomes = [x for lane_values in lanes for x in lane_values]
    save(E / 'all_worker_results.json', outcomes)
    assert len(outcomes) == 2 * len(groups) * len(cards) and all(x['returncode'] == 0 for x in outcomes)
    final = {}
    for group in groups:
        final[group] = {}
        for split, expected in [('val_seen', 333), ('val_unseen', 531)]:
            paths = list((E / group / 'attempt-001').glob('shard-%s-gpu*-shard*/episodes/*.json' % split))
            records = {p.stem: json.loads(p.read_text()) for p in paths}
            assert len(paths) == len(records) == expected and set(records) == set(plan['episode_ids'][split])
            for process in (E / group / 'attempt-001').glob('shard-%s-gpu*-shard*/exit.json' % split):
                assert json.loads(process.read_text())['success']
            keys = ['success', 'oracle_success', 'distance_to_goal', 'ndtw', 'sdtw', 'path_length', 'steps_taken']
            final[group][split] = {'episodes': expected, 'exact_official_ID_coverage': True,
                                   'success_count': sum(r['success'] for r in records.values()),
                                   'oracle_success_count': sum(r['oracle_success'] for r in records.values()),
                                   'metrics': {key: sum(float(r[key]) for r in records.values()) / expected for key in keys}}
    save(E / 'final_metrics.json', {'models': final, 'temperature': .2, 'sampling_seed': 10,
                                   'checkpoint_qualification': plan['groups'], 'time': time.time()})
    success = True
except BaseException:
    save(E / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
    raise
finally:
    save(E / 'controller_exit.json', {'success': success, 'new_paid_jobs': 0, 'time': time.time()})
