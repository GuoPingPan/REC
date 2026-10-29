"""One isolated initialization-model evaluation after the existing learning gate.

Uses existing DSW GPUs only. No training submission or automatic retry.
"""
from pathlib import Path
import argparse, concurrent.futures, fcntl, json, os, subprocess, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
E = R/'eval/initialization_policy_control_002'
PLAN = json.loads((E/'plan.json').read_text())
lock = (E/'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (E/'controller_exit.json').exists()

def write(path, value):
    q = path.with_suffix(path.suffix+'.partial')
    q.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    q.replace(path)

def run(split, gpu, shard, mode):
    tag = f'{mode}-base-{split}-shard{shard}'
    cmd = [sys.executable, str(R/'source/base_policy_worker_v2.py'), '--experiment', str(R),
           '--evaluation-root', str(E), '--group', 'base', '--attempt', 'attempt-001',
           '--mode', mode, '--split', split, '--gpu', str(gpu), '--shard', str(shard), '--shards', '5']
    env = os.environ.copy()
    env.update(PAPER_EXPERIMENT_ROOT=str(R), PYTHONPATH=str(R/'source'))
    with (E/'logs'/f'{tag}.log').open('x') as log:
        p = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(E/'live'/f'gpu-{gpu}.json', {'pid': p.pid, 'gpu': gpu, 'split': split, 'mode': mode, 'shard': shard, 'time': time.time()})
        code = p.wait()
    result = {'split': split, 'gpu': gpu, 'shard': shard, 'mode': mode, 'returncode': code, 'time': time.time()}
    write(E/'finished'/f'{tag}.json', result)
    return result

write(E/'controller_process.json', {'pid': os.getpid(), 'started_at': time.time(), 'plan': str(E/'plan.json')})
success = False
try:
    # Wait on the precise already-running diagnostic; do not create a new one.
    learning = R/'eval/eight_action_stable_rotary_001'
    prerequisite = learning/'learning_gate.json'
    write(E/'phase.json', {'phase': 'waiting_for_existing_learning_gate', 'dependency': str(prerequisite), 'time': time.time()})
    start = time.time()
    while not prerequisite.exists():
        terminal = learning/'verification-controller-exit.json'
        if terminal.exists():
            raise RuntimeError('Learning prerequisite ended without a gate: '+terminal.read_text())
        guard = json.loads((learning/'verification-controller-launch.json').read_text())['pid']
        status = (Path('/proc')/str(guard)/'stat').read_text().split(') ',1)[1].split()[0]
        assert status != 'Z', 'Dependency verifier ended; inspect the existing attempt.'
        assert time.time()-start < 3600, 'Dependency observation bound exceeded; no restart.'
        time.sleep(5)
    assert json.loads(prerequisite.read_text())['passed'], 'Basic learning gate failed; do not start full control evaluation.'
    write(E/'phase.json', {'phase': 'single_episode_pilot', 'time': time.time()})
    pilot = run('val_seen', 0, 0, 'pilot')
    write(E/'pilot_results.json', [pilot])
    assert pilot['returncode'] == 0, 'Inspect failed pilot before full dispatch.'
    write(E/'phase.json', {'phase': 'full_initialization_control_evaluation', 'time': time.time()})
    def lane(gpu):
        results = []
        for split in ['val_seen', 'val_unseen']:
            value = run(split, gpu, gpu, 'shard')
            results.append(value)
            if value['returncode'] != 0:
                break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        lanes = list(pool.map(lane, range(5)))
    outcomes = [value for lane in lanes for value in lane]
    write(E/'all_worker_results.json', outcomes)
    assert len(outcomes) == 10 and all(x['returncode'] == 0 for x in outcomes)
    final = {'kind': PLAN['kind'], 'checkpoint': PLAN['checkpoint'], 'temperature': 0.2, 'seed': 10}
    for split, count in [('val_seen',333), ('val_unseen',531)]:
        paths = list((E/'base/attempt-001').glob(f'shard-{split}-gpu*-shard*/episodes/*.json'))
        rows = {p.stem: json.loads(p.read_text()) for p in paths}
        assert len(paths) == len(rows) == count and set(rows) == set(PLAN['episode_ids'][split])
        keys = ['success','oracle_success','distance_to_goal','ndtw','sdtw','path_length','steps_taken']
        final[split] = {'episodes': count, 'exact_official_ID_coverage': True,
                        'success_count': sum(x['success'] for x in rows.values()),
                        'oracle_success_count': sum(x['oracle_success'] for x in rows.values()),
                        'metrics': {k: sum(float(x[k]) for x in rows.values())/count for k in keys}}
    write(E/'final_metrics.json', final)
    write(R/'report/initialization_policy_control_metrics.json', final)
    success = True
except BaseException:
    write(E/'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
    raise
finally:
    write(E/'controller_exit.json', {'success': success, 'time': time.time(), 'new_paid_nodes_submitted': False})
