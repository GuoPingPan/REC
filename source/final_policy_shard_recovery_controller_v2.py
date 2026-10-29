"""Retain eight original healthy shards; replay only a technically failed lane.

All policy workers, original episode ordering, model, sampling and simulator
code remain frozen. This controller records the real cross-attempt provenance.
It never retries itself, stops healthy tasks, or edits original artifacts.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

GROUP = 'rec-joint-boundary'
HEALTHY = [0, 2, 3, 4]
OLD = 'eval/final_C4_rec_seed12_001'
NEW = 'eval/final_C4_rec_seed12_002'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(pid):
    try: return (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().replace('\0', ' ')
    except OSError: return ''
def save(p, obj):
    t = p.with_suffix(p.suffix + '.partial')
    t.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    t.replace(p)
def link(source, dest):
    assert source.exists(), str(source)
    if dest.is_symlink(): assert dest.resolve() == source.resolve()
    else:
        assert not dest.exists(), str(dest)
        dest.symlink_to(source, target_is_directory=source.is_dir())

def checks(R, plan):
    for n, h in plan['runtime_source_sha256'].items(): assert sha(R / 'source' / n) == h, n
    from final_policy_checkpoint_contract_v2 import verify_final_training
    _, _, qualification = verify_final_training(R, plan, GROUP)
    assert qualification['training_seed'] == 12

def prepare(R):
    O, E = R / OLD, R / NEW
    assert not E.exists()
    Q = R / 'pipeline/final_navigation_sequence_001'
    assert not read(Q / 'exit.json')['success'] and not cmd(read(Q / 'process.json')['pid'])
    p = read(O / 'plan.json')
    assert p['GPU_indices'] == [0, 1, 2, 3, 4]
    assert (p['temperature'], p['sampling_seed'], p['max_actions']) == (.2, 10, 500)
    checks(R, p)
    D = O / GROUP / 'attempt-001/shard-val_seen-gpu1-shard1'
    failed = read(D / 'exit.json')
    assert not failed['success'] and failed['client_returncode'] == -15
    proc = read(D / 'process.json')
    assert not cmd(proc['pid']) and not cmd(proc['client_pid'])
    preserved = D / 'technical_failure_preserved_001/before_stop_manifest.json'
    manifest = read(preserved)
    assert manifest['completed_records_preserved'] == 42
    for rel, h in manifest['files_sha256'].items(): assert sha(preserved.parent / rel) == h
    pilot = O / GROUP / 'attempt-001/pilot-val_seen-gpu0-shard0'
    assert read(pilot / 'exit.json')['success'] and read(pilot / 'validated_metrics.json')['episode_count'] == 1
    selected = []
    for gpu in HEALTHY:
        d = O / GROUP / 'attempt-001' / f'shard-val_seen-gpu{gpu}-shard{gpu}'
        x = read(d / 'process.json')
        assert x['gpu'] == gpu and x['training_qualification']['training_seed'] == 12
        assert x['training_qualification']['training_config_sha256'] == p['groups'][GROUP]['training_config_sha256']
        assert (d / 'exit.json').exists() or ('final_policy_worker_v2.py' in cmd(x['pid']) and str(O) in cmd(x['pid']) and ('--gpu ' + str(gpu)) in cmd(x['pid']))
        if (d / 'exit.json').exists(): assert read(d / 'exit.json')['success']
        selected.append({'gpu': gpu, 'original_Seen_process_sha256': sha(d / 'process.json'), 'original_worker_pid': x['pid']})
    E.mkdir()
    for n in ['logs', 'live', 'finished']: (E / n).mkdir()
    (E / GROUP / 'attempt-001').mkdir(parents=True)
    (E / 'plan.json').write_bytes((O / 'plan.json').read_bytes())
    (E / 'pilot_results.json').write_bytes((O / 'pilot_results.json').read_bytes())
    link(pilot, E / GROUP / 'attempt-001' / pilot.name)
    recovery = {'kind': 'same_protocol_failed_shard_replay_with_healthy_shard_retention', 'created_at': time.time(),
        'original_evaluation_relative': OLD, 'recovered_evaluation_relative': NEW,
        'original_plan_sha256': sha(O / 'plan.json'), 'original_terminal_queue_plan_sha256': sha(Q / 'plan.json'),
        'original_terminal_queue_exit_sha256': sha(Q / 'exit.json'),
        'preserved_failed_shard_manifest': str(preserved.relative_to(R)), 'preserved_failed_shard_manifest_sha256': sha(preserved),
        'excluded_failed_Seen_partial_records': 42, 'excluded_failed_shard': str(D.relative_to(R)),
        'healthy_GPU_lanes_retained': HEALTHY, 'healthy_initial_sources': selected,
        'replay_GPU': 1, 'replay_Seen_original_ID_count': 67, 'new_Unseen_original_ID_count': 106,
        'replay_policy': 'Replay failed Seen shard from original beginning; never splice42-record prefix into reset RNG sequence. Unseen starts as its original separate process. No outcome-based selection.',
        'original_pilot_reused': True, 'original_pilot_results_sha256': sha(O / 'pilot_results.json'),
        'original_pilot_process_sha256': sha(pilot / 'process.json'),
        'orchestration_source': str(Path(__file__).relative_to(R)), 'orchestration_source_sha256': sha(Path(__file__)),
        'frozen_worker_or_policy_sources_modified': False, 'healthy_tasks_restarted': False, 'new_cloud_jobs': 0,
        'scope': 'A recovery orchestration change only. Final model/split/format/outcome audits still required; this preparation is not completion.'}
    save(E / 'recovery_plan.json', recovery)
    print(json.dumps({'prepared': str(E), 'recovery_plan_sha256': sha(E / 'recovery_plan.json'), 'no_workers_launched': True}))

def run(R):
    O, E = R / OLD, R / NEW
    recovery = read(E / 'recovery_plan.json')
    assert recovery['orchestration_source_sha256'] == sha(Path(__file__))
    assert not (E / 'controller_exit.json').exists()
    lock = (E / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(E / 'plan.json')
    assert sha(E / 'plan.json') == sha(O / 'plan.json') == recovery['original_plan_sha256']
    checks(R, plan)
    save(E / 'controller_process.json', {'pid': os.getpid(), 'started_at': time.time(), 'command': sys.argv,
        'orchestration_source_sha256': sha(Path(__file__)), 'recovery_plan_sha256': sha(E / 'recovery_plan.json')})
    success = False

    def retain():
        for split in ['val_seen', 'val_unseen']:
            for gpu in HEALTHY:
                tag = f'shard-{split}-gpu{gpu}-shard{gpu}'
                src = O / GROUP / 'attempt-001' / tag
                if (src / 'process.json').exists():
                    proc = read(src / 'process.json')
                    assert proc['gpu'] == gpu and proc['split'] == split
                    link(src, E / GROUP / 'attempt-001' / tag)
                f = O / 'finished' / f'shard-{GROUP}-{split}-shard{gpu}.json'
                if f.exists():
                    result = read(f)
                    assert result['returncode'] == 0, ('Another original shard failed; preserve and diagnose', str(f))
                    assert read(src / 'exit.json')['success']
                    link(f, E / 'finished' / f.name)

    def wait_GPU1():
        while True:
            retain()
            with open('/tmp/aeroact_official_policy_gpu_1.lock', 'a') as f:
                try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(5)
                    continue
                g = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=index,memory.used,name', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=12, check=True)
                parts = g.stdout.strip().split(',')
                if int(parts[1]) < 128 and '4090' in parts[2]:
                    return {'time': time.time(), 'GPU': 1, 'lock_free': True, 'query': g.stdout}
            time.sleep(5)

    def run_shard(split):
        proof = wait_GPU1()
        save(E / f'GPU1_free_before_{split}.json', proof)
        args = [sys.executable, str(R / 'source/final_policy_worker_v2.py'), '--experiment', str(R),
            '--evaluation-root', str(E), '--group', GROUP, '--attempt', 'attempt-001',
            '--mode', 'shard', '--split', split, '--gpu', '1', '--shard', '1', '--shards', '5']
        env = dict(os.environ, PAPER_EXPERIMENT_ROOT=str(R), PYTHONPATH=str(R / 'source'))
        tag = f'shard-{GROUP}-{split}-shard1'
        save(E / 'phase.json', {'phase': 'full_official_navigation_shard_recovery', 'split': split,
            'replayed_GPU': 1, 'retained_healthy_GPUs': HEALTHY, 'time': time.time()})
        with (E / 'logs' / (tag + '.log')).open('x') as out:
            child = subprocess.Popen(args, cwd='/tmp', env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            save(E / 'live/gpu-1.json', {'pid': child.pid, 'command': args, 'group': GROUP,
                'split': split, 'mode': 'shard', 'shard': 1, 'time': time.time()})
            while child.poll() is None:
                retain()
                D = E / GROUP / 'attempt-001' / f'shard-{split}-gpu1-shard1'
                if (D / 'process.json').exists():
                    p = read(D / 'process.json'); W = Path(p['workspace'])
                    exits = sorted((W / 'native-logs').glob('*.exit.json'), key=lambda f: f.name.rsplit('-', 1)[-1])
                    live_native = any(str(W) in cmd(read(f)['pid']) for f in (W / 'native-logs').glob('*.process.json'))
                    assert not (exits and read(exits[-1])['exit_code'] == -11 and not live_native and cmd(p.get('client_pid'))), 'Recovery shard native crash; preserve without automatic retry'
                time.sleep(10)
            result = {'group': GROUP, 'split': split, 'gpu': 1, 'shard': 1, 'mode': 'shard',
                'returncode': child.returncode, 'time': time.time()}
            save(E / 'finished' / (tag + '.json'), result)
            assert child.returncode == 0, 'Actual recovered worker failed; preserve all sources'

    try:
        retain()
        for split in ['val_seen', 'val_unseen']: run_shard(split)
        while True:
            retain()
            f = list((E / 'finished').glob('shard-*.json'))
            if len(f) == 10: break
            save(E / 'phase.json', {'phase': 'waiting_retained_original_healthy_shards', 'finished': len(f), 'time': time.time()})
            time.sleep(15)
        checks(R, plan)
        outcomes = [read(p) for p in sorted(f)]
        assert len(outcomes) == 10 and all(x['returncode'] == 0 for x in outcomes)
        provenance, final = [], {GROUP: {}}
        keys = ['success', 'oracle_success', 'distance_to_goal', 'ndtw', 'sdtw', 'path_length', 'steps_taken']
        for split, count in [('val_seen', 333), ('val_unseen', 531)]:
            records = {}
            for gpu in range(5):
                D = E / GROUP / 'attempt-001' / f'shard-{split}-gpu{gpu}-shard{gpu}'
                assert read(D / 'exit.json')['success'] and read(D / 'exit.json')['client_returncode'] == 0
                rows = {p.stem: read(p) for p in (D / 'episodes').glob('*.json')}
                assert not set(records).intersection(rows)
                assert set(rows) == set(read(D / 'process.json')['assigned_episode_ids'])
                assert read(D / 'validated_metrics.json')['episode_count'] == len(rows)
                records.update(rows)
                F = E / 'finished' / f'shard-{GROUP}-{split}-shard{gpu}.json'
                provenance.append({'split': split, 'gpu': gpu, 'episodes': len(rows), 'directory': str(D.relative_to(R)),
                    'real_source_directory': str(D.resolve().relative_to(R)), 'retained_original': gpu in HEALTHY,
                    'process_sha256': sha(D / 'process.json'), 'exit_sha256': sha(D / 'exit.json'),
                    'validated_metrics_sha256': sha(D / 'validated_metrics.json'), 'actual_finished_result_sha256': sha(F),
                    'episode_sha256': {p.name: sha(p) for p in (D / 'episodes').glob('*.json')}})
            assert len(records) == count and set(records) == set(plan['episode_ids'][split])
            final[GROUP][split] = {'episodes': count, 'exact_official_ID_coverage': True,
                'success_count': sum(x['success'] for x in records.values()),
                'oracle_success_count': sum(x['oracle_success'] for x in records.values()),
                'metrics': {k: sum(float(x[k]) for x in records.values()) / count for k in keys}}
        assert sum(x['retained_original'] for x in provenance) == 8
        save(E / 'all_worker_results.json', outcomes)
        save(E / 'final_metrics.json', {'models': final, 'temperature': .2, 'sampling_seed': 10,
            'checkpoint_qualification': plan['groups'], 'time': time.time()})
        save(E / 'recovery_composition_verified.json', {'success': True, 'time': time.time(),
            'recovery_plan_sha256': sha(E / 'recovery_plan.json'), 'shards': provenance,
            'retained_original_successful_shards': 8, 'new_successful_shards': 2,
            'excluded_failed_Seen_partial_records': 42, 'original_source_files_modified_by_recovery': False,
            'source_sha256': sha(Path(__file__)), 'scope': 'Actual ten successful worker sources and864 original records. Frozen split/format/whole/outcome audits remain required.',
            'original_controller_status': read(O / 'controller_exit.json') if (O / 'controller_exit.json').exists() else None})
        success = True
    except BaseException:
        save(E / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time(), 'no_automatic_retry': True,
            'healthy_original_workers_stopped': False, 'owned_worker_preserved_for_diagnosis': True})
        raise
    finally:
        save(E / 'controller_exit.json', {'success': success, 'time': time.time(), 'new_paid_jobs': 0,
            'orchestration': 'retained8_original_shards_plus2_real_recovery_workers',
            'recovery_plan_sha256': sha(E / 'recovery_plan.json')})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--evaluation-root', type=Path)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    assert a.prepare != a.run
    if a.evaluation_root is not None: assert a.evaluation_root.resolve() == a.root.resolve() / NEW
    if a.prepare: prepare(a.root.resolve())
    else: run(a.root.resolve())
