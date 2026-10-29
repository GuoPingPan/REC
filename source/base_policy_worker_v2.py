"""Isolated official eval.py worker, with exact episode coverage validation."""
import argparse
import fcntl
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

COMMON = Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
PAPER = Path(os.environ['PAPER_EXPERIMENT_ROOT']).resolve()
S = Path(json.loads((PAPER/'inventory/strict_release_collator_source.json').read_text())['source']).resolve()
P = Path('/tmp/aeroact_official_runtime_0417_py310_v1/bin/python')
p = argparse.ArgumentParser()
p.add_argument('--experiment', type=Path, required=True)
p.add_argument('--evaluation-root', type=Path, required=True)
p.add_argument('--group', choices=['base'], required=True)
p.add_argument('--mode', choices=['native-smoke', 'native-transition', 'pilot', 'shard'], required=True)
p.add_argument('--split', choices=['val_seen', 'val_unseen'], default='val_seen')
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--shard', type=int, default=0)
p.add_argument('--shards', type=int, default=1)
p.add_argument('--attempt', required=True)
a = p.parse_args()
R = a.experiment
assert R.resolve() == PAPER
assert a.evaluation_root.resolve().is_relative_to(PAPER)

is_native = a.mode.startswith('native-')
assert 0 <= a.gpu < 5 and 0 <= a.shard < a.shards
gpu_lock = Path(f'/tmp/aeroact_official_policy_gpu_{a.gpu}.lock').open('a')
fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert '/' not in a.attempt and '..' not in a.attempt
tag = f'{a.mode}-{a.split}-gpu{a.gpu}-shard{a.shard}'
E = a.evaluation_root / a.group / a.attempt / tag
W = Path('/dev/shm') / f'aeroact_policy_{hashlib.sha256(str(a.evaluation_root.resolve()).encode()).hexdigest()[:8]}_{a.attempt}_{tag}'
PORT = 53000 + a.gpu * 100


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temp.replace(path)


def alive_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


PLAN = json.loads((a.evaluation_root/'plan.json').read_text())
assert PLAN['kind'] == 'unfinetuned_initialization_control'
assert PLAN['source'] == str(S)
checkpoint = Path(PLAN['checkpoint'])
base = Path(PLAN['initial_weights']).resolve()
for component in ['llm','vision_tower','mm_projector']:
    assert (checkpoint/component).is_symlink()
    assert (checkpoint/component).resolve() == (base/component).resolve()
assert (checkpoint/'data_args.json').is_file()
assert json.loads((checkpoint/'config.json').read_text())['num_video_frames'] == 8
cfg = {'evaluation': {'splits': PLAN['splits']}}
cards = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,name', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
gpu_row = next(line.split(',') for line in cards.splitlines() if int(line.split(',')[0]) == a.gpu)
assert int(gpu_row[1].strip()) < 128, ('GPU in use', gpu_row)
assert '4090' in gpu_row[2]
assert not E.exists() and not W.exists(), 'Inspect existing work; never launch a duplicate worker.'
alive_port(PORT)
alive_port(PORT + 1)
E.mkdir(parents=True)
W.mkdir()
write(W / 'owner.json', {'kind': 'official_aeroact_policy_evaluation', 'experiment': str(R), 'attempt': a.attempt, 'tag': tag})
source_manifest = json.loads((PAPER/'inventory/strict_release_collator_source.json').read_text())
code = W/'code'
shutil.copytree(S, code, symlinks=True)
for rel, expected_sha in source_manifest['files'].items():
    assert sha(code/rel) == expected_sha, ('Frozen source differs', rel)
for rel, target in source_manifest['symlinks'].items():
    assert (code/rel).is_symlink() and os.readlink(code/rel) == target
assert sha(code / 'eval.py') == sha(S / 'eval.py')
raw = (COMMON / f'eval/official_annotations/{a.split}.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == cfg['evaluation']['splits'][a.split]['sha256']
episodes = json.loads(raw)['episodes']
assert len(episodes) == cfg['evaluation']['splits'][a.split]['episodes']
assigned = episodes[a.shard::a.shards]
if a.mode in ['native-smoke', 'pilot']:
    assigned = assigned[:1]
elif a.mode == 'native-transition':
    scenes = sorted({int(x['scene_id']) for x in episodes})[:2]
    assigned = [next(x for x in episodes if int(x['scene_id']) == scene) for scene in scenes]
expected_ids = {str(x['episode_id']) for x in assigned}
annotations = code / 'AirVLN-Dataset/data/aerialvln-s'
annotations.mkdir(parents=True, exist_ok=True)
write(annotations / f'{a.split}.json', {'episodes': assigned})
helper = Path(__file__).with_name('policy_eval_unprivileged_launcher_v1.py')
smoke = Path(__file__).with_name('policy_eval_native_smoke_client_v2.py')
shutil.copy2(helper, W / helper.name)
shutil.copy2(smoke, code / smoke.name)
runtime_entry = Path(__file__).with_name('policy_eval_inference_entry_v1.py')
shutil.copy2(runtime_entry, code / runtime_entry.name)
from policy_eval_stage_assets_v1 import stage_assets
server_home, tools = stage_assets(W, sorted({int(x['scene_id']) for x in assigned}))
(code / 'ENVs').symlink_to(W / 'ENVs', target_is_directory=True)
if not is_native:
    (code / 'checkpoints').mkdir(exist_ok=True)
    (code / 'checkpoints/final_model').symlink_to(checkpoint, target_is_directory=True)
env = dict(os.environ, PYTHONPATH=str(code), PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES=str(a.gpu))
env['PATH'] = str(P.parent) + ':' + str(tools) + ':' + env.get('PATH', '')
assert (tools / 'netstat').exists(), 'Use the previously verified private native tools'
common = ['--split', a.split, '--project_prefix', str(code), '--simulator_tool_port', str(PORT)]
client_cmd = [str(P), smoke.name, *common] if is_native else [str(P), runtime_entry.name, *common,
    '--model_path', 'checkpoints/final_model', '--inference_sample_temperature', '0.2',
    '--dataset', 'aerialvln-s', '--name', a.group, '--EVAL_NUM', '-1', '--maxAction', '500']
processes = {'pid': os.getpid(), 'server_pid': None, 'client_pid': None, 'workspace': str(W), 'output': str(E),
             'port': PORT, 'gpu': a.gpu, 'hardware': gpu_row[2].strip(), 'mode': a.mode, 'split': a.split,
             'source': str(S), 'source_content_sha256': sha(PAPER/'inventory/strict_release_collator_source.json'),
             'eval_py_sha256': sha(S / 'eval.py'), 'inference_runtime_entry_sha256': sha(runtime_entry), 'native_launcher_sha256': sha(helper), 'native_stage_sha256': sha(Path(__file__).with_name('policy_eval_stage_assets_v1.py')), 'simulator_server_uid': 999,
             'checkpoint': str(checkpoint) if not is_native else None, 'group': a.group, 'sampling_seed': 10, 'control_kind': PLAN['kind'], 'initial_weight_root': str(base), 'no_navigation_finetuning': True,
             'assigned_episode_ids': sorted(expected_ids), 'original_split_sha256': hashlib.sha256(raw).hexdigest(),
             'command': client_cmd, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
write(E / 'process.json', processes)
server = client = None
success = False
try:
    with (E / 'server.log').open('w') as server_log, (E / 'client.log').open('w') as client_log:
        server = subprocess.Popen(['runuser', '-u', 'uavrun', '--', 'env', f'HOME={server_home}',
            f'XDG_RUNTIME_DIR={server_home}/xdg', f'VK_ICD_FILENAMES={server_home}/nvidia_icd.json',
            f'__EGL_VENDOR_LIBRARY_FILENAMES={server_home}/nvidia_egl.json', str(P),
            'airsim_plugin/AirVLNSimulatorServerTool.py', '--gpus', str(a.gpu), '--port', str(PORT)],
            cwd=code / 'AirVLN', env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        processes['server_pid'] = server.pid
        write(E / 'process.json', processes)
        for _ in range(60):
            assert server.poll() is None, 'Official simulator server exited; inspect server.log'
            try:
                with socket.create_connection(('127.0.0.1', PORT), timeout=.5):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Simulator server listener timeout')
        client = subprocess.Popen(client_cmd, cwd=code, env=env, stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
        processes['client_pid'] = client.pid
        write(E / 'process.json', processes)
        while client.poll() is None:
            write(E / 'worker_status.json', {'phase': 'running', 'client_pid': client.pid, 'server_pid': server.pid,
                'time': time.time(), 'metric_files': len(list((code / 'AirVLN_eval').glob('**/intermediate_results_every/*.json')))})
            time.sleep(10)
        assert client.returncode == 0, ('Official client exited', client.returncode)
    if is_native:
        result = json.loads((code / 'native_smoke_result.json').read_text())
        assert result['passed'] and len(result['checks']) == 8 * len(assigned) and not result['policy_inference_run']
        shutil.copy2(code / 'native_smoke_rgb.png', E / 'rgb.png')
        write(E / 'native_smoke_result.json', result)
    else:
        files = list((code / 'AirVLN_eval').glob('**/intermediate_results_every/*.json'))
        assert len(files) == len(expected_ids), ('Incomplete evaluation', len(files), len(expected_ids))
        records = {p.stem: json.loads(p.read_text()) for p in files}
        assert set(records) == expected_ids, 'Exact assigned episode coverage failed'
        for eid, row in records.items():
            for key in ['distance_to_goal', 'success', 'oracle_success', 'ndtw', 'sdtw', 'steps_taken']:
                assert key in row and math.isfinite(float(row[key])), (eid, key)
            assert row['success'] in [0, 1] and row['oracle_success'] in [0, 1]
        metrics = {key: sum(float(x[key]) for x in records.values()) / len(records)
                   for key in ['distance_to_goal', 'success', 'oracle_success', 'ndtw', 'sdtw', 'path_length', 'steps_taken']}
        destination = E / 'episodes'
        destination.mkdir()
        for p in files:
            shutil.copy2(p, destination / p.name)
        write(E / 'validated_metrics.json', {'passed': True, 'episode_count': len(records), 'episode_ids': sorted(records),
              'metrics': metrics, 'official_eval_source_unchanged': True, 'checkpoint': str(checkpoint),
              'scope': 'Pilot pipeline check only' if a.mode == 'pilot' else 'Official protocol evaluation on one exact split shard'})
    if not is_native:
        shutil.copytree(code / 'AirVLN_eval', E / 'official_outputs')
    success = True
finally:
    for proc in [client, server]:
        if proc is None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
    # PAKs belong solely to this private worker workspace; no shared asset cleanup.
    for path in (W / 'scene-cache').glob('**/*.pak'):
        assert not path.is_symlink() and path.is_relative_to(W)
        path.unlink()
    write(E / 'exit.json', {'success': success, 'client_returncode': client.returncode if client else None,
                           'time': time.time(), 'policy_inference_run': not is_native})
print('POLICY_WORKER_COMPLETE' if not is_native else 'NATIVE_DEPLOYMENT_COMPLETE', E, flush=True)
