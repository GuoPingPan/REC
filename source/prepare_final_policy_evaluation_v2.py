"""Create a final-model evaluation plan only after real training qualification.

Default prepares reviewable files. --launch uses explicitly listed existing DSW
GPUs; no cloud jobs or resource allocation are performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from final_policy_checkpoint_contract_v2 import verify_final_training
from sampler_seed_contract_v2 import personal_path

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--evaluation-root', required=True)
ap.add_argument('--group', action='append', nargs=3, metavar=('GROUP', 'TRAINING_CONFIG', 'COMPLETION_RECEIPT'), required=True)
ap.add_argument('--gpus', nargs='+', type=int, required=True)
ap.add_argument('--launch', action='store_true')
a = ap.parse_args()
R = Path(a.root).resolve()
E = Path(a.evaluation_root).resolve()
assert E.is_relative_to(R / 'eval') and not E.exists()
assert a.gpus and len(set(a.gpus)) == len(a.gpus) and all(0 <= i < 5 for i in a.gpus)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
manifest_path = R / 'inventory/strict_release_collator_source.json'
manifest = read(manifest_path)
S = personal_path(R, manifest['source'])
runtime = ['final_policy_worker_v2.py', 'final_policy_checkpoint_contract_v2.py',
           'final_policy_controller_v2.py', 'prepare_final_policy_evaluation_v2.py',
           'policy_eval_stage_assets_v1.py', 'policy_eval_unprivileged_launcher_v1.py',
           'policy_eval_native_smoke_client_v2.py', 'policy_eval_inference_entry_v1.py',
           'sampler_seed_contract_v2.py', 'target_valid_data_contract_v1.py']
for name in runtime:
    compile((R / 'source' / name).read_bytes(), name, 'exec')
plan = {'kind': 'corrected_final_checkpoint_navigation', 'source_manifest_sha256': sha(manifest_path),
        'runtime_source_sha256': {n: sha(R / 'source' / n) for n in runtime},
        'eval_py_sha256': sha(S / 'eval.py'), 'temperature': .2, 'sampling_seed': 10,
        'max_actions': 500, 'GPU_indices': a.gpus, 'groups': {}, 'splits': {}, 'episode_ids': {},
        'scope': 'Qualified final checkpoints, unchanged official action parser, collision/termination and scoring. One-episode pipeline pilots precede complete official splits. A navigation failure is a valid evaluation result.',
        'new_paid_jobs': 0, 'created_at': time.time()}
allowed = {'gt-joint-boundary', 'rec-joint-boundary', 'gt-native-merge', 'rec-native-merge'}
for group, cfg_arg, completion_arg in a.group:
    assert group in allowed and group not in plan['groups']
    cfg_path, completion_path = Path(cfg_arg).resolve(), Path(completion_arg).resolve()
    assert cfg_path.is_relative_to(R) and completion_path.is_relative_to(R)
    cfg = read(cfg_path)
    done_path = cfg_path.parent / 'debug/training-process-complete.json'
    plan['groups'][group] = {'training_config_relative': str(cfg_path.relative_to(R)),
                             'training_config_sha256': sha(cfg_path),
                             'completion_receipt_relative': str(completion_path.relative_to(R)),
                             'completion_receipt_sha256': sha(completion_path),
                             'training_complete_receipt_sha256': sha(done_path),
                             'training_seed': cfg['seed'], 'optimizer_steps': cfg['optimizer_steps']}
    verify_final_training(R, plan, group)
assert len({v['training_seed'] for v in plan['groups'].values()}) == 1, 'Use paired training seeds in each comparison.'
COMMON = personal_path(R, '/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
for split, expected in [('val_seen', 333), ('val_unseen', 531)]:
    file = COMMON / 'eval/official_annotations' / (split + '.json')
    raw = file.read_bytes()
    episodes = json.loads(raw)['episodes']
    assert episodes == read(COMMON / 'inventory' / ('official_' + split + '.json'))['episodes']
    ids = [str(e['episode_id']) for e in episodes]
    assert len(ids) == len(set(ids)) == expected
    plan['splits'][split] = {'episodes': expected, 'sha256': hashlib.sha256(raw).hexdigest()}
    plan['episode_ids'][split] = ids
E.mkdir()
for folder in ['logs', 'live', 'finished']:
    (E / folder).mkdir()
(E / 'plan.json').write_text(json.dumps(plan, indent=2))
command = [sys.executable, '-u', str(R / 'source/final_policy_controller_v2.py'),
           '--root', str(R), '--evaluation-root', str(E)]
if a.launch:
    (E / 'launch_intent.json').write_text(json.dumps({'command': command, 'state': 'launching', 'time': time.time()}, indent=2))
    with (E / 'logs/controller.log').open('x') as stream:
        child = subprocess.Popen(command, cwd='/tmp', stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (E / 'launch.json').write_text(json.dumps({'pid': child.pid, 'command': command, 'time': time.time(), 'new_paid_jobs': 0}, indent=2))
print(json.dumps({'prepared': str(E), 'launched': a.launch, 'command': command,
                  'models': list(plan['groups']), 'new_paid_jobs': 0}, indent=2))
