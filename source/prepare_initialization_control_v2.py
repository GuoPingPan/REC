"""Stage one initialization-only policy control; dependency is an existing run."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
COMMON = Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official').resolve()
M = COMMON/'runtime/models/NVILA-Lite-8B-ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d'
S = Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source']).resolve()
E = R/'eval/initialization_policy_control_002'
assert not E.exists(), 'Inspect current control before creating another.'
old=R/'eval/initialization_policy_control_001/controller_exit.json'
assert old.exists() and json.loads(old.read_text())['success'] is False
assert not list((R/'eval/initialization_policy_control_001').glob('base/attempt-001/*/process.json'))
learning = R/'eval/eight_action_stable_rotary_001'
assert (learning/'verification-controller-launch.json').exists()
for name in ['base_policy_worker_v2.py','initialization_control_controller_v2.py', 'policy_eval_stage_assets_v1.py',
             'policy_eval_unprivileged_launcher_v1.py','policy_eval_native_smoke_client_v2.py','policy_eval_inference_entry_v1.py']:
    compile((R/'source'/name).read_bytes(), name, 'exec')
E.mkdir()
for name in ['logs','live','finished','initialization-model-view']:
    (E/name).mkdir()
V = E/'initialization-model-view'
original = json.loads((M/'config.json').read_text())
assert original['num_video_frames'] == 8
cfg = dict(original)
cfg['_name_or_path'] = str(V)
cfg['resume_path'] = str(V)
for component in ['llm','vision_tower','mm_projector']:
    (V/component).symlink_to(M/component, target_is_directory=True)
(V/'config.json').write_text(json.dumps(cfg, indent=2))
(V/'data_args.json').write_bytes((R/'eval/learning_smoke_002/checkpoints/final/data_args.json').read_bytes())
plan = {'kind': 'unfinetuned_initialization_control', 'scope': 'Same downloaded initialization weights before this AerialVLN experiment, official navigation protocol; no claim that all pretraining data are known.',
        'source': str(S), 'checkpoint': str(V), 'initial_weights': str(M),
        'model_config_changes': {key: {'original':original.get(key),'control':cfg.get(key)} for key in cfg if cfg.get(key)!=original.get(key)},
        'weights_modified': False, 'temperature': 0.2, 'seed': 10, 'split_protocol_unchanged': True,
        'depends_on': str(learning/'learning_gate.json'), 'splits': {}, 'episode_ids': {},
        'resource_policy': 'Existing five DSW GPUs, only after the eight-action learning check and its generation probe finish; no DLC node allocation.'}
assert set(plan['model_config_changes']).issubset({'_name_or_path','resume_path'})
for split, expected in [('val_seen',333), ('val_unseen',531)]:
    file = COMMON/f'eval/official_annotations/{split}.json'
    raw = file.read_bytes()
    episodes = json.loads(raw)['episodes']
    inventory = COMMON/f'inventory/official_{split}.json'
    assert episodes == json.loads(inventory.read_bytes())['episodes']
    assert len(episodes) == expected and len({str(x['episode_id']) for x in episodes}) == expected
    plan['splits'][split] = {'episodes':expected, 'sha256':hashlib.sha256(raw).hexdigest(),
                             'source':str(file), 'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest()}
    plan['episode_ids'][split] = [str(x['episode_id']) for x in episodes]
(E/'plan.json').write_text(json.dumps(plan, indent=2))
cmd = ['/tmp/aeroact_official_runtime_0417_py310_v1/bin/python', '-u', str(R/'source/initialization_control_controller_v2.py'), '--root', str(R)]
with (E/'logs/controller.log').open('w') as log:
    p = subprocess.Popen(cmd, cwd=R, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'pid':p.pid, 'command':cmd, 'started_at':time.time(), 'dependency':plan['depends_on'], 'new_paid_nodes_submitted':False}
(E/'launch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
