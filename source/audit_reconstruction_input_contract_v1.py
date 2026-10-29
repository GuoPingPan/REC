"""Check frozen ViPE entry and all selected inference receipts on CPU.

The audit inspects configuration/source/receipt correspondence. It is not an
operating-system trace of historical file accesses and does not rerun ViPE.
"""
from pathlib import Path
import argparse
import ast
import collections
import concurrent.futures
import hashlib
import json
import time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
N = Path('/mnt/data/users/kindel/uav/experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery')
C = R / 'dataset/complete-proxy-pass-metric-v2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

protocol = load(N / 'configs/protocol.json')
entries = [N / 'source' / name / 'infer_v1.py' for name in ['snapshot-c8a377db1f81c484', 'snapshot-68f9d409dba6415a']]
entry_sha = {str(p): sha(p) for p in entries}
assert set(entry_sha.values()) == {protocol['source_sha256']['infer_v1.py']}
text = entries[-1].read_text()
tree = ast.parse(text)
config_keys = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'config':
        assert isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
        config_keys.add(node.slice.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'config' and node.func.attr == 'get':
        assert isinstance(node.args[0], ast.Constant)
        config_keys.add(node.args[0].value)
assert config_keys == {'video', 'video_sha256', 'frame_count', 'frames', 'K'}
frame_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'VideoFrame']
assert len(frame_calls) == 1
frame_fields = {kw.arg for kw in frame_calls[0].keywords}
assert frame_fields == {'raw_frame_idx', 'rgb', 'intrinsics', 'camera_type'}
assignments = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'slam_cfg' and isinstance(node.value, ast.Constant):
                assignments[target.attr] = node.value.value
assert assignments['keyframe_depth'] == 'moge2-l' and assignments['optimize_intrinsics'] is False
expected_K = [[112., 0., 112.], [0., 112., 112.], [0., 0., 1.]]
inventory = C / 'route_inventory.json'
rows = load(inventory)
assert len(rows) == 2899

def audit(row):
    rid = row['route_id']
    input_path = N / 'inputs' / (rid + '.json')
    cfg = load(input_path)
    out = N / 'routes' / rid / 'attempt-001'
    receipt_path = out / 'prediction/inference_receipt.json'
    receipt = load(receipt_path)
    original_result = load(out / 'result.json')
    assert sha(receipt_path) == original_result['inference_receipt_sha256']
    assert set(cfg) == config_keys
    assert all(set(f) == {'sha256_rgb'} for f in cfg['frames'])
    assert cfg['K'] == receipt['K_initial'] == expected_K
    assert cfg['frame_count'] == receipt['frame_count'] == row['frame_count'] == len(cfg['frames'])
    assert receipt['input_config_sha256'] == sha(input_path)
    assert receipt['input_video_sha256'] == cfg['video_sha256']
    assert receipt['decoded_rgb_sha256'] == [f['sha256_rgb'] for f in cfg['frames']]
    assert receipt['poses_or_depth_labels_used_for_inference'] is False
    assert receipt['variant'] == 'fixed nominal K; no dynamic/sky masks; upstream dense visual infill'
    assert receipt['method'] == 'ViPE SLAM core with MoGe2-L keyframe depth prior'
    prediction = out / 'prediction/predicted_camera_c2w.npz'
    prediction_sha = sha(prediction)
    assert prediction_sha == receipt['predicted_poses_sha256'] == original_result['prediction_sha256']
    label = load(C / 'route_labels' / (rid + '.json'))
    assert label['source_prediction_sha256'] == prediction_sha and label['scale'] == 1.0
    assert label['GT_poses_used_for_REC_labels'] is False and label['GT_actions_used_for_REC_labels'] is False
    with np.load(prediction, allow_pickle=False) as data:
        poses = data['data']
        assert poses.shape == (row['frame_count'], 4, 4) and np.isfinite(poses).all()
    return {'route_id': rid, 'scene': row['scene_id'], 'frames': row['frame_count'],
            'input_config_sha256': sha(input_path), 'inference_receipt_sha256': sha(receipt_path),
            'source_prediction_sha256': prediction_sha}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    proofs = list(pool.map(audit, rows))
summary = {'created_at': time.time(), 'routes': len(proofs),
           'instructions': sum(len(x['episodes']) for x in rows),
           'RGB_frames': sum(x['frames'] for x in proofs),
           'by_scene_routes': dict(collections.Counter(x['scene'] for x in proofs)),
           'nominal_intrinsics_K': expected_K, 'optimize_intrinsics': False,
           'source_entry_SHA256': entry_sha, 'source_config_keys_read': sorted(config_keys),
           'VideoFrame_fields': sorted(frame_fields), 'depth_prior': 'MoGe2-L learned pretrained prior',
           'dynamic_or_sky_masks': 'not supplied by the frozen inference entry',
           'seed_in_frozen_entry': 0, 'REC_label_scale': 1.0,
           'selected_input_receipts_and_prediction_hashes_verified': True,
           'historical_OS_file_access_traced': False, 'ViPE_rerun': False,
           'running_jobs_or_training_data_modified': False,
           'scope': 'All selected v2 candidate routes. Any v3 subset inherits these source-input checks by route ID and unchanged label/prediction hashes.',
           'paper_implication': 'State calibrated monocular RGB + fixed nominal K, known camera extrinsic, frame/action-clock and clip endpoint priors. Do not claim camera-agnostic unknown-intrinsics recovery. No poses or depth label arrays are fields of the inspected inference input/config/VideoFrame contract.',
           'route_inventory_sha256': sha(inventory), 'source_sha256': sha(Path(__file__))}
E = R / 'eval/reconstruction_input_contract_001'
E.mkdir(exist_ok=False)
(E / 'route_proofs.json').write_text(json.dumps(proofs, indent=2))
summary['route_proofs_sha256'] = sha(E / 'route_proofs.json')
(E / 'summary.json').write_text(json.dumps(summary, indent=2))
(E / 'exit.json').write_text(json.dumps({'success': True, 'routes': len(proofs),
                                         'summary_sha256': sha(E / 'summary.json'), 'time': time.time()}, indent=2))
print(json.dumps(summary, indent=2))
