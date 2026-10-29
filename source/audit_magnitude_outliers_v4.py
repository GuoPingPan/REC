"""Read-only raw-versus-oracle-scale diagnostic; no data filtering or relabeling."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time
import numpy as np
from final_cohort_geometry_contract_v3 import verify_final_cohort
from primitive_calibration import OPTICAL_TO_FRD, TRANSLATIONS, motion_features, predict

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--dataset', required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
C = R / 'dataset' / a.dataset
final_provenance = verify_final_cohort(R, a.dataset)
U = Path('/mnt/data/users/kindel/uav')
N = U / 'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery'
O = U / 'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official'
E = R / 'eval' / ('magnitude_outliers_' + a.dataset + '_001')
E.mkdir(exist_ok=False)
helper_sha = hashlib.sha256((R / 'source/primitive_calibration.py').read_bytes()).hexdigest()
assert helper_sha == '736fca2a1c76f02701347fbbe0578d9c93c32a39586f2ed950f2ad3988f76fe7'
thresholds = [10., 25., 100., 1000., 10000.]
protocol = {'at': time.time(), 'dataset': a.dataset, 'helper_sha256': helper_sha,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'raw_step_thresholds_m_declared_before_GT_read': thresholds,
            'alignment': 'Existing historical per-route Sim3 fit to full GT camera trajectory; oracle diagnostic only. Never used to create primary recovered labels.',
            'selection': 'All frozen selected routes; top30 events by raw recovered body translation magnitude, deterministic tie break by routeID/frame.',
            'changes_to_data_labels_or_selection': False,
            'parent_training_hold': (C / 'TRAINING_USE_HOLD.json').exists()}
protocol['final_cohort_provenance'] = final_provenance
(E / 'protocol.json').write_text(json.dumps(protocol, indent=2))
rows = json.loads((C / 'route_inventory.json').read_text())
GT = {str(e['episode_id']): e for e in json.loads((O / 'inventory/official_train.json').read_text())['episodes']}

def stat(v):
    return {'n': len(v), 'mean': float(np.mean(v)) if len(v) else None,
            'quantile_levels': [0, .1, .25, .5, .75, .9, .99, 1],
            'quantiles': np.quantile(v, [0, .1, .25, .5, .75, .9, .99, 1]).tolist() if len(v) else None}

def run(row):
    rid = row['route_id']
    A = N / 'routes' / rid / 'attempt-001'
    result = json.loads((A / 'result.json').read_text())
    label = json.loads((C / 'route_labels' / (rid + '.json')).read_text())
    raw_bytes = (A / 'prediction/predicted_camera_c2w.npz').read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == label['source_prediction_sha256'] == result['prediction_sha256']
    assert hashlib.sha256((A / 'aligned_trajectories.npz').read_bytes()).hexdigest() == result['aligned_sha256']
    with np.load(A / 'prediction/predicted_camera_c2w.npz', allow_pickle=False) as data:
        raw = data['data'].astype(float)
    with np.load(A / 'aligned_trajectories.npz', allow_pickle=False) as data:
        aligned = data['sim3_camera'].astype(float)
        assert np.array_equal(data['predicted_camera'], raw)
        stored_aligned_body = data['sim3_body_position'].copy()
    gt = np.asarray(GT[row['episodes'][0]]['actions'][:-1], dtype=int)
    predicted = predict(raw, calibrate=False)
    assert np.array_equal(predicted['actions'], label['GT_free_REC_actions'][:-1])
    assert len(raw) == len(gt) + 1
    raw_body_r = raw[:, :3, :3] @ OPTICAL_TO_FRD.T
    raw_body_p = raw[:, :3, 3] - .5 * raw_body_r[:, :, 0]
    raw_step = np.linalg.norm(np.diff(raw_body_p, axis=0), axis=1)
    camera_step = np.linalg.norm(np.diff(raw[:, :3, 3], axis=0), axis=1)
    # Independent world-coordinate reconstruction must match the earlier local-frame calculation.
    local, lever, _ = motion_features(raw)
    assert np.allclose(raw_step, np.linalg.norm(local - lever, axis=1), rtol=1e-5, atol=1e-4)
    aligned_body = aligned[:, :3, 3] - .5 * (aligned[:, :3, :3] @ OPTICAL_TO_FRD.T)[:, :, 0]
    assert np.allclose(aligned_body, stored_aligned_body, rtol=1e-5, atol=1e-4)
    aligned_step = np.linalg.norm(np.diff(aligned_body, axis=0), axis=1)
    mask = np.isin(gt, [1, 4, 5, 6, 7])
    expected = np.linalg.norm(TRANSLATIONS[gt[mask] - 1], axis=1)
    raw_err = np.abs(raw_step[mask] - expected) / expected
    aligned_err = np.abs(aligned_step[mask] - expected) / expected
    good = predicted['actions'][mask] == gt[mask]
    i = int(np.argmax(raw_step))
    rotation_orthogonality_error = np.max(np.abs(np.swapaxes(raw_body_r, 1, 2) @ raw_body_r - np.eye(3)))
    events = []
    for j in np.argsort(-raw_step, kind='stable')[:min(5, len(gt))]:
        events.append({'route_id': rid, 'scene_id': row['scene_id'], 'source_frame': int(j), 'target_frame': int(j + 1),
                       'raw_body_step_m': float(raw_step[j]), 'raw_camera_step_m': float(camera_step[j]),
                       'oracle_Sim3_body_step_m': float(aligned_step[j]), 'GT_action': int(gt[j]),
                       'REC_action': int(predicted['actions'][j]), 'REC_confidence': float(predicted['confidence'][j]),
                       'raw_camera_start_xyz': raw[j, :3, 3].tolist(), 'raw_camera_end_xyz': raw[j + 1, :3, 3].tolist(),
                       'prediction_sha256': result['prediction_sha256'], 'aligned_sha256': result['aligned_sha256'],
                       'source_RGB': str(Path(row['rgb']) / f'frame_{j:03d}.jpg'),
                       'target_RGB': str(Path(row['rgb']) / f'frame_{j+1:03d}.jpg')})
        assert Path(events[-1]['source_RGB']).is_file() and Path(events[-1]['target_RGB']).is_file()
    record = {'route_id': rid, 'scene_id': row['scene_id'], 'instructions': len(row['episodes']),
              'transitions': len(gt), 'translation_transitions': int(mask.sum()),
              'oracle_Sim3_scale': result['sim3_scale'], 'max_raw_body_step_m': float(raw_step[i]),
              'max_raw_step_source_frame': i, 'rotation_orthogonality_max_error': float(rotation_orthogonality_error),
              'raw_counts_over_thresholds': [int((raw_step > t).sum()) for t in thresholds],
              'oracle_aligned_counts_over_thresholds': [int((aligned_step > t).sum()) for t in thresholds]}
    return record, events, raw_err, aligned_err, raw_err[good], aligned_err[good], raw_step[mask] / expected

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run, rows):
        results.append(result)
        if len(results) % 100 == 0:
            print(json.dumps({'completed_routes': len(results), 'at': time.time()}), flush=True)
records = [x[0] for x in results]
events = sorted([e for x in results for e in x[1]], key=lambda e: (-e['raw_body_step_m'], e['route_id'], e['source_frame']))[:30]
summary = {'at': time.time(), 'dataset': a.dataset, 'routes': len(records),
           'instructions': sum(x['instructions'] for x in records),
           'transitions': sum(x['transitions'] for x in records),
           'translation_transitions': sum(x['translation_transitions'] for x in records),
           'raw_relative_translation_magnitude_error': stat(np.concatenate([x[2] for x in results])),
           'oracle_Sim3_relative_translation_magnitude_error': stat(np.concatenate([x[3] for x in results])),
           'type_correct_raw_relative_translation_magnitude_error': stat(np.concatenate([x[4] for x in results])),
           'type_correct_oracle_Sim3_relative_translation_magnitude_error': stat(np.concatenate([x[5] for x in results])),
           'raw_translation_magnitude_divided_by_GT_primitive_magnitude': stat(np.concatenate([x[6] for x in results])),
           'oracle_scale_route_statistics': stat(np.array([x['oracle_Sim3_scale'] for x in records])),
           'raw_step_thresholds_m': thresholds,
           'raw_transitions_over_thresholds': [sum(x['raw_counts_over_thresholds'][i] for x in records) for i in range(len(thresholds))],
           'raw_routes_over_thresholds': [sum(x['raw_counts_over_thresholds'][i] > 0 for x in records) for i in range(len(thresholds))],
           'oracle_aligned_transitions_over_thresholds': [sum(x['oracle_aligned_counts_over_thresholds'][i] for x in records) for i in range(len(thresholds))],
           'maximum_rotation_orthogonality_error': max(x['rotation_orthogonality_max_error'] for x in records),
           'largest_step_events': events,
           'oracle_alignment_uses_GT': True, 'primary_label_estimator_unchanged': True,
           'physical_collision_certified': False,
           'scope': final_provenance['scope']}
(E / 'route_records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
assert summary['routes'] == final_provenance['routes']
assert summary['instructions'] == final_provenance['instructions']
assert summary['transitions'] == final_provenance['nonstop_actions']
summary['final_cohort_provenance'] = final_provenance
summary['scope'] = final_provenance['scope']
(E / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(E / 'exit.json').write_text(json.dumps({'success': True, 'at': time.time(),
    'summary_sha256': hashlib.sha256((E / 'summary.json').read_bytes()).hexdigest(),
    'route_records_sha256': hashlib.sha256((E / 'route_records.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256((E / 'protocol.json').read_bytes()).hexdigest()}, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'largest_step_events'}, indent=2), flush=True)
