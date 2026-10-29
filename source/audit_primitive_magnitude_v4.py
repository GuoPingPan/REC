"""Metric-sensitivity diagnostic; never changes poses, labels or training data.

Distinguish correct primitive type from correct recovered metric displacement.
Synthetic examples characterize the existing quantizer, not ViPE performance.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time
import numpy as np
from final_cohort_geometry_contract_v3 import verify_final_cohort
from scipy.spatial.transform import Rotation
from primitive_calibration import OPTICAL_TO_FRD, TRANSLATIONS, UNITS, motion_features, predict

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
E = R / 'eval' / ('primitive_magnitude_' + a.dataset + '_001')
E.mkdir(exist_ok=False)
helper = R / 'source/primitive_calibration.py'
helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
assert helper_sha == '736fca2a1c76f02701347fbbe0578d9c93c32a39586f2ed950f2ad3988f76fe7'
thresholds = [1e-6, .001, .01, .05, .10, .25, .50]
protocol = {'started_at': time.time(), 'dataset': a.dataset,
            'helper_sha256': helper_sha, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'low_motion_thresholds_declared_before_GT_read': thresholds,
            'motion_norm': 'sqrt(sum((body_translation/[5,5,2])**2)+(relative_rotation_angle_degrees/15)**2)',
            'motion_scale': 'Original raw ViPE scale1 with unchanged known0.5mlever arm.',
            'scope': 'Diagnostic on selected frozen routes. Low magnitude is not by itself a certified algorithm failure or physical collision.',
            'training_data_labels_or_pose_predictions_modified': False,
            'parent_dataset_on_training_hold': (C / 'TRAINING_USE_HOLD.json').exists()}
protocol['final_cohort_provenance'] = final_provenance
(E / 'protocol.json').write_text(json.dumps(protocol, indent=2))

def camera_pair(distance=0., yaw_degrees=0.):
    body_r = np.stack([np.eye(3), Rotation.from_euler('z', yaw_degrees, degrees=True).as_matrix()])
    body_p = np.array([[0., 0., 0.], [distance, 0., 0.]])
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, :3, :3] = body_r @ OPTICAL_TO_FRD
    poses[:, :3, 3] = body_p + .5 * body_r[:, :, 0]
    return poses

synthetic = []
for distance in [.005, .05, .5, 1., 2.5, 5., 10., 25., 50., 100.]:
    p = predict(camera_pair(distance=distance), calibrate=False)
    assert int(p['actions'][0]) == 1
    synthetic.append({'case': 'pure_forward', 'recovered_magnitude_m': distance,
                      'declared_primitive_magnitude_m': 5., 'predicted_action': 1,
                      'relative_magnitude_error_vs_5m': abs(distance - 5) / 5,
                      'confidence': float(p['confidence'][0]), 'minimum_cost': float(p['minimum_cost'][0])})
for angle in [.015, .15, 1.5, 3., 7.5, 15., 30., 60., 90.]:
    p = predict(camera_pair(yaw_degrees=angle), calibrate=False)
    assert int(p['actions'][0]) == 3
    synthetic.append({'case': 'pure_right_turn', 'recovered_angle_degrees': angle,
                      'declared_primitive_angle_degrees': 15., 'predicted_action': 3,
                      'relative_angle_error_vs_15deg': abs(angle - 15) / 15,
                      'confidence': float(p['confidence'][0]), 'minimum_cost': float(p['minimum_cost'][0])})
stationary = predict(camera_pair(), calibrate=False)
assert int(stationary['actions'][0]) == 1 and float(stationary['confidence'][0]) == 0.

rows = json.loads((C / 'route_inventory.json').read_text())
validation = json.loads((C / 'validation_report.json').read_text())
GT = {str(e['episode_id']): e for e in json.loads((O / 'inventory/official_train.json').read_text())['episodes']}
def work(row):
    path = N / 'routes' / row['route_id'] / 'attempt-001/prediction/predicted_camera_c2w.npz'
    frozen = json.loads((C / 'route_labels' / (row['route_id'] + '.json')).read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen['source_prediction_sha256']
    with np.load(path, allow_pickle=False) as data:
        poses = data['data']
    cam, lever, angular = motion_features(poses)
    delta = cam - lever
    translation = np.linalg.norm(delta, axis=1)
    rotation_degrees = np.sqrt(angular[:, 0]) * 15
    norm = np.sqrt(np.sum((delta / UNITS)**2, axis=1) + angular[:, 0])
    recovered = predict(poses, calibrate=False)['actions']
    assert np.array_equal(recovered, frozen['GT_free_REC_actions'][:-1])
    gold = np.asarray(GT[row['episodes'][0]]['actions'][:-1])
    assert len(gold) == len(norm) == len(recovered)
    return {'route_id': row['route_id'], 'scene': int(row['scene_id']),
            'norm': norm, 'translation_m': translation, 'rotation_degrees': rotation_degrees,
            'pred': recovered, 'gt': gold, 'correct': recovered == gold}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(work, rows))
all_data = {key: np.concatenate([x[key] for x in results]) for key in ['norm', 'translation_m', 'rotation_degrees', 'pred', 'gt', 'correct']}
assert len(all_data['gt']) == validation['nonstop_actions']
assert int(all_data['correct'].sum()) == validation['correct_nonstop_actions']
def stats(values):
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'quantiles_at_0_10_25_50_75_90_99_100_percent': np.quantile(values, [0, .1, .25, .5, .75, .9, .99, 1]).tolist() if len(values) else None}
low = []
for threshold in thresholds:
    mask = all_data['norm'] <= threshold
    low.append({'threshold': threshold, 'transitions': int(mask.sum()),
                'fraction': float(mask.mean()),
                'action_agreement': float(all_data['correct'][mask].mean()) if mask.any() else None,
                'predicted_forward': int((all_data['pred'][mask] == 1).sum()),
                'GT_forward': int((all_data['gt'][mask] == 1).sum())})
gt_t = np.isin(all_data['gt'], [1, 4, 5, 6, 7])
gt_r = np.isin(all_data['gt'], [2, 3])
expected_t = np.linalg.norm(TRANSLATIONS[all_data['gt'][gt_t] - 1], axis=1)
t_error = np.abs(all_data['translation_m'][gt_t] - expected_t) / expected_t
r_error = np.abs(all_data['rotation_degrees'][gt_r] - 15) / 15
summary = {'finished_at': time.time(), 'dataset': a.dataset, 'routes': len(rows),
           'instructions': sum(len(x['episodes']) for x in rows), 'transitions': len(all_data['gt']),
           'micro_action_agreement': float(all_data['correct'].mean()),
           'synthetic_quantizer_examples': synthetic,
           'exact_stationary_example': {'predicted_action': int(stationary['actions'][0]), 'confidence': float(stationary['confidence'][0]),
                                        'scope': 'Outside known-one-active-primitive assumption; illustrates forced-choice/tie behavior, not a measured dataset failure.'},
           'low_motion': low, 'motion_norm': stats(all_data['norm']),
           'GT_translation_relative_magnitude_error': stats(t_error),
           'GT_translation_relative_magnitude_error_when_action_type_correct': stats(t_error[all_data['correct'][gt_t]]),
           'GT_rotation_relative_angle_magnitude_error': stats(r_error),
           'GT_rotation_relative_angle_magnitude_error_when_action_type_correct': stats(r_error[all_data['correct'][gt_r]]),
           'mathematical_counterexample': 'For ideal pure forward d>0 with zero rotation, D_forward=(d/5-1)^2, all competing costs=(d/5)^2+1. Competitor minus forward cost=2d/5>0. Thus correct primitive type does not establish correct metric magnitude. This is a limiting-case statement, not full invariance under arbitrary mixed pose noise/scale.',
           'interpretation': 'Type accuracy evaluates discrete labels under known unit/clock priors. Independently report continuous SE3/Sim3 ATE/RPE and magnitude errors. Do not infer metric trajectory accuracy, collision freedom or navigation SR from type accuracy alone.',
           'data_or_labels_modified': False, 'physical_legality_certified': False,
           'route_inventory_sha256': hashlib.sha256((C / 'route_inventory.json').read_bytes()).hexdigest(),
           'protocol_sha256': hashlib.sha256((E / 'protocol.json').read_bytes()).hexdigest()}
assert summary['routes'] == final_provenance['routes']
assert summary['instructions'] == final_provenance['instructions']
assert summary['transitions'] == final_provenance['nonstop_actions']
summary['final_cohort_provenance'] = final_provenance
summary['scope'] = final_provenance['scope']
(E / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(E / 'exit.json').write_text(json.dumps({'success': True, 'summary_sha256': hashlib.sha256((E / 'summary.json').read_bytes()).hexdigest(), 'time': time.time()}, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'synthetic_quantizer_examples'}, indent=2), flush=True)
