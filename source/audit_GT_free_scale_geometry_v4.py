"""Score the frozen RGB-pose-only scale estimator; never relabel or select data."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time
import numpy as np
from final_cohort_geometry_contract_v3 import verify_final_cohort
from primitive_calibration import OPTICAL_TO_FRD, TRANSLATIONS, costs_at_scale, motion_features

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--dataset', required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
C = R / 'dataset' / a.dataset
final_provenance = verify_final_cohort(R, a.dataset)
N = Path('/mnt/data/users/kindel/uav/experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery')
K = R / 'eval/primitive_calibration_v1'
E = R / 'eval' / ('GT_free_scale_geometry_' + a.dataset + '_001')
E.mkdir(exist_ok=False)
cache_summary = json.loads((K / 'summary.json').read_text())
helper_sha = hashlib.sha256((R / 'source/primitive_calibration.py').read_bytes()).hexdigest()
assert helper_sha == cache_summary['method_script_sha256'] == '736fca2a1c76f02701347fbbe0578d9c93c32a39586f2ed950f2ad3988f76fe7'
assert hashlib.sha256((R / 'source/benchmark_calibration.py').read_bytes()).hexdigest() == cache_summary['source_script_sha256']
protocol = {'at': time.time(), 'dataset': a.dataset,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'estimator_sha256': helper_sha,
            'estimator_cache_summary_sha256': hashlib.sha256((K / 'summary.json').read_bytes()).hexdigest(),
            'scale_estimation': 'Reuse previously frozen primitive-cost scalar fitted from raw RGB-derived camera poses and known camera extrinsic/primitive units/clock; no GT input to the estimator.',
            'evaluation_alignment': 'Raw and GT-free-scaled poses each receive a proper-rotation rigid position alignment for scoring, with fixed scale1. Existing full-GT Sim3 is a separate oracle comparator.',
            'label_policy': 'Recompute action types from cached scalar to verify cache; primary training label files are not modified.',
            'hyperparameters_reselected_from_GT': False,
            'parent_training_hold': (C / 'TRAINING_USE_HOLD.json').exists()}
protocol['final_cohort_provenance'] = final_provenance
(E / 'protocol.json').write_text(json.dumps(protocol, indent=2))
rows = json.loads((C / 'route_inventory.json').read_text())

def position_rigid_align(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(yc.T @ xc / len(x))
    rot = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return xc @ rot.T + y.mean(0)

def body_steps(camera):
    body = camera[:, :3, 3] - .5 * (camera[:, :3, :3] @ OPTICAL_TO_FRD.T)[:, :, 0]
    return np.linalg.norm(np.diff(body, axis=0), axis=1)

def run(row):
    rid = row['route_id']
    D = N / 'routes' / rid / 'attempt-001'
    source = D / 'prediction/predicted_camera_c2w.npz'
    cached_path = K / 'routes' / (rid + '.json')
    cache = json.loads(cached_path.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == cache['predicted_poses_sha256']
    with np.load(source, allow_pickle=False) as data:
        raw = data['data'].astype(float)
    scale = float(cache['calibrated_scale'])
    assert np.isfinite(scale) and scale > 0
    features = motion_features(raw)
    raw_actions = np.argmin(costs_at_scale(features, 1.), axis=1) + 1
    calibrated_actions = np.argmin(costs_at_scale(features, scale), axis=1) + 1
    assert calibrated_actions.tolist() == cache['GT_free_actions']
    calibrated = raw.copy()
    calibrated[:, :3, 3] *= scale
    # Only the evaluation below this line reads reference poses or action labels.
    frozen = json.loads((C / 'route_labels' / (rid + '.json')).read_text())
    assert digest == frozen['source_prediction_sha256']
    assert np.array_equal(raw_actions, frozen['GT_free_REC_actions'][:-1])
    result = json.loads((D / 'result.json').read_text())
    aligned_path = D / 'aligned_trajectories.npz'
    assert hashlib.sha256(aligned_path.read_bytes()).hexdigest() == result['aligned_sha256']
    with np.load(aligned_path, allow_pickle=False) as data:
        gt_camera = data['gt_camera'].copy()
        oracle = data['sim3_camera'].copy()
        assert np.array_equal(raw, data['predicted_camera'])
    observed = json.loads((D / 'actions.json').read_text())
    gold = np.asarray(observed['GT_observed_transition_actions'])
    assert len(gold) == len(raw_actions) == len(raw) - 1
    assert int((calibrated_actions == gold).sum()) == cache['GT_free_calibration_correct']
    assert int((raw_actions == gold).sum()) == cache['raw_metric_correct']
    mask = np.isin(gold, [1, 4, 5, 6, 7])
    expected = np.linalg.norm(TRANSLATIONS[gold[mask] - 1], axis=1)
    errors = [np.abs(body_steps(p)[mask] - expected) / expected for p in [raw, calibrated, oracle]]
    y = gt_camera[:, :3, 3]
    rmses = [float(np.sqrt(np.mean(np.sum((position_rigid_align(p[:, :3, 3], y) - y) ** 2, axis=1)))) for p in [raw, calibrated]]
    oracle_ate = float(np.sqrt(np.mean(np.sum((oracle[:, :3, 3] - y) ** 2, axis=1))))
    record = {'route_id': rid, 'scene_id': row['scene_id'], 'instructions': len(row['episodes']),
              'transitions': len(gold), 'raw_correct': int((raw_actions == gold).sum()),
              'calibrated_correct': int((calibrated_actions == gold).sum()),
              'changed_action_types': int((raw_actions != calibrated_actions).sum()),
              'scale': scale, 'scale_at_search_boundary': cache['scale_at_boundary'],
              'raw_rigid_aligned_camera_ATE_RMSE_m': rmses[0],
              'GT_free_scale_rigid_aligned_camera_ATE_RMSE_m': rmses[1],
              'oracle_Sim3_camera_ATE_RMSE_m': oracle_ate,
              'cached_scale_record_sha256': hashlib.sha256(cached_path.read_bytes()).hexdigest(),
              'predicted_camera_sha256': digest}
    return record, errors

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run, rows):
        results.append(result)
        if len(results) % 100 == 0:
            print(json.dumps({'completed_routes': len(results), 'at': time.time()}), flush=True)
records = [x[0] for x in results]
count = sum(x['transitions'] for x in records)

def stats(v):
    return {'n': len(v), 'mean': float(np.mean(v)),
            'quantile_levels': [0, .1, .25, .5, .75, .9, .99, 1],
            'quantiles': np.quantile(v, [0, .1, .25, .5, .75, .9, .99, 1]).tolist()}

summary = {'at': time.time(), 'dataset': a.dataset, 'routes': len(records),
           'instructions': sum(x['instructions'] for x in records), 'transitions': count,
           'raw_action_accuracy': sum(x['raw_correct'] for x in records) / count,
           'GT_free_scale_action_accuracy': sum(x['calibrated_correct'] for x in records) / count,
           'action_types_changed': sum(x['changed_action_types'] for x in records),
           'scale_search_boundary_routes': sum(x['scale_at_search_boundary'] for x in records),
           'raw_relative_magnitude_error': stats(np.concatenate([x[1][0] for x in results])),
           'GT_free_scale_relative_magnitude_error': stats(np.concatenate([x[1][1] for x in results])),
           'oracle_Sim3_relative_magnitude_error': stats(np.concatenate([x[1][2] for x in results])),
           'camera_ATE_route_statistics': {key: stats(np.array([x[key] for x in records]))
                for key in ['raw_rigid_aligned_camera_ATE_RMSE_m', 'GT_free_scale_rigid_aligned_camera_ATE_RMSE_m', 'oracle_Sim3_camera_ATE_RMSE_m']},
           'GT_free_scale_improves_camera_ATE_routes': sum(x['GT_free_scale_rigid_aligned_camera_ATE_RMSE_m'] < x['raw_rigid_aligned_camera_ATE_RMSE_m'] for x in records),
           'training_data_or_labels_modified': False,
           'scope': final_provenance['scope']}
reference = cache_summary['methods']['complete_proxy_pass']
if a.dataset == 'complete-proxy-pass-metric-v2':
    assert summary['routes'] == reference['routes'] and summary['transitions'] == reference['transitions']
    assert abs(summary['raw_action_accuracy'] - reference['raw_metric_accuracy']) < 1e-12
    assert abs(summary['GT_free_scale_action_accuracy'] - reference['GT_free_calibration_accuracy']) < 1e-12
(E / 'route_records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
assert summary['routes'] == final_provenance['routes']
assert summary['instructions'] == final_provenance['instructions']
assert summary['transitions'] == final_provenance['nonstop_actions']
summary['final_cohort_provenance'] = final_provenance
summary['scope'] = final_provenance['scope']
(E / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(E / 'exit.json').write_text(json.dumps({'success': True, 'at': time.time(), **{k.replace('.json', '') + '_sha256': hashlib.sha256((E / k).read_bytes()).hexdigest() for k in ['protocol.json', 'summary.json', 'route_records.json']}}, indent=2))
print(json.dumps(summary, indent=2), flush=True)
