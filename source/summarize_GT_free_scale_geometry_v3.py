"""Paired route bootstrap and an auditable ablation figure; no model changes."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['complete-optical-target-valid-v4'])
a = ap.parse_args()
L = Path(__file__).resolve().parents[1]
D = L / 'evidence/eval' / ('GT_free_scale_geometry_' + a.dataset + '_001')
receipt = json.loads((D / 'exit.json').read_text())
assert receipt['success']
for name in ['summary', 'route_records', 'protocol']:
    assert hashlib.sha256((D / (name + '.json')).read_bytes()).hexdigest() == receipt[name + '_sha256']
summary = json.loads((D / 'summary.json').read_text())
records = json.loads((D / 'route_records.json').read_text())
protocol = json.loads((D / 'protocol.json').read_text())
final_provenance = protocol['final_cohort_provenance']
assert final_provenance['dataset'] == a.dataset == summary['dataset']
assert final_provenance == summary['final_cohort_provenance']
assert final_provenance['routes'] == summary['routes'] == len(records)
assert final_provenance['instructions'] == summary['instructions']
assert final_provenance['nonstop_actions'] == summary['transitions']
raw = np.array([x['raw_rigid_aligned_camera_ATE_RMSE_m'] for x in records])
cal = np.array([x['GT_free_scale_rigid_aligned_camera_ATE_RMSE_m'] for x in records])
oracle = np.array([x['oracle_Sim3_camera_ATE_RMSE_m'] for x in records])
transitions = np.array([x['transitions'] for x in records])
correct_delta = np.array([x['calibrated_correct'] - x['raw_correct'] for x in records])
if a.dataset == 'complete-proxy-pass-metric-v2':
    prior = json.loads((L / 'evidence/report/metric_label_analysis_v2.json').read_text())
    assert np.isclose(np.median(raw), prior['geometry_route_medians']['SE3_ATE_RMSE_m'])
    assert np.isclose(np.median(oracle), prior['geometry_route_medians']['Sim3_ATE_RMSE_m'])
seed, replicates = 20261022, 10000
rng = np.random.default_rng(seed)
bootstrap_ATE, bootstrap_labels = [], []
for start in range(0, replicates, 100):
    index = rng.integers(0, len(records), size=(min(100, replicates - start), len(records)))
    bootstrap_ATE.extend((np.median(cal[index], axis=1) - np.median(raw[index], axis=1)).tolist())
    bootstrap_labels.extend((correct_delta[index].sum(1) / transitions[index].sum(1)).tolist())
out = {'at': time.time(), 'dataset': a.dataset, 'routes': len(records),
       'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'input_exit_sha256': hashlib.sha256((D / 'exit.json').read_bytes()).hexdigest(),
       'method': 'Paired nonparametric physical-route bootstrap. The same sampled route indices select both methods; repeated instructions are not independent units.',
       'scope': final_provenance['scope'] + ' Paired uncertainty is conditional on this selected training cohort. No threshold, label, scale fit or training configuration is selected using this interval.',
       'final_cohort_provenance': final_provenance,
       'seed': seed, 'replicates': replicates,
       'route_median_camera_ATE_RMSE_m': {'raw': float(np.median(raw)), 'GT_free_scale': float(np.median(cal)), 'oracle_Sim3': float(np.median(oracle))},
       'route_mean_camera_ATE_RMSE_m': {'raw': float(np.mean(raw)), 'GT_free_scale': float(np.mean(cal)), 'oracle_Sim3': float(np.mean(oracle))},
       'median_ATE_difference_GT_free_minus_raw_m': float(np.median(cal) - np.median(raw)),
       'median_ATE_difference_percentile_CI95_m': np.quantile(bootstrap_ATE, [.025, .975]).tolist(),
       'action_accuracy_difference_GT_free_minus_raw': float(correct_delta.sum() / transitions.sum()),
       'action_accuracy_difference_percentile_CI95': np.quantile(bootstrap_labels, [.025, .975]).tolist(),
       'calibration_improves_ATE_routes': int((cal < raw).sum()),
       'calibration_improves_ATE_fraction': float((cal < raw).mean()),
       'scale_boundary_route_ids': [x['route_id'] for x in records if x['scale_at_search_boundary']],
       'data_and_training_labels_modified': False}
path = D / 'paired_route_bootstrap_001.json'
assert not path.exists()
path.write_text(json.dumps(out, indent=2, allow_nan=False))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.7))
styles = [('raw', 'Raw ViPE scale', '#c05621', 'o'),
          ('GT_free_scale', 'Known-primitive scale (no GT scale)', '#805ad5', '^'),
          ('oracle_Sim3', 'Full-GT Sim3 (oracle)', '#2b6cb0', 's')]
q = np.array([.1, .25, .5, .75, .9, .99])
for key, label, color, marker in styles:
    item = summary[key + '_relative_magnitude_error']
    indices = [item['quantile_levels'].index(float(v)) for v in q]
    axes[0].plot(q * 100, np.array(item['quantiles'])[indices] * 100,
                 color=color, marker=marker, lw=1.5, label=label)
    values = {'raw': raw, 'GT_free_scale': cal, 'oracle_Sim3': oracle}[key]
    axes[1].plot(q * 100, np.quantile(values, q), color=color, marker=marker, lw=1.5)
axes[0].set(title='A  Local displacement magnitude', ylabel='Relative translation-magnitude error (%)')
axes[1].set(title='B  Full-route camera trajectory', ylabel='Route camera ATE RMSE (m)')
for ax in axes:
    ax.set(xlabel='Selected quantile (%)', yscale='log')
    ax.set_xticks([10, 25, 50, 75, 90, 99])
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(alpha=.18)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, .91))
fig.suptitle('Frozen known-primitive scale calibration: geometry and limits', y=.98, fontsize=13)
fig.text(.5, .045,
         f'{len(records):,} optically screened selected training routes. Raw and GT-free-scale ATE use evaluation-only rigid alignment; only the oracle fits GT scale.\n'
         'Selected quantiles are shown; all extreme failures remain in the analysis. '
         'GT-free calibration assumes known action units and one primitive per frame interval.\n'
         'This is a geometry ablation, not policy navigation SR. Training labels and data selection were not changed.', ha='center', fontsize=8)
fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.26, wspace=.30)
stem = 'GT_free_scale_geometry_' + a.dataset
outputs = []
for extension in ['png', 'pdf']:
    p = L / 'paper_assets' / (stem + '.' + extension)
    assert not p.exists()
    fig.savefig(p, dpi=180)
    outputs.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
plt.close(fig)
provenance = {'source_sha256': out['source_sha256'],
              'input_exit_sha256': out['input_exit_sha256'],
              'bootstrap_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'outputs': outputs, 'visual_review_pending': True}
(L / 'paper_assets' / (stem + '.provenance.json')).write_text(json.dumps(provenance, indent=2))
print(json.dumps(out, indent=2))
