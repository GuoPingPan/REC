"""Frozen RGB-pose-only calibration, with GT accessed only for scoring afterward."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, time
import numpy as np
from primitive_calibration import predict, verify_geometry, PROTOCOL

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
R=Path(a.root).resolve();N=Path('/mnt/data/users/kindel/uav/experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery')
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
E=R/'eval/primitive_calibration_v1';assert not E.exists(),'Use new experiment for protocol changes.';E.mkdir()
write(E/'protocol.json',PROTOCOL);write(E/'geometry_unit_verification.json',verify_geometry())
inv=json.loads((N/'inventory/routes.json').read_text())
def run(item):
 rid,row=item;res=json.loads((N/'results'/f'{rid}.json').read_text())
 if res['status']!='completed':return None
 p=N/'routes'/rid/'attempt-001/prediction/predicted_camera_c2w.npz'
 poses=np.load(p,allow_pickle=False)['data'].astype(float)
 # Predictions and protocol are frozen before looking up any action reference.
 raw=predict(poses,False);cal=predict(poses,True)
 path=N/'routes'/rid/'attempt-001/actions.json';labels=json.loads(path.read_text())
 gt=np.array(labels['GT_observed_transition_actions']);oracle=np.array(labels['observed_transition_actions'])
 assert len(raw['actions'])==len(cal['actions'])==len(gt)==len(poses)-1
 x={'route_id':rid,'scene_id':row['scene_id'],'full_proxy_pass':row['termination_reason']=='terminal_STOP','instructions':len(row['train_episode_ids']),'transitions':len(gt),'raw_metric_correct':int((raw['actions']==gt).sum()),'GT_free_calibration_correct':int((cal['actions']==gt).sum()),'GT_aligned_oracle_correct':int((oracle==gt).sum()),'calibrated_scale':cal['scale'],'oracle_GT_scale':res['sim3_scale'],'scale_at_boundary':cal['scale_at_search_boundary'],'confidence_median':float(np.median(cal['confidence'])),'objective':cal['objective'],'predicted_poses_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'GT_free_actions':cal['actions'].tolist()}
 write(E/'routes'/f'{rid}.json',x);return x
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for x in pool.map(run,inv.items()):
  if x:rows.append(x)
  if len(rows) and len(rows)%100==0:write(E/'progress.json',{'completed_routes':len(rows),'time':time.time()})
summary={'complete':True,'protocol':PROTOCOL,'methods':{},'scope':'New diagnostic using existing ViPE outputs. No model retraining; arbitrary real-world cadence is out of scope. GT is used for scoring and proxy-pass cohort definition, not scale estimation.','source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method_script_sha256':hashlib.sha256(Path(__file__).with_name('primitive_calibration.py').read_bytes()).hexdigest()}
for scope,rr in [('all_usable',[x for x in rows]),('complete_proxy_pass',[x for x in rows if x['full_proxy_pass']])]:
 count=sum(x['transitions'] for x in rr)
 summary['methods'][scope]={'routes':len(rr),'instructions':sum(x['instructions'] for x in rr),'transitions':count,'raw_metric_accuracy':sum(x['raw_metric_correct'] for x in rr)/count,'GT_free_calibration_accuracy':sum(x['GT_free_calibration_correct'] for x in rr)/count,'GT_aligned_oracle_accuracy':sum(x['GT_aligned_oracle_correct'] for x in rr)/count,'scale_boundary_routes':sum(x['scale_at_boundary'] for x in rr),'median_scale':float(np.median([x['calibrated_scale'] for x in rr]))}
write(E/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
