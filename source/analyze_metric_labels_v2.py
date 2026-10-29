"""Independent held-label scoring of frozen raw-metric recovered actions."""
from pathlib import Path
import argparse,collections,concurrent.futures,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from primitive_calibration import TRANSLATIONS,ROTATIONS
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--dataset',default='complete-optical-screened-metric-v3');a=ap.parse_args();R=Path(a.root).resolve()
U=Path('/mnt/data/users/kindel/uav');N=U/'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery';O=U/'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official';C=R/'dataset'/a.dataset
by={str(e['episode_id']):e for e in json.loads((O/'inventory/official_train.json').read_text())['episodes']};rows=json.loads((C/'route_inventory.json').read_text());validation=json.loads((C/'validation_report.json').read_text());assert len(rows)==validation['usable_routes']
def run(row):
 rid=row['route_id'];label=json.loads((C/'route_labels'/f'{rid}.json').read_text());pred=np.array(label['GT_free_REC_actions'][:-1]);ep=by[row['episodes'][0]];gt=np.array(ep['actions'][:-1]);assert len(gt)==len(pred)
 cm=np.zeros((7,7),np.int64);np.add.at(cm,(gt-1,pred-1),1)
 gbound=set((np.flatnonzero(gt[1:]!=gt[:-1])+1).tolist());pbound=set((np.flatnonzero(pred[1:]!=pred[:-1])+1).tolist())
 p=np.array(ep['start_position'],float);w,x,y,z=ep['start_rotation'];rot=Rotation.from_quat([x,y,z,w]).as_matrix()
 for act in pred:p=p+rot@TRANSLATIONS[act-1];rot=rot@ROTATIONS[act-1]
 distances=[float(np.linalg.norm(p[:2]-np.array(by[eid]['goals'][0]['position'])[:2])) for eid in row['episodes']]
 geo=json.loads((N/'routes'/rid/'attempt-001/geometry.json').read_text());assert 'camera offset 0.5 m' in geo['GT_reference']
 return {'route_id':rid,'scene_id':row['scene_id'],'transitions':len(gt),'correct':int(np.trace(cm)),'confusion':cm,'boundaries_true_positive':len(gbound&pbound),'GT_boundaries':len(gbound),'REC_boundaries':len(pbound),'instruction_count':len(row['episodes']),'kinematic_endpoint_successes':sum(x<=20 for x in distances),'endpoint_XY_distance_mean_m':float(np.mean(distances)),'SE3_ATE_RMSE_m':geo['se3']['ate_m']['rmse'],'Sim3_ATE_RMSE_m':geo['sim3']['ate_m']['rmse'],'SE3_RPE_translation_RMSE_m':geo['se3']['rpe_step_translation_m']['rmse'],'Sim3_RPE_translation_RMSE_m':geo['sim3']['rpe_step_translation_m']['rmse']}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(run,rows))
cm=sum((x['confusion'] for x in results),np.zeros((7,7),np.int64));support=cm.sum(1);prec=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(7),where=cm.sum(0)>0);recall=np.divide(np.diag(cm),support,out=np.zeros(7),where=support>0);f1=np.divide(2*prec*recall,prec+recall,out=np.zeros(7),where=prec+recall>0)
def aggregate(rr):
 n=sum(x['transitions'] for x in rr)
 return {'routes':len(rr),'transitions':n,'micro_accuracy':sum(x['correct'] for x in rr)/n if n else None,'route_mean_accuracy':float(np.mean([x['correct']/x['transitions'] for x in rr])) if rr else None,'exact_action_sequence_routes':sum(x['correct']==x['transitions'] for x in rr),'kinematic_endpoint_successes':sum(x['kinematic_endpoint_successes'] for x in rr),'instruction_count':sum(x['instruction_count'] for x in rr)}
summary={'completed_at':time.time(),'cohort':a.dataset,'physical_legality_certified':False,'geometry_reference':'Predicted optical-camera poses compared with calibrated GT camera trajectory (body telemetry plus known0.5mextrinsic), not a body-center ATE. Per-step RPE is per navigation observation interval, not per second.','overall':aggregate(results),'confusion_GT_rows_REC_columns_actions1to7':cm.tolist(),'per_class':{str(i+1):{'support':int(support[i]),'precision':float(prec[i]),'recall':float(recall[i]),'F1':float(f1[i])} for i in range(7)},'macro_F1':float(f1.mean()),'geometry_route_medians':{k:float(np.median([x[k] for x in results])) for k in ['SE3_ATE_RMSE_m','Sim3_ATE_RMSE_m','SE3_RPE_translation_RMSE_m','Sim3_RPE_translation_RMSE_m']},'by_length':{},'by_scene':{str(s):aggregate([x for x in results if x['scene_id']==s]) for s in sorted(set(x['scene_id'] for x in results))},'STOP_is_not_in_accuracy':'STOP is a shared known demonstration boundary, not an estimated motion class.','endpoint_protocol':'Recorded raw REC action sequence integrated from official initial pose; final XY distance<=20m. No rendering/collision guards, not official closed-loop navigation SR.'}
for label,lo,hi in [('1to50',1,50),('51to100',51,100),('101to200',101,200),('201plus',201,100000)]:summary['by_length'][label]=aggregate([x for x in results if lo<=x['transitions']<=hi])
tp=sum(x['boundaries_true_positive'] for x in results);g=sum(x['GT_boundaries'] for x in results);p=sum(x['REC_boundaries'] for x in results);summary['merge_boundary_exact_step']={'true_positive':tp,'GT_boundaries':g,'REC_boundaries':p,'precision':tp/p,'recall':tp/g,'F1':2*tp/(g+p)}
out=R/'report'/('metric_label_analysis_'+a.dataset+'.json');assert not out.exists();out.write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
