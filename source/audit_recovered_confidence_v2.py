"""Exploratory confidence/error diagnostic only; never rewrites training labels.

The fixed geometry score is not a calibrated correctness probability. Thresholds
are declared before reading GT labels. They are not selected on validation SR.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, time
import numpy as np
from primitive_calibration import predict

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--dataset',default='complete-optical-screened-metric-v3')
a=ap.parse_args();R=Path(a.root).resolve()
U=Path('/mnt/data/users/kindel/uav')
N=U/'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery'
O=U/'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official'
C=R/'dataset'/a.dataset
validation=json.loads((C/'validation_report.json').read_text())
denominator=int(validation['nonstop_actions'])
expected_correct=int(validation['correct_nonstop_actions'])
OUT=R/'eval'/('recovered_confidence_'+a.dataset+'_001')
OUT.mkdir(exist_ok=False)
thresholds=[0.,.05,.10,.20,.30,.40,.50,.60]
protocol={'scope':'Exploratory training-route label diagnostic, not navigation evaluation or a new training dataset.',
          'thresholds_declared_before_GT_read':thresholds,
          'score':'exp(-min(cost_best,50))*(cost_second-cost_best)/(1+cost_second-cost_best)',
          'score_is_calibrated_probability':False,'actions_use_GT':False,'training_data_modified':False,
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'prediction_source_sha256':hashlib.sha256((R/'source/primitive_calibration.py').read_bytes()).hexdigest(),
          'started_at':time.time()}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
rows=json.loads((C/'route_inventory.json').read_text())
GT={str(e['episode_id']):e for e in json.loads((O/'inventory/official_train.json').read_text())['episodes']}

def work(row):
    rid=row['route_id']
    p=N/'routes'/rid/'attempt-001/prediction/predicted_camera_c2w.npz'
    with np.load(p,allow_pickle=False) as f:poses=f['data']
    prediction=predict(poses,calibrate=False)
    pred=prediction['actions'];confidence=prediction['confidence']
    frozen=json.loads((C/'route_labels'/(rid+'.json')).read_text())
    assert np.array_equal(pred,np.array(frozen['GT_free_REC_actions'][:-1]))
    # GT is read only here for scoring, after all prediction/score computation.
    gold=np.array(GT[row['episodes'][0]]['actions'][:-1])
    assert len(gold)==len(pred)==len(confidence) and np.isfinite(confidence).all()
    matrices=[]
    for threshold in thresholds:
        mask=confidence>=threshold
        cm=np.zeros((7,7),dtype=np.int64)
        np.add.at(cm,(gold[mask]-1,pred[mask]-1),1)
        matrices.append(cm)
    return {'route_id':rid,'scene_id':int(row['scene_id']),'transitions':len(pred),
            'correct':int((pred==gold).sum()),'mean_score':float(confidence.mean()),
            'matrices':np.array(matrices),'scores':confidence,'correctness':pred==gold}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(work,rows))
all_cm=sum((x['matrices'] for x in results),np.zeros((len(thresholds),7,7),dtype=np.int64))
assert int(all_cm[0].sum())==denominator and int(np.trace(all_cm[0]))==expected_correct
def summarize(cm,denominator):
    n=int(cm.sum());correct=int(np.trace(cm));g=cm.sum(1);p=cm.sum(0)
    precision=np.divide(np.diag(cm),p,out=np.zeros(7),where=p>0)
    recall=np.divide(np.diag(cm),g,out=np.zeros(7),where=g>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(7),where=precision+recall>0)
    return {'retained':n,'correct':correct,'errors':n-correct,'coverage':n/denominator,
            'agreement':correct/n if n else None,'macro_F1_retained_only':float(f1.mean()),
            'support_by_GT_class':g.tolist(),'confusion_GT_rows_REC_columns':cm.tolist()}
summary={'scope':protocol['scope'],'thresholds':[],
    'routes':len(results),'transitions':denominator,
    'GT_used_only_for_scoring':True,'training_annotations_modified':False,
    'official_eval_not_used':True,'probability_calibration_not_claimed':True,
    'interpretation':'Coverage and accuracy on retained transitions do not imply complete valid trajectories, usable terminal labels, collision freedom, or improved downstream policy SR.',
    'finished_at':time.time()}
for i,t in enumerate(thresholds):
    summary['thresholds'].append({'threshold':t,**summarize(all_cm[i],denominator)})
summary['by_scene']={}
for scene in sorted({x['scene_id'] for x in results}):
    rr=[x for x in results if x['scene_id']==scene]
    cms=sum((x['matrices'] for x in rr),np.zeros_like(all_cm));n=sum(x['transitions'] for x in rr)
    summary['by_scene'][str(scene)]=[{'threshold':t,**summarize(cms[i],n)} for i,t in enumerate(thresholds)]
score=np.concatenate([x['scores'] for x in results]);correct=np.concatenate([x['correctness'] for x in results])
summary['score_by_correctness']={name:{'count':int(mask.sum()),'mean':float(score[mask].mean()) if mask.any() else None,
    'quantiles':np.quantile(score[mask],[0,.1,.25,.5,.75,.9,1]).tolist() if mask.any() else None} for name,mask in [('correct',correct),('incorrect',~correct)]}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
(OUT/'route_summaries.json').write_text(json.dumps([{k:v for k,v in x.items() if k not in ['scores','correctness','matrices']} for x in results],indent=2))
for x in summary['thresholds']:
    print(json.dumps({k:v for k,v in x.items() if k not in ['confusion_GT_rows_REC_columns','support_by_GT_class']}),flush=True)
