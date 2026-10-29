"""Corrected matched controls: complete GT proxy-pass RGB and raw ViPE scale.

Candidate cohort inventory is independently validated by the v1 audit. Shared
RGB folders can contain unselected episodes; only frozen annotation IDs train.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,os,shutil,time
import numpy as np
from primitive_calibration import predict
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root).resolve()
U=Path('/mnt/data/users/kindel/uav');N=U/'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery';O=U/'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official'
old_inventory=R/'dataset/controlled-cohort-v1/route_inventory.json';rows=json.loads(old_inventory.read_text());assert len(rows)==2899
C=R/'dataset/complete-proxy-pass-metric-v2';assert not C.exists();C.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False));q.replace(p)
instructions=json.loads((N/'dataset/data/aerialvln-s/train_episode2instruction.json').read_text());by={str(e['episode_id']):e for e in json.loads((O/'inventory/official_train.json').read_text())['episodes']}
verbs={1:('move forward',5,'units'),2:('turn left',15,'degrees'),3:('turn right',15,'degrees'),4:('ascend',2,'units'),5:('descend',2,'units'),6:('move left',5,'units'),7:('move right',5,'units')}
def text(act,k):return 'The next action is stop.' if act==0 else 'The next action is %s %s %s.'%(verbs[act][0],verbs[act][1]*k,verbs[act][2])
def segments(x,y=None):
 out=[];other=[];i=0
 while i<len(x):
  k=1
  while i+k<len(x) and k<3 and x[i+k]==x[i] and (y is None or y[i+k]==y[i]):k+=1
  out.append({'timestep':i,'action':text(x[i],k),'next_timestep':i+k if x[i] else i})
  if y is not None:other.append({'timestep':i,'action':text(y[i],k),'next_timestep':i+k if y[i] else i})
  i+=k
 return (out,other) if y is not None else out
def process(row):
 rid=row['route_id'];p=N/'routes'/rid/'attempt-001/prediction/predicted_camera_c2w.npz'
 poses=np.load(p,allow_pickle=False)['data'];prediction=predict(poses,calibrate=False)
 # Only after prediction, retrieve GT for the baseline, scoring, and matched control.
 gt=by[row['episodes'][0]]['actions'];rec=prediction['actions'].tolist()+[0]
 assert len(gt)==len(rec)==row['frame_count'] and gt[-1]==0
 assert sha(Path(row['rgb'])/'manifest.json')==row['RGB_manifest_sha256']
 for eid in row['episodes']:assert by[eid]['actions']==gt
 jj=segments(gt,rec)
 write(C/'route_labels'/f'{rid}.json',{'route_id':rid,'GT_free_REC_actions':rec,'source_prediction_sha256':sha(p),'scale':1.0,'camera_offset_m':.5,'STOP_source':'End of known-complete demonstration; shared boundary prior, not a learned stopping estimator.','GT_poses_used_for_REC_labels':False,'GT_actions_used_for_REC_labels':False})
 return row,{'gt-native-merge':segments(gt),'rec-native-merge':segments(rec),'gt-joint-boundary':jj[0],'rec-joint-boundary':jj[1]},sum(x==y for x,y in zip(gt[:-1],rec[:-1]))
groups={k:{'samples':[],'idx':{},'instructions':{}} for k in ['gt-native-merge','rec-native-merge','gt-joint-boundary','rec-joint-boundary']};correct=total=0
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for row,segs,ncorrect in pool.map(process,rows):
  correct+=ncorrect;total+=row['frame_count']-1
  for name,g in groups.items():
   for eid in row['episodes']:
    g['idx'][eid]=[x['timestep'] for x in segs[name]];g['instructions'][eid]=instructions[eid];g['samples'].extend({'episode':eid,**x} for x in segs[name])
left=groups['gt-joint-boundary'];right=groups['rec-joint-boundary'];assert left['idx']==right['idx'] and left['instructions']==right['instructions']
assert len(left['samples'])==len(right['samples'])
for l,r in zip(left['samples'],right['samples']):assert {k:v for k,v in l.items() if k!='action'}=={k:v for k,v in r.items() if k!='action'}
digests={}
for name,g in groups.items():
 D=C/name;A=D/'data/aerialvln-s'
 assert len(g['idx'])==7616 and sum(s['action']=='The next action is stop.' for s in g['samples'])==7616
 for f,obj in [('train_merged_triple.json',g['samples']),('train_episodes2idx_merge.json',g['idx']),('train_episode2instruction.json',g['instructions'])]:write(A/f,obj)
 shutil.copy2(N/'dataset/data/aerialvln-s/train_action_prob_weight.json',A/'train_action_prob_weight.json')
 rgb=D/'Raw_data/aerialvln-s';rgb.parent.mkdir(parents=True);rgb.symlink_to(os.path.relpath(N/'dataset/Raw_data/aerialvln-s',rgb.parent),target_is_directory=True)
 for eid in g['idx']:assert (rgb/eid/'rgb').is_dir()
 digests[name]={p.name:sha(p) for p in A.glob('*.json')}
summary={'created_at':time.time(),'status':'built_structural_checks_passed_official_loader_not_yet_run','source_routes':3834,'source_instructions':10097,'usable_routes':2899,'usable_instructions':7616,'physical_RGB_frames':sum(r['frame_count'] for r in rows),'nonstop_actions':total,'correct_nonstop_actions':correct,'nonstop_agreement':correct/total,'sample_counts':{k:len(g['samples']) for k,g in groups.items()},'all_samples_endpoint_STOP_count':7616,'native_merge_pair_input_equality':False,'joint_boundary_pair_input_equality':True,'joint_boundary_uses_GT_for_control_design':True,'GT_poses_used_for_REC_labels':False,'GT_actions_used_for_REC_labels':False,'scale_estimation':'None; original ViPE metric scale and calibrated 0.5m optical/body lever arm.','one_primitive_per_frame_interval_assumed':True,'terminal_policy':'Shared known-complete demonstration endpoint is STOP, never a collision-truncated endpoint.','cohort_selection':'Same complete observed GT-native routes passing official observation depth proxy, with valid ViPE output. Not full official GT reproduction.','physical_legality_certified':False,'source_candidate_inventory_sha256':sha(old_inventory),'annotation_sha256':digests,'shared_RGB_root_contains_other_unselected_episodes':True,'training_IDs_controlled_by_frozen_annotations':True,'old_datasets_modified':False}
write(C/'route_inventory.json',rows);write(C/'validation_report.json',summary);write(R/'report/metric_cohort_v2.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
