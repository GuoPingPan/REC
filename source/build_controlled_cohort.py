"""Freeze a complete/proxy-pass matched cohort and exact-input label controls.

This is a disclosed research subset, never the full official training set, and
proxy-pass does not certify physical or swept-volume flight feasibility.
"""
from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, os, shutil, time

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
R=Path(a.root).resolve();U=Path('/mnt/data/users/kindel/uav')
OLD=U/'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official'
N=U/'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False));q.replace(p)
inv=json.loads((N/'inventory/routes.json').read_text())
official=json.loads((OLD/'inventory/official_train.json').read_text())['episodes'];by={str(e['episode_id']):e for e in official}
instructions=json.loads((N/'dataset/data/aerialvln-s/train_episode2instruction.json').read_text())
weights=N/'dataset/data/aerialvln-s/train_action_prob_weight.json'
verbs={1:('move forward',5,'units'),2:('turn left',15,'degrees'),3:('turn right',15,'degrees'),4:('ascend',2,'units'),5:('descend',2,'units'),6:('move left',5,'units'),7:('move right',5,'units')}
def text(act,k):
 return 'The next action is stop.' if act==0 else 'The next action is %s %s %s.'%(verbs[act][0],verbs[act][1]*k,verbs[act][2])
def segment(x):
 out=[];i=0
 while i<len(x):
  k=1
  while i+k<len(x) and x[i+k]==x[i] and k<3:k+=1
  out.append({'timestep':i,'action':text(x[i],k),'next_timestep':i+k if x[i] else i});i+=k
 return out
def joint(x,y):
 assert len(x)==len(y) and x[-1]==y[-1]==0
 left=[];right=[];i=0
 while i<len(x):
  k=1
  while i+k<len(x) and x[i+k]==x[i] and y[i+k]==y[i] and k<3:k+=1
  for out,z in [(left,x),(right,y)]:out.append({'timestep':i,'action':text(z[i],k),'next_timestep':i+k if z[i] else i})
  i+=k
 assert [{k:v for k,v in x.items() if k!='action'} for x in left]==[{k:v for k,v in x.items() if k!='action'} for x in right]
 return left,right
def inspect(item):
 rid,row=item;res=json.loads((N/'results'/f'{rid}.json').read_text())
 reason=row['termination_reason']
 outcome={'route_id':rid,'scene_id':row['scene_id'],'instructions':row['train_episode_ids'],'source_reason':reason,'recovery_status':res['status'],'source_frames':row['frame_count'],'official_frames':row['planned_frame_count']}
 if reason!='terminal_STOP':return outcome,None
 assert row['frame_count']==row['planned_frame_count']
 if res['status']!='completed':return outcome,None
 n=row['frame_count'];ep=by[row['train_episode_ids'][0]];gt=ep['actions'];assert len(gt)==n and gt[-1]==0
 m=json.loads(Path(row['manifest']).read_text());assert len(m['frames'])==n
 assert all(f['collision_fraction']<=.1 for f in m['frames'])
 aa=json.loads((N/'routes'/rid/'attempt-001/actions.json').read_text());rec=aa['actions_per_frame'];assert len(rec)==n and rec[-1]==0
 assert aa['GT_observed_transition_actions']==gt[:-1]
 rgb=N/'dataset/route_rgb'/rid;assert sha(rgb/'manifest.json')==res['RGB_manifest_sha256']
 assert {p for p in os.listdir(rgb) if p.endswith('.jpg')}=={'frame_%03d.jpg'%i for i in range(n)}
 for eid in row['train_episode_ids']:assert by[eid]['actions']==gt
 return outcome,{'route_id':rid,'scene_id':row['scene_id'],'episodes':row['train_episode_ids'],'frame_count':n,'GT':gt,'REC':rec,'rgb':str(rgb),'source_manifest_sha256':sha(row['manifest']),'RGB_manifest_sha256':res['RGB_manifest_sha256']}

assert not (R/'dataset/controlled-cohort-v1').exists(),'Preserve published dataset; use a new version.'
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:checked=list(pool.map(inspect,inv.items()))
outcomes=[x[0] for x in checked];used=[x[1] for x in checked if x[1]]
assert len(used)==2899 and sum(len(r['episodes']) for r in used)==7616
C=R/'dataset/controlled-cohort-v1';C.mkdir()
write(C/'route_inventory.json',[{k:v for k,v in r.items() if k not in ['GT','REC']} for r in used])
write(C/'all_source_outcomes.json',outcomes)
groups={k:{'samples':[],'idx':{},'instructions':{}} for k in ['gt-native-merge','rec-native-merge','gt-joint-boundary','rec-joint-boundary']}
correct=total=0
for r in used:
 gt=r['GT'];rec=r['REC'];correct+=sum(x==y for x,y in zip(gt[:-1],rec[:-1]));total+=len(gt)-1
 gtr,recr=joint(gt,rec)
 segs={'gt-native-merge':segment(gt),'rec-native-merge':segment(rec),'gt-joint-boundary':gtr,'rec-joint-boundary':recr}
 for name,g in groups.items():
  D=C/name
  for eid in r['episodes']:
   link=D/'Raw_data/aerialvln-s'/eid/'rgb';link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(r['rgb'],link.parent),target_is_directory=True)
   assert link.resolve()==Path(r['rgb']).resolve()
   g['idx'][eid]=[x['timestep'] for x in segs[name]];g['instructions'][eid]=instructions[eid]
   g['samples'].extend({'episode':eid,**s} for s in segs[name])
assert groups['gt-joint-boundary']['idx']==groups['rec-joint-boundary']['idx']
for g in groups.values():
 assert sum(x['action']=='The next action is stop.' for x in g['samples'])==7616
 assert len(g['idx'])==7616
digests={}
for name,g in groups.items():
 A=C/name/'data/aerialvln-s'
 for f,obj in [('train_merged_triple.json',g['samples']),('train_episodes2idx_merge.json',g['idx']),('train_episode2instruction.json',g['instructions'])]:write(A/f,obj)
 shutil.copy2(weights,A/'train_action_prob_weight.json')
 digests[name]={p.name:sha(p) for p in A.glob('*.json')}
# Strong paired-input proof: IDs, current frame, segment duration, history keyframe
# list, language, and resolved RGB directories are identical; only answers differ.
g=groups['gt-joint-boundary'];h=groups['rec-joint-boundary'];assert len(g['samples'])==len(h['samples'])
for left,right in zip(g['samples'],h['samples']):assert {k:v for k,v in left.items() if k!='action'}=={k:v for k,v in right.items() if k!='action'}
summary={'created_at':time.time(),'status':'built_structural_checks_passed_official_loader_not_yet_run','source_routes':len(inv),'source_instructions':sum(len(x['train_episode_ids']) for x in inv.values()),'usable_routes':len(used),'usable_instructions':7616,'physical_RGB_frames':sum(r['frame_count'] for r in used),'nonstop_actions':total,'correct_nonstop_actions':correct,'nonstop_agreement':correct/total,'sample_counts':{k:len(g['samples']) for k,g in groups.items()},'source_outcome_route_counts':dict(collections.Counter((x['source_reason']+'|'+x['recovery_status']) for x in outcomes)),'all_samples_endpoint_STOP_count':7616,'native_merge_pair_input_equality':False,'joint_boundary_pair_input_equality':True,'joint_boundary_uses_GT_for_control_design':True,'joint_boundary_is_GT_free_deployment_method':False,'cohort_selection':'Complete GT-native RGB with no recorded official depth proxy flag and usable ViPE output. Same source videos and IDs in every group. This is not a full official training reproduction.','physical_legality_certified':False,'collision_scope':'Recorded GT observation depth proxy only; swept path and true geometry contacts not certified.','source_inventory_sha256':sha(N/'inventory/routes.json'),'source_official_GT_sha256':sha(OLD/'inventory/official_train.json'),'annotation_sha256':digests,'old_datasets_modified':False}
write(C/'validation_report.json',summary);write(R/'report/controlled_cohort.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
