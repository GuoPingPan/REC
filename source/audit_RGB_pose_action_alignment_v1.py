"""Complete source-frame/action temporal alignment audit; CPU, no simulator calls.

A positive audit proves internal correspondence, not physical legality or policy
performance. Frozen v2 data remains on hold while optical screening completes.
"""
from pathlib import Path
import argparse,ast,collections,hashlib,importlib.util,json,os,time,traceback,re,math
import numpy as np
from scipy.spatial.transform import Rotation
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root).resolve();E=R/'eval/RGB_pose_action_alignment_001';E.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=E/n;q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
C=R/'dataset/complete-proxy-pass-metric-v2';inventory=C/'route_inventory.json';rows=json.loads(inventory.read_text());V=json.loads((C/'validation_report.json').read_text());assert len(rows)==V['usable_routes']==2899
S=Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source']);p=S/'eval.py';text=p.read_text(errors='replace');fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='convert_text_to_action');ns={'re':re,'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns);parse=ns['convert_text_to_action']
spec=importlib.util.spec_from_file_location('native',R/'source/gt_native_replay_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.S=S;m.Q=E;m.official_imports()
import torch
torch.set_num_threads(2)
O=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official');original=O/'inventory/official_train.json';by={str(x['episode_id']):x for x in json.loads(original.read_text())['episodes']}
protocol={'pid':os.getpid(),'source_sha256':sha(Path(__file__)),'inventory_sha256':sha(inventory),'official_annotations_sha256':sha(original),'official_action_parser_sha256':sha(S/'eval.py'),'scope':'Every selected v2source RGB/frame manifest vs official GT action integration, and every merged label vs its primitive sequence. No simulator, policy or raw data changes.','expected_routes':2899,'expected_instructions':7616,'expected_RGB_frames':330315,'started_at':time.time()};save('protocol.json',protocol)
route_proof=[];targets={};success=False;group_results={}
try:
 for row in rows:
  rid=row['route_id'];episode=by[str(row['episodes'][0])];poses=m.integrate(episode,episode['actions']);ref=np.asarray(episode['reference_path'],float);assert len(poses)==row['frame_count'];referr=float(np.max(np.abs(poses[:,:3]-ref[:,:3])));assert referr<1e-7
  rgbmanifest=Path(row['rgb'])/'manifest.json';assert sha(rgbmanifest)==row['RGB_manifest_sha256'];rgb=json.loads(rgbmanifest.read_text());source=Path(rgb['source_video']).parent/'manifest.json';assert sha(source)==row['source_manifest_sha256']==rgb['source_manifest_sha256'];src=json.loads(source.read_text());assert src['route_id']==rid and src['frame_count']==len(src['frames'])==rgb['frame_count']==len(rgb['frames'])==len(poses)
  actual=np.array([f['actual_body_pose'] for f in src['frames']],float);desired=np.array([f['desired_body_pose'] for f in src['frames']],float);poserr=float(np.linalg.norm(actual[:,:3]-poses[:,:3],axis=1).max());commanderr=float(np.linalg.norm(desired[:,:3]-poses[:,:3],axis=1).max());rot_err=float(np.degrees((Rotation.from_quat(poses[:,3:]).inv()*Rotation.from_quat(actual[:,3:])).magnitude()).max());assert poserr<.01 and commanderr<1e-7 and rot_err<.01
  metadata_fallback=0
  for i,(f,g) in enumerate(zip(src['frames'],rgb['frames'])):
   assert f['index']==g['index']==i and f['sha256_rgb']==g['source_rgb_sha256']
   if 'planned_gt_action' in f:recorded=f['planned_gt_action']
   else:recorded=f['action_after_observation'];metadata_fallback+=1
   assert recorded==episode['actions'][i],(rid,i,recorded,episode['actions'][i])
   assert f['image_camera_position_error_m']<.02
  rec=json.loads((C/'route_labels'/(rid+'.json')).read_text())['GT_free_REC_actions'];assert len(rec)==len(poses) and rec[-1]==episode['actions'][-1]==0
  for eid in row['episodes']:
   assert by[eid]['actions']==episode['actions'];targets[eid]={'GT':episode['actions'],'REC':rec,'frames':len(poses)}
  rr={'route_id':rid,'frames':len(poses),'instruction_count':len(row['episodes']),'official_integrated_vs_reference_max_error_m':referr,'source_desired_vs_GT_max_error_m':commanderr,'source_actual_vs_GT_max_position_error_m':poserr,'source_actual_vs_GT_max_rotation_error_deg':rot_err,'every_source_RGB_pixel_hash_matches_frame_manifest':True,'every_source_action_matches_official_frame':True,'source_frames_using_action_metadata_fallback':metadata_fallback,'source_manifest_sha256':sha(source),'RGB_manifest_sha256':sha(rgbmanifest)};route_proof.append(rr)
  if len(route_proof)%100==0:save('progress.json',{'phase':'source_RGB_pose_action','completed_routes':len(route_proof),'expected_routes':len(rows),'last_route':rid,'time':time.time()});print('ALIGNMENT_ROUTES',len(route_proof),flush=True)
 save('route_proofs.json',route_proof);assert len(targets)==7616 and sum(x['frames'] for x in route_proof)==330315
 for group,expected in V['sample_counts'].items():
  D=C/group/'data/aerialvln-s'
  for filename,h in V['annotation_sha256'][group].items():assert sha(D/filename)==h
  samples=json.loads((D/'train_merged_triple.json').read_text());idx=json.loads((D/'train_episodes2idx_merge.json').read_text());assert len(samples)==expected and set(idx)==set(targets)
  grouped=collections.defaultdict(list)
  for sample in samples:grouped[sample['episode']].append(sample)
  assert set(grouped)==set(targets)
  target_name='GT' if group.startswith('gt-') else 'REC';counts=collections.Counter()
  for eid,items in grouped.items():
   items.sort(key=lambda x:x['timestep']);target=targets[eid][target_name];expanded=[];cursor=0
   assert idx[eid]==[x['timestep'] for x in items]
   for sample in items:
    t=sample['timestep'];assert t==cursor and 0<=t<targets[eid]['frames'];actions=parse(sample['action']);assert actions and all(x in range(8) for x in actions)
    if actions==[0]:assert t==len(target)-1 and sample['next_timestep']==t and sample is items[-1];cursor=t+1
    else:
     assert 1<=len(actions)<=3 and sample['next_timestep']==t+len(actions);assert target[t:t+len(actions)]==actions;cursor=t+len(actions)
    # The actual public dataset constructs frame candidates with t<=current_t.
    # Since each index list equals these increasing starts, current_t is included,
    # future indices are excluded, and public sampling explicitly keeps the latest.
    assert t in idx[eid];expanded.extend(actions);counts['samples_checked']+=1
   assert expanded==target and cursor==len(target);counts['exact_primitive_sequences_checked']+=1
  group_results[group]=dict(counts);save('group_proofs.json',group_results);print('ALIGNMENT_GROUP',group,dict(counts),flush=True)
 # Preserve actual released selection implementations as evidence. No imitation
 # of a different loader and no claim that every image tensor was reprocessed.
 ds_source=S/'llava/data/dataset.py';tree=ast.parse(ds_source.read_text(errors='replace'));cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LazyUpdateTripleMergeAirVLNDataset');getitem=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__getitem__');code=ast.unparse(getitem);assert "sources[0]['timestep']" in code and 'if t <= timestep' in code and 'next_timestep' not in code
 mm=S/'llava/mm_utils.py';tree=ast.parse(mm.read_text(errors='replace'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='vlnce_frame_sampling');code=ast.unparse(fn);assert 'latest_frame = frames[-1]' in code and '[latest_frame]' in code
 save('source_selection_proof.json',{'dataset_source_sha256':sha(ds_source),'frame_sampler_source_sha256':sha(mm),'uses_current_timestep_as_observation':True,'future_frame_candidates_excluded':True,'current_frame_explicitly_preserved_by_sampler':True,'scope':'Actual pinned public AST plus complete annotation-clock checks. Representative actual image tensor checks are a separate existing preflight, not replaced here.'});success=True
except BaseException:
 save('failure.json',{'traceback':traceback.format_exc(),'completed_routes':len(route_proof),'time':time.time()});raise
finally:
 save('exit.json',{'success':success,'completed_routes':len(route_proof),'instructions':len(targets),'source_RGB_frames':sum(x['frames'] for x in route_proof),'max_RGB_source_position_error_vs_GT_m':max([x['source_actual_vs_GT_max_position_error_m'] for x in route_proof],default=0),'max_RGB_source_rotation_error_vs_GT_deg':max([x['source_actual_vs_GT_max_rotation_error_deg'] for x in route_proof],default=0),'groups':group_results,'source_RGB_and_labels_modified':False,'physical_legality_certified':False,'formal_training_hold_cleared':False,'finished_at':time.time()})
