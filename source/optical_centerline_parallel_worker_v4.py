"""One scene in an explicitly owned four-simulator group on one existing GPU.

The measurement implementation is the unchanged, hashed OpticalProbe v1.
Reused completed route JSON remains byte-identical to its original attempt.
"""
from pathlib import Path
import argparse,json,time,os,fcntl,subprocess,hashlib,importlib.util,traceback,shutil
import numpy as np
from optical_centerline_probe_v1 import OpticalProbe
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--scene',required=True,type=int);ap.add_argument('--slot',required=True,type=int);a=ap.parse_args();R=Path(a.root).resolve();plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text());GROUP=plan_path.parent;E=GROUP/'scenes'/f'scene_{a.scene}'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert GROUP.resolve().is_relative_to((R/'eval').resolve()) and plan['gpu']==0 and plan['simulator_slots']==4 and a.slot in range(4)
assert sha(R/'source/optical_centerline_probe_v1.py')==plan['probe_sha256'] and sha(Path(__file__))==plan['worker_source_sha256']
parent=Path('/proc')/str(plan['coordinator_pid'])/'cmdline';assert parent.exists() and 'optical_centerline_parallel_controller_v4.py' in parent.read_bytes().decode().replace('\0',' ')
# The coordinator holds the existing exclusive GPU lock for this entire group.
gpu_lock=Path('/tmp/aeroact_official_policy_gpu_0.lock').open('a')
try:
 fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:pass
else:raise RuntimeError('Coordinator does not hold exclusive GPU ownership')
E.mkdir(parents=True,exist_ok=False);(E/'routes').mkdir();(E/'diagnostic_views').mkdir()
def save(n,d):
 p=E/n;q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
spec=importlib.util.spec_from_file_location('native',R/'source/gt_native_replay_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.S=Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source']);m.Q=E;m.official_imports()
inventory=R/'dataset/complete-proxy-pass-metric-v2/route_inventory.json';assert sha(inventory)==plan['route_inventory_sha256'];rows=sorted([x for x in json.loads(inventory.read_text()) if int(x['scene_id'])==a.scene],key=lambda x:x['route_id']);assert rows
O=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official');annotation=O/'inventory/official_train.json';assert sha(annotation)==plan['official_annotations_sha256'];by={str(x['episode_id']):x for x in json.loads(annotation.read_text())['episodes']}
protocol={'pid':os.getpid(),'coordinator_pid':plan['coordinator_pid'],'group_nonce':plan['nonce'],'host':plan['host'],'gpu':0,'shared_gpu_slots':4,'slot':a.slot,'scene':a.scene,'expected_routes':len(rows),'expected_instructions':sum(len(x['episodes']) for x in rows),'source_sha256':sha(Path(__file__)),'probe_source_sha256':plan['probe_sha256'],'cohort_inventory_sha256':plan['route_inventory_sha256'],'official_annotations_sha256':plan['official_annotations_sha256'],'measurement_protocol':'Exactly the unchanged OpticalProbe v1; camera/body tolerances and positive/uncertain rules unchanged.','resumption':'Reuse only complete validated route records from explicitly named previous attempts, with byte hash preserved.','source_training_data_modified':False,'official_eval_modified':False,'new_paid_nodes':0,'started_at':time.time()};save('protocol.json',protocol)
def verify_record(rr,row):
 ep=by[str(row['episodes'][0])];poses=np.asarray(ep['reference_path'],float)[:,:3];trans=int((np.linalg.norm(np.diff(poses,axis=0),axis=1)>1e-4).sum())
 assert rr['route_id']==row['route_id'] and set(rr['episodes'])==set(row['episodes']) and rr['frames']==row['frame_count']
 assert rr['tested_segments']==len(rr['segments'])==trans and rr['stationary_transitions']+trans==len(poses)-1 and rr['GT_action_vs_reference_max_error_m']<1e-7
 assert rr['confirmed_optical_conflict_segments']==sum(x['repeat_confirmed_rendered_centerline_conflict'] for x in rr['segments'])
 assert rr['unresolved_segments']==sum(x['unresolved'] for x in rr['segments'])
 for x in rr['segments']:
  assert x['camera_position_error_m']<.001 and x['camera_forward_dot_GT']>.99999
  if x['repeat_confirmed_rendered_centerline_conflict']:
   assert len(x['samples'])==3 and x['stable_center_depth'] and all(.1<z['min_m'] and z['max_m']<x['segment_length_m']-.15 for z in x['samples'])
def diagnostic_files(cache,rid,seen=None):
 seen=set() if seen is None else seen
 if str(cache) in seen:return []
 seen.add(str(cache));files=list((cache/'diagnostic_views').glob(rid+'_segment*'))
 if files:return files
 record=cache/'reused_routes.json'
 if record.exists():
  for entry in json.loads(record.read_text()):
   if entry['route_id']==rid:return diagnostic_files(Path(entry['source']).parent.parent,rid,seen)
 return []
results=[];reused=[];success=False
try:
 cache_roots=[Path(p) for p in plan.get('reuse_roots_by_scene',{}).get(str(a.scene),[])]
 for row in rows:
  for cache in cache_roots:
   src=cache/'routes'/(row['route_id']+'.json')
   if src.exists():
    oldproto=json.loads((cache/'protocol.json').read_text());assert oldproto['probe_source_sha256']==plan['probe_sha256'] and oldproto['cohort_inventory_sha256']==plan['route_inventory_sha256'] and oldproto['source_sha256'] in plan['allowed_reuse_source_sha256']
    rr=json.loads(src.read_text());verify_record(rr,row);target=E/'routes'/src.name;shutil.copy2(src,target);assert sha(src)==sha(target)
    image_proof=[]
    for image in diagnostic_files(cache,row['route_id']):
     dst=E/'diagnostic_views'/image.name;shutil.copy2(image,dst);assert sha(dst)==sha(image);image_proof.append({'source':str(image),'sha256':sha(image)})
    if rr['confirmed_optical_conflict_segments'] or rr['unresolved_segments']:assert image_proof,'Do not lose diagnostic images when resuming completed routes'
    results.append(rr);reused.append({'route_id':row['route_id'],'source':str(src),'sha256':sha(src),'source_protocol_sha256':sha(cache/'protocol.json'),'copied_diagnostic_files':image_proof});break
 save('reused_routes.json',reused);existing={x['route_id'] for x in results}
 def progress(rid):
  d={'pid':os.getpid(),'completed_routes':len(results),'expected_routes':len(rows),'reused_routes':len(reused),'tested_segments':sum(x['tested_segments'] for x in results),'routes_with_confirmed_optical_conflict':sum(x['confirmed_optical_conflict_segments']>0 for x in results),'routes_with_unresolved_segments':sum(x['unresolved_segments']>0 for x in results),'last_route':rid,'time':time.time()};save('progress.json',d);print('OPTICAL_ROUTE',json.dumps(d),flush=True)
 progress(None)
 with m.Renderer(a.scene,0,plan['ports_by_slot'][str(a.slot)],'parallel_optical_screen') as renderer:
  probe=OpticalProbe(renderer.client)
  for row in rows:
   if row['route_id'] in existing:continue
   rid=row['route_id'];ep=by[str(row['episodes'][0])];poses=m.integrate(ep,ep['actions']);ref=np.asarray(ep['reference_path'],float);error=float(np.max(np.abs(poses[:,:3]-ref[:,:3])));assert error<1e-7
   assert len(poses)==row['frame_count'] and ep['actions'][-1]==0
   for eid in row['episodes']:assert by[str(eid)]['actions']==ep['actions'] and by[str(eid)]['reference_path']==ep['reference_path']
   rr={'route_id':rid,'episodes':row['episodes'],'frames':len(poses),'GT_action_vs_reference_max_error_m':error,'stationary_transitions':0,'segments':[]}
   for i,(p,q) in enumerate(zip(poses[:-1],poses[1:])):
    length=float(np.linalg.norm(q[:3]-p[:3]))
    if length<1e-4:rr['stationary_transitions']+=1;continue
    assert min(abs(length-2),abs(length-5))<.01
    item=probe.probe(p,q,E/'diagnostic_views'/f'{rid}_segment{i:03d}.png');item.update(from_frame=i,to_frame=i+1,body_start_xyz=p[:3].tolist(),body_end_xyz=q[:3].tolist());rr['segments'].append(item)
   rr['tested_segments']=len(rr['segments']);rr['confirmed_optical_conflict_segments']=sum(x['repeat_confirmed_rendered_centerline_conflict'] for x in rr['segments']);rr['unresolved_segments']=sum(x['unresolved'] for x in rr['segments']);verify_record(rr,row);save('routes/'+rid+'.json',rr);results.append(rr);progress(rid)
 success=True
except BaseException:
 save('failure.json',{'traceback':traceback.format_exc(),'completed_routes':len(results),'time':time.time()});raise
finally:
 save('exit.json',{'success':success,'completed_routes':len(results),'expected_routes':len(rows),'reused_routes':len(reused),'completed_instructions':sum(len(x['episodes']) for x in results),'tested_segments':sum(x['tested_segments'] for x in results),'routes_with_confirmed_optical_conflict':sum(x['confirmed_optical_conflict_segments']>0 for x in results),'confirmed_optical_conflict_segments':sum(x['confirmed_optical_conflict_segments'] for x in results),'routes_with_unresolved_segments':sum(x['unresolved_segments']>0 for x in results),'unresolved_segments':sum(x['unresolved_segments'] for x in results),'body_collision_certificate':False,'source_training_data_modified':False,'official_eval_modified':False,'finished_at':time.time()})
