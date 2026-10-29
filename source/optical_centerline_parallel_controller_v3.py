"""Two private renderers share the idle existing ali-eval GPU under one owner lock.

Identical frozen optical measurement, disjoint scenes, resume complete old rows.
No paid nodes, training submissions or policy-evaluation modifications.
"""
from pathlib import Path
import json,time,os,sys,subprocess,fcntl,hashlib,traceback,uuid,signal,socket
R=Path(sys.argv[1]).resolve();E=R/'eval/optical_centerline_screen_all_selected_003';E.mkdir(exist_ok=False)
lock=(E/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu_lock=Path('/tmp/aeroact_official_policy_gpu_0.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines();assert len(used)==1 and int(used[0])<512,'Initial GPU must actually be idle; do not mix with another workload'
assert len(os.sched_getaffinity(0))>=4
for port in [55611,55612]:
 with socket.socket() as sock:
  sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
def save(n,d):
 p=E/n;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory=R/'dataset/complete-proxy-pass-metric-v2/route_inventory.json';rows=json.loads(inventory.read_text());ann=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official/inventory/official_train.json');by={str(x['episode_id']):x for x in json.loads(ann.read_text())['episodes']};scenes=sorted({int(x['scene_id']) for x in rows if int(x['scene_id'])!=17},key=lambda s:-sum(sum(y in [1,4,5,6,7] for y in by[str(x['episodes'][0])]['actions']) for x in rows if int(x['scene_id'])==s));expected={str(s):{'routes':sum(int(x['scene_id'])==s for x in rows),'instructions':sum(len(x['episodes']) for x in rows if int(x['scene_id'])==s)} for s in [17]+scenes}
worker=R/'source/optical_centerline_parallel_worker_v3.py';probe=R/'source/optical_centerline_probe_v1.py';initial=R/'eval/optical_centerline_screen_scene17_001';first=json.loads((initial/'exit.json').read_text());firstproto=json.loads((initial/'protocol.json').read_text());assert first['success'] and first['completed_routes']==62 and first['completed_instructions']==175 and json.loads((initial/'pilot_gate.json').read_text())['passed'] and firstproto['probe_source_sha256']==sha(probe)
old11=R/'eval/optical_centerline_screen_scene11_001';assert (old11/'exit.json').exists() and not json.loads((old11/'exit.json').read_text())['success'],'Preserve old attempt with explicit interruption receipt before resumption'
plan={'coordinator_pid':os.getpid(),'host':subprocess.check_output(['hostname'],text=True).strip(),'nonce':uuid.uuid4().hex,'gpu':0,'simulator_slots':2,'ports_by_slot':{'0':55611,'1':55612},'initial_GPU_memory_MiB':int(used[0]),'GPU_lock_ownership':'Coordinator holds existing exclusive GPU0lock; only its two child slots share this GPU.','probe_sha256':sha(probe),'worker_source_sha256':sha(worker),'allowed_worker_source_sha256':[firstproto['source_sha256'],sha(worker)],'controller_sha256':sha(Path(__file__)),'route_inventory_sha256':sha(inventory),'official_annotations_sha256':sha(ann),'expected_routes':len(rows),'expected_instructions':sum(len(x['episodes']) for x in rows),'expected_translation_segments':sum(sum(y in [1,4,5,6,7] for y in by[str(x['episodes'][0])]['actions']) for x in rows),'by_scene':expected,'scene_order':scenes,'reuse_roots_by_scene':{'11':[str(old11)]},'initial_scene17_attempt':str(initial),'original_training_RGB_or_labels_modified':False,'official_eval_modified':False,'paid_submissions':0,'started_at':time.time()};save('plan.json',plan)
completed={'17':{'attempt':str(initial),'exit':first}};save('completed_scenes.json',completed);running={};success=False
try:
 while scenes or running:
  for slot in [0,1]:
   if slot in running or not scenes:continue
   assert sha(worker)==plan['worker_source_sha256'] and sha(probe)==plan['probe_sha256']
   scene=scenes.pop(0);log=E/f'scene_{scene}.log'
   with log.open('x') as f:
    cmd=['/tmp/aeroact_official_runtime_0417_py310_v1/bin/python','-u',str(worker),'--root',str(R),'--plan',str(E/'plan.json'),'--scene',str(scene),'--slot',str(slot)];p=subprocess.Popen(cmd,cwd='/tmp',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   running[slot]={'process':p,'scene':scene,'command':cmd};save(f'workers/scene_{scene}.json',{'pid':p.pid,'scene':scene,'slot':slot,'command':cmd,'time':time.time()})
  state=[]
  for slot,item in list(running.items()):
   p=item['process'];scene=item['scene'];target=E/'scenes'/f'scene_{scene}';entry={'slot':slot,'scene':scene,'pid':p.pid,'returncode':p.poll()}
   progress=target/'progress.json'
   if progress.exists():entry['progress']=json.loads(progress.read_text())
   state.append(entry)
   if p.poll() is not None:
    assert p.returncode==0,(scene,p.returncode,'Preserve partial scene output; no automatic retry')
    receipt=json.loads((target/'exit.json').read_text());assert receipt['success'] and receipt['completed_routes']==expected[str(scene)]['routes'] and receipt['completed_instructions']==expected[str(scene)]['instructions']
    completed[str(scene)]={'attempt':str(target),'exit':receipt};save('completed_scenes.json',completed);del running[slot];print('SCENE_DONE',scene,receipt['completed_routes'],flush=True)
  save('status.json',{'phase':'parallel_optical_screen','workers':state,'remaining_scenes':scenes,'completed_scenes':list(completed),'time':time.time()})
  if running:time.sleep(5)
 assert sum(x['exit']['completed_routes'] for x in completed.values())==plan['expected_routes'] and sum(x['exit']['completed_instructions'] for x in completed.values())==plan['expected_instructions'];success=True
except BaseException:
 save('failure.json',{'traceback':traceback.format_exc(),'live_children':[{'pid':x['process'].pid,'scene':x['scene']} for x in running.values() if x['process'].poll() is None],'time':time.time()})
 for item in running.values():
  p=item['process']
  if p.poll() is None:p.send_signal(signal.SIGINT)
 for item in running.values():
  p=item['process']
  try:p.wait(timeout=60)
  except subprocess.TimeoutExpired:save('child_cleanup_pending_'+str(p.pid)+'.json',{'pid':p.pid,'scene':item['scene'],'requires_inspection':True})
 raise
finally:
 totals={k:sum(x['exit'][k] for x in completed.values()) for k in ['completed_routes','completed_instructions','tested_segments','routes_with_confirmed_optical_conflict','confirmed_optical_conflict_segments','routes_with_unresolved_segments','unresolved_segments']};save('exit.json',{'success':success,'completed_scenes':completed,'totals':totals,'body_collision_certificate':False,'training_hold_cleared':False,'paid_submissions':0,'source_data_modified':False,'finished_at':time.time()})
