"""Replay official GT actions under the recovered-action native replay protocol."""
from pathlib import Path
import argparse, collections, copy, datetime, hashlib, importlib.util, io, json
import math, os, pwd, shutil, signal, socket, subprocess, sys, time, fcntl

R=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
S=R/'source/source-3cb14de4fedd047b'
Q=R/'eval/gt_action_replay_20261019'
REC=R/'eval/discrete_action_replay_20261018'
BASE=Path('/mnt/data/users/kindel/uav/experiments/aerialvln-s/20261017-220532-vipe-full-s-route-recovery')
SOURCE=R/'inventory/action_recovery_protocol_preview_v1.json'

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def official_imports():
    sys.path.insert(0,str(S));sys.argv=[sys.argv[0],'--project_prefix',str(Q/'official_logs')]
    global np, airsim, Image, Rotation, SimState, AirVLNENV, action_pose, parse_action
    import numpy as np
    import airsim
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from AirVLN_utils.env_utils import SimState,getPoseAfterMakeAction
    from AirVLN_src.vlnce_src.env import AirVLNENV
    action_pose=getPoseAfterMakeAction
    # Load only the unmodified parser AST; avoid importing the policy/model stack.
    import ast,re
    tree=ast.parse((S/'eval.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='convert_text_to_action')
    namespace={'re':re,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'official_eval_action_parser','exec'),namespace)
    parse_action=namespace['convert_text_to_action']

def pose_values(p):
    return [p.position.x_val,p.position.y_val,p.position.z_val,p.orientation.x_val,p.orientation.y_val,p.orientation.z_val,p.orientation.w_val]

def to_pose(v): return airsim.Pose(airsim.Vector3r(*map(float,v[:3])),airsim.Quaternionr(*map(float,v[3:])))

def start_pose(ep):
    w,x,y,z=ep['start_rotation'];return airsim.Pose(airsim.Vector3r(*ep['start_position']),airsim.Quaternionr(x,y,z,w))

def integrate(ep,actions):
    p=start_pose(ep);poses=[pose_values(p)]
    for action in actions[:-1]:
        p=action_pose(p,int(action));poses.append(pose_values(p))
    return np.asarray(poses)

def score(ep,poses,incremental_ndtw=False,collision_ended=False):
    env=AirVLNENV.__new__(AirVLNENV);state=SimState(index=0,step=0,episode_info=ep,pose=to_pose(poses[0]));env.sim_states=[state]
    unique=[]
    for i,v in enumerate(poses):
        state.pose=to_pose(v);state.step=i;state.is_end=(i==len(poses)-1 and not collision_ended)
        env._update_DistanceToGoal();env._updata_Success();env._update_OracleSuccess();env._update_PathLength();env._update_StepsTaken()
        if incremental_ndtw:env._updata_NDTW()
        if not unique or v[:3].tolist()!=unique[-1].tolist():unique.append(v[:3])
    # Released get_obs marks a depth collision after makeActions updated Success.
    # Preserve that ordering: collision termination does not refresh Success.
    if collision_ended:state.is_end=True
    if not incremental_ndtw:
        state.NDTW['locations']=unique[:-1];env._updata_NDTW()
    env._updata_SDTW()
    return {'episode_id':str(ep['episode_id']),'distance_to_goal':float(state.DistanceToGoal['_metric']),
            'success':float(state.Success['_metric']),'oracle_success':float(state.OracleSuccess['_metric']),
            'ndtw':float(state.NDTW['_metric']),'sdtw':float(state.SDTW['_metric']),
            'path_length':float(state.PathLength['_metric']),'steps_taken':int(state.StepsTaken['_metric'])}

def initialize():
    assert not (Q/'protocol.json').exists(),'Preserve prior run.'
    rows=json.loads(SOURCE.read_text())['records']
    train=json.loads((R/'inventory/official_train.json').read_text())['episodes']
    by_id={str(e['episode_id']):e for e in train}
    assert len(rows)==3834 and len({x['route_id'] for x in rows})==3834
    ids=[eid for row in rows for eid in row['train_episode_ids']]
    assert len(ids)==10097 and len(set(ids))==10097
    assert sha(R/'inventory/official_train.json')=='64a06ef2228579e6e19f688d1eaa3fd3a407d8913fad38ce74e133e0b9336dee'
    groups=collections.defaultdict(list)
    for row in rows:
        ep=by_id[row['train_episode_ids'][0]]
        assert len(ep['actions'])==row['frame_count'] and ep['actions'][-1]==0
        for eid in row['train_episode_ids']:
            target=by_id[eid]
            assert target['actions']==ep['actions']
            assert target['start_position']==ep['start_position'] and target['start_rotation']==ep['start_rotation']
        groups[row['scene_id']].append(row['route_id'])
    old=json.loads((REC/'protocol.json').read_text())
    extension=json.loads((REC/'protocol_termination_extension_v2.json').read_text())
    pilot=old['pilot_route_id']
    protocol={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'routes':3834,'train_instructions':10097,
      'physical_target_frames':sum(x['frame_count'] for x in rows),'excluded_routes':6,
      'action_input_sha256':sha(SOURCE),'official_train_sha256':sha(R/'inventory/official_train.json'),
      'action_source':'Official raw GT primitive actions, independently checked against merged GT training-label parser expansion.',
      'initialization':'Official initial pose; subsequent poses only integrated from official actions; no GT reference-position snapping.',
      'primary_metric':old['primary_metric'],'termination':extension['primary_metric'],
      'RGB':old['RGB'],'playback_fps':5,'playback_fps_is_capture_rate':False,'pose_settle_seconds':.08,
      'pilot_route_id':pilot,'pilot_selection':'Same preselected route as the recovered replay, prior to inspecting GT-native success.',
      'policy_model_evaluated':False,'source':str(SOURCE),'recovered_comparator':str(REC),
      'parallel_gate':'Single-route parser expansion, native pose/camera alignment, RGB hashes, and incremental/final metric equivalence must pass; navigation success is not a gate.'}
    write(Q/'protocol.json',protocol)
    loads={scene:sum(row['frame_count'] for row in rows if row['scene_id']==scene) for scene in groups}
    scenes=sorted(groups,key=lambda scene:-loads[scene])
    write(Q/'queue-v4/plan.json',{'routes_by_scene':{str(k):sorted(v) for k,v in groups.items()},
         'initial_scene_by_gpu':{str(i):scenes[i%len(scenes)] for i in range(5)}})
    print('GT_NATIVE_PLAN_READY',len(rows),len(ids),pilot,flush=True)

class Renderer:
    def __init__(self,scene,gpu,port,tag):
        self.scene,self.gpu,self.port=scene,gpu,port
        self.work=Path('/tmp')/f'aeroact_discrete_{hashlib.sha256(str(Q).encode()).hexdigest()[:8]}_g{gpu}_s{scene}'
        self.debug=Q/'runtime'/tag/f'scene_{scene}';self.process=None
    def __enter__(self):
        self.debug.mkdir(parents=True,exist_ok=True)
        src=Path('/mnt/data/users/kindel/uav/envs/aerialvln')/f'env_{self.scene}'
        owner={'run':str(Q),'scene':self.scene,'gpu':self.gpu}
        if not self.work.exists():
            self.work.mkdir();write(self.work/'owner.json',owner)
            receipt=json.loads((src/'download_receipt.json').read_text())
            for item in receipt['source']['files']:
                rel=Path(*Path(item['name']).parts[2:]);target=self.work/rel;target.parent.mkdir(parents=True,exist_ok=True)
                assert (src/rel).stat().st_size==item['bytes']
                if target.suffix!='.pak':shutil.copy2(src/rel,target)
        else:assert json.loads((self.work/'owner.json').read_text())==owner
        # uavrun cannot traverse the CPFS alias on ali-dojo-eval. Stage PAKs
        # privately as root, then assign the private copy to the runtime user.
        # Never chmod/chown the CPFS source, nor follow the old staging symlink.
        self.deployed_paks=[]
        self.pak_work=Path('/dev/shm')/f'aeroact_gt_replay_paks_{hashlib.sha256(str(Q).encode()).hexdigest()[:8]}_g{self.gpu}_s{self.scene}'
        self.pak_work.mkdir(exist_ok=True)
        marker=self.pak_work/'owner.json'
        if marker.exists():assert json.loads(marker.read_text())==owner
        else:write(marker,owner)
        receipt=json.loads((src/'download_receipt.json').read_text())
        pak_proofs=[]
        for item in receipt['source']['files']:
            rel=Path(*Path(item['name']).parts[2:]);target=self.work/rel
            if target.suffix!='.pak':continue
            cached=self.pak_work/rel;cached.parent.mkdir(parents=True,exist_ok=True)
            if not cached.exists():
                assert shutil.disk_usage(self.pak_work).free>item['bytes']+2_000_000_000
                available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
                assert available>item['bytes']+32_000_000_000
                temporary=cached.with_suffix('.pak.staging')
                shutil.copy2(src/rel,temporary)
                assert temporary.stat().st_size==item['bytes']
                h=hashlib.sha256()
                with temporary.open('rb') as f:
                    for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
                original=hashlib.sha256()
                with (src/rel).open('rb') as f:
                    for block in iter(lambda:f.read(16*1024**2),b''):original.update(block)
                assert h.hexdigest()==original.hexdigest()
                temporary.chmod(0o444);temporary.replace(cached)
                pak_proofs.append({'file':str(rel),'sha256':h.hexdigest(),'byte_equal_source':True})
            assert not cached.is_symlink() and cached.stat().st_size==item['bytes']
            if target.is_symlink():
                assert target.readlink()==cached
                target.unlink()
            assert not target.exists(), 'Inspect unexpected existing private PAK before relinking'
            target.symlink_to(cached)
            self.deployed_paks.append(cached)
        write(self.debug/'private_pak_deployment.json',{'private_staging':str(self.pak_work),'owner':owner,'newly_copied_paks':pak_proofs,'source_assets_modified':False})
        home=self.work/'home';(home/'Documents/AirSim').mkdir(parents=True,exist_ok=True);(home/'xdg').mkdir(exist_ok=True)
        spec=importlib.util.spec_from_file_location('official_server_template',S/'AirVLN/airsim_plugin/AirVLNSimulatorServerTool.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        settings=mod.create_drones();settings['ApiServerPort']=self.port;settings['LocalHostIp']='127.0.0.1'
        write(home/'settings.json',settings);write(home/'Documents/AirSim/settings.json',settings)
        write(home/'nvidia_icd.json',{'file_format_version':'1.0.0','ICD':{'library_path':'libGLX_nvidia.so.0','api_version':'1.3.242'}})
        write(home/'nvidia_egl.json',{'file_format_version':'1.0.0','ICD':{'library_path':'libEGL_nvidia.so.0'}})
        user=pwd.getpwnam('uavrun')
        for directory,_,files in os.walk(self.work):
            os.chown(directory,user.pw_uid,user.pw_gid);os.chmod(directory,0o755)
            for name in files:
                f=Path(directory)/name
                if not f.is_symlink():os.chown(f,user.pw_uid,user.pw_gid)
        (home/'xdg').chmod(0o700);binary=self.work/'LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping';binary.chmod(0o755)
        with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',self.port))
        cmd=['runuser','-u','uavrun','--','env',f'HOME={home}',f'XDG_RUNTIME_DIR={home}/xdg',f'VK_ICD_FILENAMES={home}/nvidia_icd.json',f'__EGL_VENDOR_LIBRARY_FILENAMES={home}/nvidia_egl.json',str(binary),'AirVLN','-RenderOffscreen','-NoSound','-NoVSync',f'-GraphicsAdapter={self.gpu}',f'-UserDir={home}','-unattended','-NoSplash','--settings',str(home/'settings.json')]
        self.log=(self.debug/f'native_{int(time.time())}.log').open('w');self.process=subprocess.Popen(cmd,cwd=self.work/'LinuxNoEditor',stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        write(self.debug/'process.json',{'pid':self.process.pid,'command':cmd,'port':self.port})
        try:
            for _ in range(90):
                assert self.process.poll() is None,'Native scene exited'
                try:
                    with socket.create_connection(('127.0.0.1',self.port),timeout=.5):pass
                    break
                except OSError:time.sleep(1)
            else:raise RuntimeError('Native RPC listener timeout')
            self.client=airsim.VehicleClient(ip='127.0.0.1',port=self.port,timeout_value=60);self.client.confirmConnection();return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*_):
        if self.process is not None:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=10)
        if hasattr(self,'log'):self.log.close()
        if hasattr(self,'deployed_paks'):
            assert json.loads((self.work/'owner.json').read_text())=={'run':str(Q),'scene':self.scene,'gpu':self.gpu}
            for target in self.deployed_paks:
                assert target.is_relative_to(self.pak_work) and not target.is_symlink()
                target.unlink(missing_ok=True)
    def set_pose(self,v):self.client.simSetVehiclePose(to_pose(v),True,vehicle_name='Drone_1')
    def capture(self,desired):
        for attempt in range(8):
            responses=self.client.simGetImages([airsim.ImageRequest('front_0',airsim.ImageType.Scene,False,True),airsim.ImageRequest('front_0',airsim.ImageType.DepthPerspective,True,False)],vehicle_name='Drone_1')
            response,depth=responses
            try:
                assert (response.width,response.height)==(224,224)
                rgb=np.asarray(Image.open(io.BytesIO(response.image_data_uint8)).convert('RGB')).copy()
                actual=np.asarray(pose_values(self.client.simGetVehiclePose(vehicle_name='Drone_1')))
                poserr=float(np.linalg.norm(actual[:3]-desired[:3]));rot=Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(desired[3:]);roterr=float(np.degrees(rot.magnitude()))
                camera_pos=np.asarray([response.camera_position.x_val,response.camera_position.y_val,response.camera_position.z_val]);expected_camera=desired[:3]+Rotation.from_quat(desired[3:]).apply([.5,0,0]);cameraerr=float(np.linalg.norm(camera_pos-expected_camera))
                assert rgb.shape==(224,224,3) and poserr<.01 and roterr<.01 and cameraerr<.02,(poserr,roterr,cameraerr)
                depth_m=np.asarray(depth.image_data_float,dtype=np.float32).reshape(depth.height,depth.width)
                assert depth_m.shape==(256,256)
                return rgb,actual,{'position_error_m':poserr,'rotation_error_deg':roterr,'image_camera_position_error_m':cameraerr,'api_timestamp_ns':int(response.time_stamp),'collision_fraction':float((np.clip(depth_m,0,100)/100<.004).mean()),'capture_attempts':attempt+1}
            except (AssertionError,ValueError,OSError):
                if attempt==7:raise
                time.sleep(.1)

def check_video(folder,frames):
    import cv2
    cv2.setNumThreads(1);cap=cv2.VideoCapture(str(folder/'rgb_lossless.mp4'));n=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert n<len(frames) and bgr.shape==(224,224,3)
            assert hashlib.sha256(bgr[:,:,::-1].tobytes()).hexdigest()==frames[n]['sha256_rgb'],n
            n+=1
    finally:cap.release()
    assert n==len(frames)
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=pix_fmt,nb_read_frames','-of','json',str(folder/'display.mp4')],capture_output=True,text=True,check=True).stdout)['streams'][0]
    assert probe['pix_fmt']=='yuv420p' and int(probe['nb_read_frames'])==n

def replay(row,renderer,by_id,annotations,pilot=False):
    rid=row['route_id'];parent=Q/('pilot' if pilot else 'routes')/rid
    attempts=[int(p.name.split('-')[1]) for p in parent.glob('attempt-*') if p.is_dir()]
    folder=parent/f'attempt-{max(attempts,default=0)+1:03d}';folder.mkdir(parents=True,exist_ok=False)
    ep=by_id[row['train_episode_ids'][0]];actions=ep['actions'];expanded=[]
    for x in annotations[ep['episode_id']]:expanded+=parse_action(x['action'])
    assert expanded==actions,'Merged training labels must reproduce the exact primitive sequence.'
    desired=integrate(ep,actions);gt_control=integrate(ep,ep['actions']);reference=np.asarray(ep['reference_path'])
    gt_error=float(np.max(np.abs(gt_control[:,:3]-reference[:,:3])));assert gt_error<1e-7
    mismatch=[i for i,(a,b) in enumerate(zip(actions,ep['actions'])) if a!=b]
    selected={0,len(actions)//2,len(actions)-1}
    if mismatch:selected.update(i for i in [mismatch[0]-1,mismatch[0],mismatch[0]+1] if 0<=i<len(actions))
    renderer.set_pose(desired[0])
    for _ in range(8):time.sleep(.04);renderer.capture(desired[0])
    encoder_log=(folder/'encoder.log').open('w');command=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','224x224','-r','5','-i','-','-map','0:v','-c:v','libx264rgb','-crf','0','-preset','ultrafast','-threads','1','-movflags','+faststart',str(folder/'rgb_lossless.mp4'),'-map','0:v','-vf','format=yuv420p','-c:v','libx264','-crf','20','-preset','ultrafast','-threads','1','-movflags','+faststart',str(folder/'display.mp4')]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=encoder_log);frames=[];actual=[]
    termination_reason='terminal_STOP';same_count=0;turn_count=0;last_action=None
    try:
        for i,v in enumerate(desired):
            renderer.set_pose(v);time.sleep(.08);rgb,body,diag=renderer.capture(v);encoder.stdin.write(rgb.tobytes());actual.append(body)
            frame={'index':i,'action_after_observation':int(actions[i]),'desired_body_pose':v.tolist(),'actual_body_pose':body.tolist(),'sha256_rgb':hashlib.sha256(rgb.tobytes()).hexdigest(),'mean_rgb':rgb.mean((0,1)).tolist(),'std_rgb':float(rgb.std()),**diag};frames.append(frame)
            if i>0:
                previous=actions[i-1];same_count=same_count+1 if previous==last_action else 1;last_action=previous;turn_count=turn_count+1 if previous in [2,3] else 0
            reason=None
            if diag['collision_fraction']>.1:reason='official_depth_collision_threshold'
            elif same_count>=40:reason='40_identical_consecutive_actions'
            elif turn_count>=30:reason='30_consecutive_turns'
            elif i>=500:reason='500_action_limit'
            elif actions[i]==0:reason='terminal_STOP'
            frame['planned_gt_action']=int(actions[i]);frame['action_after_observation']=0 if reason else int(actions[i])
            frame['termination_reason']=reason
            if i in selected or reason:Image.fromarray(rgb).save(folder/f'frame_{i:04d}.png')
            if pilot and i%30==0:print('PILOT_FRAME',rid,i+1,len(desired),flush=True)
            if reason:
                termination_reason=reason;break
        encoder.stdin.close();assert encoder.wait(timeout=90)==0
    finally:
        if encoder.poll() is None:encoder.kill();encoder.wait()
        encoder_log.close()
    check_video(folder,frames);actual=np.asarray(actual)
    manifest={'route_id':rid,'scene_id':row['scene_id'],'frame_count':len(frames),'planned_frame_count':len(desired),'termination_reason':termination_reason,'resolution':[224,224],'frames':frames,'rgb_lossless_sha256':sha(folder/'rgb_lossless.mp4'),'display_sha256':sha(folder/'display.mp4'),'all_lossless_decoded_frame_hashes_verified':True,'playback_fps':5,'playback_fps_is_capture_rate':False,'script_sha256':sha(Path(__file__))};write(folder/'manifest.json',manifest)
    metrics={}
    for eid in row['train_episode_ids']:
        target=by_id[eid];metrics[eid]={'GT_native_actual':score(target,actual,collision_ended=termination_reason=='official_depth_collision_threshold'),'GT_full_sequence_kinematic_only':score(target,desired)}
    equivalence=True
    if pilot:
        slow=score(ep,actual,True,collision_ended=termination_reason=='official_depth_collision_threshold');fast=metrics[str(ep['episode_id'])]['GT_native_actual'];equivalence=all(abs(slow[k]-fast[k])<1e-10 for k in ['ndtw','sdtw','success','oracle_success','distance_to_goal']);assert equivalence
    result={'route_id':rid,'scene_id':row['scene_id'],'status':'completed','frame_count':len(actual),'instruction_count':len(row['train_episode_ids']),'metrics_by_episode':metrics,'first_action_disagreement_index':mismatch[0] if mismatch else None,'action_disagreement_count':len(mismatch),'GT_action_integration_max_position_error_m':gt_error,'actual_body_max_position_error_m':max(x['position_error_m'] for x in frames),'actual_body_max_rotation_error_deg':max(x['rotation_error_deg'] for x in frames),'RGB_camera_max_position_error_m':max(x['image_camera_position_error_m'] for x in frames),'collision_flagged_frames':sum(x['collision_fraction']>.1 for x in frames),'first_collision_frame':next((x['index'] for x in frames if x['collision_fraction']>.1),None),'all_RGB_frames_hash_verified':True,'official_metric_incremental_final_equivalence_verified':equivalence if pilot else 'validated_on_pilot','manifest_sha256':sha(folder/'manifest.json'),'output':str(folder),'note':'Full recorded action replay. Collision diagnostics do not truncate this reconstruction experiment; not a learned-policy navigation score.'}
    result['planned_frame_count']=len(desired);result['termination_frame']=len(actual)-1;result['termination_reason']=termination_reason
    result['note']='Native replay stops at depth collision, 40 same primitives, 30 turns, 500 actions or terminal STOP. RGB covers exactly the executed prefix. Full-sequence endpoint score is a separate kinematic-only diagnostic. No VLM policy is evaluated.'
    write(folder/'result.json',result);write(Q/'results'/f'{rid}.json',result)
    if pilot:
        write(Q/'pilot/gate.json',{'passed':True,'route_id':rid,'pipeline_execution_verified':True,'native_navigation_success':metrics[str(ep['episode_id'])]['GT_native_actual']['success'],'allow_parallel':True,'termination_reason':termination_reason,'source_sha256':sha(Path(__file__)),'checked':['exact training-label parser expansion','official GT action integration','native pose execution','image camera/pose alignment','all RGB lossless hashes and video frame count','official metric incremental/final equivalence','same native collision and termination implementation as recovered replay'],'result':str(folder/'result.json')})
    print('ROUTE_COMPLETE',rid,json.dumps(metrics[str(ep['episode_id'])]['GT_native_actual']),flush=True);return result


def claim_route(route_ids):
    """Hold an OS file lock for the full replay; completed results are immutable."""
    locks=Q/'queue-v4/locks';locks.mkdir(parents=True,exist_ok=True)
    for rid in route_ids:
        if (Q/'results'/f'{rid}.json').exists():continue
        lock=(locks/f'{rid}.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if (Q/'results'/f'{rid}.json').exists():lock.close();continue
        return rid,lock
    return None

def shared_worker(gpu,rows,by_id,annotations):
    plan=json.loads((Q/'queue-v4/plan.json').read_text())
    scenes={int(k):v for k,v in plan['routes_by_scene'].items()}
    initial=plan['initial_scene_by_gpu'][str(gpu)]
    status=Q/'queue-v4'/f'worker_{gpu}.json'
    while True:
        pending={scene:[rid for rid in ids if not (Q/'results'/f'{rid}.json').exists()] for scene,ids in scenes.items()}
        order=sorted(pending,key=lambda scene:sum(rows[r]['frame_count'] for r in pending[scene]),reverse=True)
        if initial is not None:order=[initial]+[s for s in order if s!=initial];initial=None
        chosen=None
        for scene in order:
            claimed=claim_route(pending[scene])
            if claimed is not None:chosen=(scene,claimed);break
        if chosen is None:
            write(status,{'gpu':gpu,'phase':'done_or_all_remaining_routes_claimed','time':time.time()})
            return
        scene,current=chosen
        try:
            write(status,{'gpu':gpu,'phase':'starting_native_scene','scene':scene,'route_id':current[0],'time':time.time()})
            with Renderer(scene,gpu,52001+gpu*1000,f'queue_v4_worker_{gpu}') as renderer:
                while current is not None:
                    rid,lock=current
                    write(status,{'gpu':gpu,'phase':'replaying','scene':scene,'route_id':rid,'time':time.time()})
                    try:replay(rows[rid],renderer,by_id,annotations)
                    finally:lock.close();current=None
                    current=claim_route(scenes[scene])
        finally:
            if current is not None:current[1].close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--init',action='store_true');parser.add_argument('--pilot',action='store_true');parser.add_argument('--worker',type=int);a=parser.parse_args()
    if a.init:initialize();return
    gpu=0 if a.pilot else a.worker
    assert gpu in range(5)
    gpu_lock=Path(f'/tmp/aeroact_official_policy_gpu_{gpu}.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    official_imports();protocol=json.loads((Q/'protocol.json').read_text());rows={x['route_id']:x for x in json.loads(SOURCE.read_text())['records']};assert sha(SOURCE)==protocol['action_input_sha256']
    train=json.loads((R/'inventory/official_train.json').read_text())['episodes'];by_id={str(e['episode_id']):e for e in train};annotations=collections.defaultdict(list)
    for item in json.loads((R/'dataset/official-s-full-64a06ef2-aeroact0417-exclude6-v1/data/aerialvln-s/train_merged_triple.json').read_text()):annotations[item['episode']].append(item)
    if a.pilot:
        row=rows[protocol['pilot_route_id']]
        with Renderer(row['scene_id'],0,51601,'pilot') as renderer:replay(row,renderer,by_id,annotations,True)
    else:
        assert a.worker in range(5) and json.loads((Q/'pilot/gate.json').read_text())['passed']
        shared_worker(a.worker,rows,by_id,annotations)

if __name__=='__main__':main()
