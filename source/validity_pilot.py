"""Independent GT validity diagnostic; never rewrites training or evaluation data."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--scene', type=int, required=True)
ap.add_argument('--gpu', type=int, required=True)
ap.add_argument('--route', action='append', required=True)
args = ap.parse_args()
ROOT = Path(args.root).resolve()
U = Path('/mnt/data/users/kindel/uav')
OLD = U/'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official'
N = U/'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery'
legacy = ROOT/'source/gt_native_replay_v2.py'
spec = importlib.util.spec_from_file_location('legacy_renderer', legacy)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.Q = ROOT/'eval/validity_pilot'
mod.official_imports()
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(p)

def collision_info(client):
    try:
        c = client.simGetCollisionInfo(vehicle_name='Drone_1')
        return {'api_available': True, 'has_collided': bool(c.has_collided),
                'object_name': str(c.object_name), 'object_id': int(c.object_id),
                'penetration_depth': float(c.penetration_depth), 'time_stamp': int(c.time_stamp),
                'interpretation': 'ComputerVision + ignore_collision pose stepping; a false/stale flag does not prove absence of swept collision.'}
    except Exception as e:
        return {'api_available': False, 'error': str(e)}

class Renderer(mod.Renderer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.work = Path('/dev/shm')/('aerialvln_validity_'+ROOT.name+'_g'+str(self.gpu)+'_s'+str(self.scene))

def capture(renderer, desired):
    a = mod.airsim
    responses = renderer.client.simGetImages([
        a.ImageRequest('front_0', a.ImageType.Scene, False, True),
        a.ImageRequest('front_0', a.ImageType.DepthPerspective, True, False)], vehicle_name='Drone_1')
    import io
    rgb = np.asarray(Image.open(io.BytesIO(responses[0].image_data_uint8)).convert('RGB')).copy()
    depth = np.asarray(responses[1].image_data_float, dtype=np.float32).reshape(256, 256)
    actual = np.asarray(mod.pose_values(renderer.client.simGetVehiclePose(vehicle_name='Drone_1')))
    pe = float(np.linalg.norm(actual[:3]-desired[:3]))
    re = float(np.degrees((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(desired[3:])).magnitude()))
    assert pe < .01 and re < .01 and rgb.shape == (224,224,3)
    valid = np.isfinite(depth)
    diag = {'desired_body_pose': desired.tolist(), 'actual_body_pose': actual.tolist(),
            'position_error_m': pe, 'rotation_error_deg': re,
            'rgb_time_stamp': int(responses[0].time_stamp), 'depth_time_stamp': int(responses[1].time_stamp),
            'depth_finite_fraction': float(valid.mean()), 'depth_zero_fraction': float((depth==0).mean()),
            'depth_negative_fraction': float((depth<0).mean()),
            'depth_quantiles_m': np.quantile(depth[valid], [0,.01,.05,.1,.5,.9,1]).astype(float).tolist() if valid.any() else [],
            'collision_fraction': float((np.clip(depth,0,100)/100 < .004).mean()),
            'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'contact_info': collision_info(renderer.client)}
    return rgb, depth, diag

def run():
    inv = json.loads((N/'inventory/routes.json').read_text())
    by = {str(e['episode_id']):e for e in json.loads((OLD/'inventory/official_train.json').read_text())['episodes']}
    out = ROOT/'eval/validity_pilot'/('scene_'+str(args.scene))
    assert not (out/'exit.json').exists(), 'Inspect previous pilot before retry.'
    write(out/'process.json', {'pid':os.getpid(), 'scene':args.scene, 'gpu':args.gpu, 'routes':args.route, 'started_at':time.time()})
    results = []
    with Renderer(args.scene, args.gpu, 59001+args.gpu*1000, 'scene_'+str(args.scene)) as renderer:
        for rid in args.route:
            row = inv[rid]; assert row['scene_id'] == args.scene
            ep = by[row['train_episode_ids'][0]]
            poses = mod.integrate(ep, ep['actions'])
            assert len(poses) == len(ep['actions']) and ep['actions'][-1] == 0
            assert np.max(np.abs(poses[:,:3]-np.asarray(ep['reference_path'])[:,:3])) < 1e-7
            D = out/rid; D.mkdir(parents=True, exist_ok=False)
            renderer.set_pose(poses[0]); time.sleep(.2)
            for _ in range(8): capture(renderer, poses[0]); time.sleep(.04)
            frames = []; flagged = []
            for i,v in enumerate(poses):
                renderer.set_pose(v); time.sleep(.08)
                rgb, depth, diag = capture(renderer,v)
                Image.fromarray(rgb).save(D/('frame_%04d.png'%i))
                diag.update(index=i, original_GT_action=int(ep['actions'][i]))
                if diag['collision_fraction'] > .1:
                    flagged.append(i)
                    # Store depths and repeated observations for the first flagged frame.
                    if len(flagged) == 1:
                        np.savez_compressed(D/'first_flag_depth.npz',depth_m=depth)
                        repeats = []
                        for j in range(3):
                            time.sleep(.15); r2,d2,q2=capture(renderer,v)
                            Image.fromarray(r2).save(D/('first_flag_repeat_%d.png'%j))
                            repeats.append(q2)
                        diag['repeated_same_pose_observations'] = repeats
                frames.append(diag)
                write(out/'progress.json', {'route_id':rid,'frame':i+1,'planned':len(poses),'flags':len(flagged),'time':time.time()})
            # This is diagnostics, never a new official SR: collect the tail to inspect validity.
            result = {'route_id':rid,'scene_id':args.scene,'frames':len(frames),
                      'original_recorded_frames':row['frame_count'],'planned_frames':len(poses),
                      'original_termination':row['termination_reason'],'flagged_frame_indices':flagged,
                      'contact_true_frames':[f['index'] for f in frames if f['contact_info'].get('has_collided')],
                      'terminal_original_STOP':int(ep['actions'][-1])==0,
                      'physical_legality':'unproven; inspect depth/RGB and contact API semantics',
                      'all_frames':frames}
            write(D/'diagnostic.json',result);results.append({k:v for k,v in result.items() if k!='all_frames'})
            print(json.dumps(results[-1]),flush=True)
    write(out/'exit.json',{'success':True,'finished_at':time.time(),'routes':results,'official_evaluation_modified':False,'training_data_modified':False})

if __name__ == '__main__':
    try: run()
    except BaseException:
        write(ROOT/'eval/validity_pilot'/('scene_'+str(args.scene))/'failure.json',{'time':time.time(),'traceback':traceback.format_exc()})
        raise
