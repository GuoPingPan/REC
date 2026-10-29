"""Pose-to-action calibration without reference trajectories or GT actions.

Assumptions: known camera/body extrinsic and primitive magnitudes; exactly one
primitive transition per adjacent observed frame. These are simulator-specific
priors, not a solution for unclocked arbitrary real-world video.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

OPTICAL_TO_FRD=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
TRANSLATIONS=np.array([[5,0,0],[0,0,0],[0,0,0],[0,0,-2],[0,0,2],[0,-5,0],[0,5,0]],float)
ROTATIONS=Rotation.from_euler('z',np.array([0,-15,15,0,0,0,0]),degrees=True).as_matrix()
UNITS=np.array([5.,5.,2.])
PROTOCOL={'version':'primitive-calibration-v1','scale_grid_log_bounds':[-3.,3.],
          'scale_grid_points':121,'robust_objective':'mean(sqrt(1+minimum_joint_primitive_cost)-1)',
          'camera_forward_offset_m':.5,'angular_unit_degrees':15,
          'reads_reference_poses':False,'reads_GT_actions':False,
          'known_one_primitive_per_frame_interval':True,'known_primitive_magnitudes':True}

def motion_features(camera_poses):
 p=np.asarray(camera_poses,float)
 assert p.ndim==3 and p.shape[1:]==(4,4) and len(p)>=2 and np.isfinite(p).all()
 br=p[:,:3,:3]@OPTICAL_TO_FRD.T
 rel=np.einsum('nji,njk->nik',br[:-1],br[1:])
 cam_local=np.einsum('nji,nj->ni',br[:-1],np.diff(p[:,:3,3],axis=0))
 lever=.5*(rel[:,:,0]-np.array([1.,0.,0.]))
 trace=np.einsum('aij,nij->na',ROTATIONS,rel)
 angular=np.arccos(np.clip((trace-1)/2,-1,1))/(np.pi/12)
 return cam_local,lever,angular**2

def costs_at_scale(features,scale):
 cam,lever,angular=features
 delta=scale*cam-lever
 return np.sum(((delta[:,None]-TRANSLATIONS[None])/UNITS)**2,axis=2)+angular

def predict(camera_poses,calibrate=True):
 f=motion_features(camera_poses)
 def objective(logscale):
  best=np.min(costs_at_scale(f,np.exp(logscale)),axis=1)
  return float(np.mean(np.sqrt(1+best)-1))
 if calibrate:
  grid=np.linspace(*PROTOCOL['scale_grid_log_bounds'],PROTOCOL['scale_grid_points'])
  values=np.array([objective(x) for x in grid]);i=int(np.argmin(values))
  lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
  fit=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-7})
  scale=float(np.exp(fit.x));boundary=i in [0,len(grid)-1]
 else:scale=1.;boundary=False
 c=costs_at_scale(f,scale);order=np.sort(c,axis=1);best=order[:,0]
 confidence=np.exp(-np.minimum(best,50))*(order[:,1]-best)/(1+order[:,1]-best)
 return {'actions':np.argmin(c,axis=1).astype(int)+1,'scale':scale,
         'objective':float(np.mean(np.sqrt(1+best)-1)), 'scale_at_search_boundary':boundary,
         'confidence':confidence,'minimum_cost':best}

def verify_geometry():
 # Independent synthetic generator: all seven primitives, three gauges/scales.
 # A known 0.5m camera lever arm must not be scaled with video coordinates.
 seq=np.array([1,2,3,4,5,6,7,1,1,3,6,4,2,7,5]*3)
 body_p=[np.zeros(3)];body_r=[np.eye(3)]
 for act in seq:
  body_p.append(body_p[-1]+body_r[-1]@TRANSLATIONS[act-1])
  body_r.append(body_r[-1]@ROTATIONS[act-1])
 body_p=np.asarray(body_p);body_r=np.asarray(body_r)
 cam=np.repeat(np.eye(4)[None],len(body_p),axis=0)
 cam[:,:3,:3]=body_r@OPTICAL_TO_FRD;cam[:,:3,3]=body_p+.5*body_r[:,:,0]
 results=[]
 for factor in [.23,1.,4.2]:
  gauge=Rotation.from_euler('xyz',[.31,-.54,.82]).as_matrix();pred=cam.copy()
  pred[:,:3,:3]=gauge[None]@cam[:,:3,:3]
  pred[:,:3,3]=(cam[:,:3,3]@gauge.T)/factor+np.array([3.,-8.,12.])
  x=predict(pred);assert np.array_equal(x['actions'],seq)
  assert abs(np.log(x['scale']/factor))<1e-4
  results.append({'true_scale':factor,'estimated_scale':x['scale'],'all_primitives_correct':True})
 return results

if __name__=='__main__':print(verify_geometry())
