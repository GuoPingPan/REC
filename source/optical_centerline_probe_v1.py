"""Body-centered rendered-depth segment screen; no geodetic/ray API dependency.

Positive is an optical geometry proxy, not a swept-body contact certificate.
Negative does not establish collision freedom. Diagnostic camera is private.
"""
import io,time
import numpy as np
import airsim
from scipy.spatial.transform import Rotation
from PIL import Image

class OpticalProbe:
 def __init__(self,client):self.c=client
 def set_view(self,p,q):
  delta=np.asarray(q[:3])-np.asarray(p[:3]);length=float(np.linalg.norm(delta));assert length>.1
  direction=delta/length;body=Rotation.from_quat(p[3:]);local=body.inv().apply(direction)
  relative,_=Rotation.align_vectors(local[None],np.array([[1.,0.,0.]]))
  self.c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*map(float,p[:3])),airsim.Quaternionr(*map(float,p[3:]))),True,vehicle_name='Drone_1')
  self.c.simSetCameraPose('front_0',airsim.Pose(airsim.Vector3r(0,0,0),airsim.Quaternionr(*map(float,relative.as_quat()))),vehicle_name='Drone_1')
  time.sleep(.04)
  cam=self.c.simGetCameraInfo('front_0',vehicle_name='Drone_1');cp=np.array([cam.pose.position.x_val,cam.pose.position.y_val,cam.pose.position.z_val]);cr=Rotation.from_quat([cam.pose.orientation.x_val,cam.pose.orientation.y_val,cam.pose.orientation.z_val,cam.pose.orientation.w_val])
  poserr=float(np.linalg.norm(cp-np.asarray(p[:3])));dot=float(np.dot(cr.apply([1,0,0]),direction));assert poserr<.001 and dot>.99999,(poserr,dot)
  return length,poserr,dot
 def depth(self):
  response=self.c.simGetImages([airsim.ImageRequest('front_0',airsim.ImageType.DepthPerspective,True,False)],vehicle_name='Drone_1')[0]
  assert (response.height,response.width)==(256,256)
  d=np.asarray(response.image_data_float,np.float32).reshape(256,256);central=d[127:129,127:129]
  assert np.isfinite(central).all() and float(central.min())>=0
  return d,central.copy(),int(response.time_stamp)
 def probe(self,p,q,save_image=None):
  length,poserr,dot=self.set_view(p,q);d,center,stamp=self.depth()
  samples=[{'timestamp_ns':stamp,'central_2x2_m':center.tolist(),'min_m':float(center.min()),'max_m':float(center.max())}]
  candidate=float(center.max())<length-.15
  if candidate:
   for repeat in range(2):
    time.sleep(.04);d,cen,ts=self.depth();assert ts>samples[-1]['timestamp_ns'],'Depth sample was not refreshed'
    samples.append({'timestamp_ns':ts,'central_2x2_m':cen.tolist(),'min_m':float(cen.min()),'max_m':float(cen.max())})
  stable=all(float(np.max(np.abs(np.asarray(x['central_2x2_m'])-center)))<=max(.02,.01*length) for x in samples)
  conflict=candidate and stable and all(.1<x['min_m'] and x['max_m']<length-.15 for x in samples)
  near=candidate and any(x['min_m']<=.1 for x in samples)
  result={'segment_length_m':length,'camera_position_error_m':poserr,'camera_forward_dot_GT':dot,'samples':samples,'positive_candidate':candidate,'stable_center_depth':stable,'repeat_confirmed_rendered_centerline_conflict':bool(conflict),'near_plane_or_inside_unresolved':bool(near),'unresolved':bool(candidate and not conflict)}
  if save_image is not None and candidate:
   r=self.c.simGetImages([airsim.ImageRequest('front_0',airsim.ImageType.Scene,False,True)],vehicle_name='Drone_1')[0]
   Image.open(io.BytesIO(r.image_data_uint8)).convert('RGB').save(save_image)
   np.savez_compressed(str(save_image)+'.depth.npz',depth_m=d)
  return result
