"""One-video ViPE baseline. Reads RGB and nominal K; never reads pose labels."""
import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from vipe.slam.system import SLAMSystem
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
config = json.loads(args.input.read_text())
video = Path(config['video'])
assert hashlib.sha256(video.read_bytes()).hexdigest() == config['video_sha256']
cap = cv2.VideoCapture(str(video))
images = []
decoded_hashes = []
while True:
    ok, image = cap.read()
    if not ok:
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image.shape == (224, 224, 3)
    images.append(image)
    decoded_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
cap.release()
assert len(images) == config['frame_count']
if config.get('frames') and 'sha256_rgb' in config['frames'][0]:
    assert decoded_hashes == [f['sha256_rgb'] for f in config['frames']]
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
K = np.asarray(config['K'])
intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], device='cuda', dtype=torch.float32)


class RGBVideo(VideoStream):
    def __len__(self):
        return len(images)

    def name(self):
        return 'env18_one_video'

    def fps(self):
        return 5.0

    def frame_size(self):
        return (224, 224)

    def attributes(self):
        return {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}

    def __iter__(self):
        for i, rgb in enumerate(images):
            yield VideoFrame(raw_frame_idx=i, rgb=torch.as_tensor(rgb, device='cuda').float() / 255,
                             intrinsics=intrinsics.clone(), camera_type=CameraType.PINHOLE)


slam_cfg = OmegaConf.load(args.source / 'configs/slam/default.yaml')
slam_cfg.keyframe_depth = 'moge2-l'
slam_cfg.optimize_intrinsics = False
slam_cfg.buffer = len(images) + int(slam_cfg.infill_chunk_size) + 8
(args.output / 'slam_config.yaml').write_text(OmegaConf.to_yaml(slam_cfg))
print('INFERENCE_START', len(images), 'frames', torch.cuda.get_device_name(0), flush=True)
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
with torch.inference_mode():
    system = SLAMSystem(torch.device('cuda'), slam_cfg)
    result = system.run([RGBVideo()], camera_type=CameraType.PINHOLE)
    poses = result.get_view_trajectory(0).matrix().cpu().numpy()
    keyframes = np.asarray(result.keyframe_ids)
    final_K = result.intrinsics.cpu().numpy()
torch.cuda.synchronize()
assert poses.shape == (len(images), 4, 4) and np.isfinite(poses).all()
path = args.output / 'predicted_camera_c2w.npz'
np.savez_compressed(path, data=poses, inds=np.arange(len(images)), keyframes=keyframes, intrinsics=final_K)
receipt = {
    'method': 'ViPE SLAM core with MoGe2-L keyframe depth prior',
    'variant': 'fixed nominal K; no dynamic/sky masks; upstream dense visual infill',
    'input_video_sha256': config['video_sha256'], 'input_config_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
    'decoded_rgb_sha256': decoded_hashes, 'frame_count': len(images), 'keyframe_count': len(keyframes),
    'poses_or_depth_labels_used_for_inference': False, 'predicted_poses_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'pose_convention': 'camera-to-world, OpenCV optical axes, arbitrary world gauge',
    'scale': 'learned MoGe2 prior; not fitted to reference during inference',
    'K_initial': config['K'], 'intrinsics_output': final_K.tolist(), 'seconds': time.monotonic() - started,
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    'time_axis': 'frame/navigation-step index; playback FPS does not specify physical motion speed',
}
(args.output / 'inference_receipt.json').write_text(json.dumps(receipt, indent=2))
print('INFERENCE_COMPLETE', json.dumps({k: receipt[k] for k in ['frame_count', 'keyframe_count', 'seconds']}), flush=True)
