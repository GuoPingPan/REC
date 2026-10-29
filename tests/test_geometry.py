"""Independent kinematics checks for the released pose-to-action entry."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from primitive_calibration import predict, verify_geometry  # noqa: E402


def camera_path(actions):
    # Construct body motions independently of the estimator's candidate arrays.
    position = np.zeros(3)
    rotation = np.eye(3)
    optical_to_body = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    poses = []
    for action in [None, *actions]:
        if action in (2, 3):
            yaw = -15 if action == 2 else 15
            rotation = rotation @ Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        elif action is not None:
            delta = {1: [5, 0, 0], 4: [0, 0, -2], 5: [0, 0, 2],
                     6: [0, -5, 0], 7: [0, 5, 0]}[action]
            position = position + rotation @ np.asarray(delta)
        pose = np.eye(4)
        pose[:3, :3] = rotation @ optical_to_body
        pose[:3, 3] = position + rotation @ np.array([.5, 0, 0])
        poses.append(pose)
    return np.asarray(poses)


def test_raw_actions_preserve_camera_lever_arm_and_global_gauge():
    actions = [1, 2, 1, 3, 4, 5, 6, 7, 2, 2, 7]
    poses = camera_path(actions)
    gauge = Rotation.from_euler("xyz", [.3, -.6, .9]).as_matrix()
    poses[:, :3, :3] = gauge @ poses[:, :3, :3]
    poses[:, :3, 3] = poses[:, :3, 3] @ gauge.T + np.array([19, -4, 32])
    result = predict(poses, calibrate=False)
    np.testing.assert_array_equal(result["actions"], actions)
    assert result["scale"] == 1


def test_primitive_scale_ablation_recovers_synthetic_scales():
    assert len(verify_geometry()) == 3


def test_cli_defaults_to_raw_scale_and_requires_explicit_stop(tmp_path):
    poses = tmp_path / "camera.npz"
    np.savez(poses, data=camera_path([1, 2, 4, 7]))
    output = tmp_path / "actions.json"
    args = [sys.executable, str(ROOT / "scripts/pose_to_actions.py"),
            "--poses", str(poses), "--output", str(output)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    data = json.loads(output.read_text())
    assert data["actions"] == [1, 2, 4, 7]
    assert data["scale_mode"] == "raw" and data["scale"] == 1
    assert data["known_complete_endpoint_STOP"] is False
    stopped = tmp_path / "complete.json"
    args[args.index("--output") + 1] = str(stopped)
    subprocess.run([*args, "--complete-demo"], check=True, capture_output=True, text=True)
    assert json.loads(stopped.read_text())["actions"] == [1, 2, 4, 7, 0]


def test_cli_rejects_nonfinite_poses_and_existing_output(tmp_path):
    poses = tmp_path / "bad.npz"
    data = camera_path([1])
    data[1, 0, 3] = np.nan
    np.savez(poses, data=data)
    output = tmp_path / "actions.json"
    args = [sys.executable, str(ROOT / "scripts/pose_to_actions.py"),
            "--poses", str(poses), "--output", str(output)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()
    output.write_text("preserved")
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and output.read_text() == "preserved"
