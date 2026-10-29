"""Run the frozen final estimator; raw ViPE scale is the default."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from primitive_calibration import predict  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-mode", choices=["raw", "primitive"], default="raw")
    parser.add_argument("--complete-demo", action="store_true",
                        help="Append Stop using a known complete demonstration endpoint.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file.")
    with np.load(args.poses, allow_pickle=False) as source:
        poses = np.asarray(source["data"], dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        parser.error("Expected at least two camera-to-world poses with shape N×4×4.")
    if not np.isfinite(poses).all():
        parser.error("Poses contain non-finite values.")
    prediction = predict(poses, calibrate=args.scale_mode == "primitive")
    actions = prediction["actions"].tolist()
    if args.complete_demo:
        actions.append(0)
    result = {
        "frame_count": len(poses),
        "actions": actions,
        "action_ids": {"0": "stop", "1": "forward", "2": "left_turn",
                       "3": "right_turn", "4": "ascend", "5": "descend",
                       "6": "move_left", "7": "move_right"},
        "scale_mode": args.scale_mode,
        "scale": prediction["scale"],
        "primary_training_label_mode": args.scale_mode == "raw",
        "GT_used_for_action_values": False,
        "known_complete_endpoint_STOP": args.complete_demo,
        "known_one_primitive_per_frame_interval": True,
        "camera_forward_offset_m": 0.5,
        "scale_at_search_boundary": prediction["scale_at_search_boundary"],
        "source_prediction_sha256": hashlib.sha256(args.poses.read_bytes()).hexdigest(),
        "estimator_sha256": hashlib.sha256(
            (ROOT / "source/primitive_calibration.py").read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
