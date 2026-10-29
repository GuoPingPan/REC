"""Verify copied code bytes and the actual final training/evaluation contracts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT / "provenance/source_manifest.json")
    for relative, item in manifest["files"].items():
        assert sha(ROOT / relative) == item["sha256"], relative
    training = list((ROOT / "configs/training").glob("*.json"))
    assert len(training) == 6
    pairs = set()
    for path in training:
        config = read(path)
        pairs.add((config["group"], config["seed"]))
        for key, expected in {"nodes": 4, "world_size": 32, "global_batch": 256,
                              "epochs": 1, "samples": 351983,
                              "optimizer_steps": 1374}.items():
            assert config[key] == expected, (path.name, key)
        for key, name in {"bootstrap_sha256": "formal_training_bootstrap_v6.py",
                          "entry_sha256": "formal_training_entry.py",
                          "rotary_helper_sha256": "rotary_compat.py",
                          "sampler_contract_sha256": "sampler_seed_contract_v2.py",
                          "target_valid_contract_sha256": "target_valid_data_contract_v1.py"}.items():
            assert config[key] == sha(ROOT / "source" / name), (path.name, key)
    assert pairs == {(group, seed) for group in ["gt-joint-boundary", "rec-joint-boundary"]
                     for seed in [10, 11, 12]}
    evaluation = list((ROOT / "configs/evaluation").glob("*.json"))
    assert len(evaluation) == 6
    for path in evaluation:
        plan = read(path)
        assert (plan["temperature"], plan["sampling_seed"], plan["max_actions"]) == (.2, 10, 500)
        assert plan["splits"]["val_seen"]["episodes"] == 333
        assert plan["splits"]["val_unseen"]["episodes"] == 531
        assert plan["source_manifest_sha256"] == sha(
            ROOT / "provenance/strict_release_collator_source.json")
        for name, expected in plan["runtime_source_sha256"].items():
            assert sha(ROOT / "source" / name) == expected, name
        assert plan["eval_py_sha256"] == sha(ROOT / "third_party/aeroact/eval.py")
    original = read(ROOT / "provenance/strict_release_collator_source.json")
    prefix = ROOT / "third_party/aeroact"
    for path in prefix.rglob("*"):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            relative = str(path.relative_to(prefix))
            assert sha(path) == original["files"][relative], relative
    for relative, target in original["symlinks"].items():
        path = prefix / relative
        assert path.is_symlink() and str(path.readlink()) == target
    print(json.dumps({"copied_files_verified": len(manifest["files"]),
                      "training_configs_verified": len(training),
                      "evaluation_plans_verified": len(evaluation),
                      "cloud_jobs_launched": 0}, indent=2))


if __name__ == "__main__":
    main()
