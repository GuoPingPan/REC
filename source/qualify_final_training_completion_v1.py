"""Observe one existing DLC job and qualify its completed final model, read-only.

Never submits a job, edits training data/config, or launches inference. A queued
or running observation is saved without publishing a completion receipt.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import time
from final_policy_checkpoint_contract_v2 import read, sha, verify_final_training
from sampler_seed_contract_v2 import personal_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training-config', type=Path, required=True)
    args = parser.parse_args()
    root, cfg_path = args.root.resolve(), args.training_config.resolve()
    assert cfg_path.is_relative_to(root / 'training_candidates_v2')
    cfg = read(cfg_path)
    run = personal_path(root, cfg['run_root'])
    assert cfg_path == run / 'training_config.json'
    submission = read(run / 'submission.json')
    assert submission['status'] == 'submitted' and submission['training_config_sha256'] == sha(cfg_path)
    debug = run / 'debug'
    debug.mkdir(exist_ok=True)
    with (debug / 'completion-qualification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = debug / 'completion_verified.json'
        assert not destination.exists(), 'Existing qualification: verify it; do not recreate.'
        stamp = time.time_ns()
        command = ['pai', 'dlc', 'job', 'get', submission['job_id'], '--region', submission['region'], '-o', 'json']
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=True)
        platform = json.loads(result.stdout)
        assert platform['job_id'] == submission['job_id']
        snapshot = debug / ('platform-qualification-observed-%d.json' % stamp)
        with snapshot.open('x') as stream:
            stream.write(result.stdout)
        if platform['status'] != 'Succeeded':
            print(json.dumps({'job_id': platform['job_id'], 'platform_status': platform['status'],
                              'allocated_pods': len(platform.get('pods') or []),
                              'platform_snapshot': str(snapshot), 'completion_published': False}))
            return 3
        # Stage a candidate, then publish it only after every real data/rank/model check.
        candidate = debug / ('completion-candidate-%d.json' % stamp)
        completion = {'job_id': platform['job_id'], 'platform_status': platform['status'],
                      'training_config_sha256': sha(cfg_path),
                      'optimizer_steps': cfg['optimizer_steps'], 'final_checkpoint_path': cfg['output_dir'],
                      'platform_snapshot_relative': str(snapshot.relative_to(root)),
                      'platform_snapshot_sha256': sha(snapshot),
                      'qualification_source_sha256': sha(Path(__file__)),
                      'scope': 'Final training termination/data/header qualification. Actual inference is separately required.',
                      'created_at': time.time()}
        with candidate.open('x') as stream:
            json.dump(completion, stream, indent=2, allow_nan=False)
        entry = {'training_config_relative': str(cfg_path.relative_to(root)),
                 'training_config_sha256': sha(cfg_path),
                 'training_seed': cfg['seed'], 'optimizer_steps': cfg['optimizer_steps'],
                 'completion_receipt_relative': str(candidate.relative_to(root)),
                 'completion_receipt_sha256': sha(candidate),
                 'training_complete_receipt_sha256': sha(debug / 'training-process-complete.json')}
        plan = {'kind': 'corrected_final_checkpoint_navigation',
                'temperature': .2, 'sampling_seed': 10, 'max_actions': 500,
                'source_manifest_sha256': sha(root / 'inventory/strict_release_collator_source.json'),
                'groups': {cfg['group']: entry}}
        _, _, qualification = verify_final_training(root, plan, cfg['group'])
        assert not destination.exists()
        os.rename(candidate, destination)
        proof = debug / 'completion_qualification.json'
        with proof.open('x') as stream:
            json.dump({'completion_receipt_sha256': sha(destination), 'qualification': qualification,
                       'source_sha256': sha(Path(__file__)), 'actual_inference_pending': True}, stream, indent=2, allow_nan=False)
        print(json.dumps({'qualified': True, 'completion_receipt': str(destination),
                          'job_id': platform['job_id'], 'actual_inference_pending': True}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
