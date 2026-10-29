"""Bind a qualified final model to streamed checksums of every tensor shard."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import time

from final_policy_checkpoint_contract_v2 import verify_final_training
from sampler_seed_contract_v2 import personal_path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training-config', type=Path, required=True)
    args = parser.parse_args()
    root, config_path = args.root.resolve(), args.training_config.resolve()
    run = config_path.parent
    assert run.is_relative_to(root / 'training_candidates_v2')
    cfg = read(config_path)
    group = cfg['group']
    receipt = run / 'debug/completion_verified.json'
    plan = {'kind': 'corrected_final_checkpoint_navigation', 'temperature': .2, 'sampling_seed': 10,
            'max_actions': 500, 'source_manifest_sha256': digest(root / 'inventory/strict_release_collator_source.json'),
            'groups': {group: {'training_config_relative': str(config_path.relative_to(root)),
                              'training_config_sha256': digest(config_path), 'training_seed': cfg['seed'],
                              'optimizer_steps': cfg['optimizer_steps'],
                              'completion_receipt_relative': str(receipt.relative_to(root)),
                              'completion_receipt_sha256': digest(receipt),
                              'training_complete_receipt_sha256': digest(run / 'debug/training-process-complete.json')}}}
    checkpoint, _, qualification = verify_final_training(root, plan, group)
    destination = run / 'debug/final_tensor_checksums.json'
    with (run / 'debug/final-checksum.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            existing = read(destination)
            assert existing['training_config_sha256'] == digest(config_path)
            print(json.dumps({'already_recorded': True, 'manifest': str(destination)}))
            return
        rows = []
        for item in qualification['checkpoint_bundle']['files']:
            path = checkpoint / item['relative_path']
            before = path.stat()
            checksum = digest(path)
            after = path.stat()
            assert before.st_size == after.st_size == item['bytes'] and before.st_mtime_ns == after.st_mtime_ns
            rows.append({'relative_path': item['relative_path'], 'bytes': before.st_size, 'sha256': checksum})
            print(json.dumps({'hashed': item['relative_path'], 'bytes': before.st_size}), flush=True)
        files = {str(path.relative_to(checkpoint)): digest(path) for path in
                 [checkpoint / 'config.json', checkpoint / 'data_args.json', checkpoint / 'trainer_state.json']}
        manifest = {'training_config_sha256': digest(config_path), 'completion_receipt_sha256': digest(receipt),
                    'checkpoint': str(checkpoint), 'group': group, 'training_seed': cfg['seed'],
                    'tensor_files': rows, 'metadata_sha256': files, 'scope': 'Full tensor-file byte hashes. This is distinct from optimizer-state resume verification.',
                    'source_sha256': digest(Path(__file__)), 'created_at': time.time()}
        with destination.open('x') as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        print(json.dumps({'manifest': str(destination), 'weight_files': len(rows),
                          'tensor_bytes': sum(row['bytes'] for row in rows)}), flush=True)


if __name__ == '__main__':
    main()
