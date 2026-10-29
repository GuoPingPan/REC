"""Verify every paired annotation's input metadata without decoding RGB again.

This strengthens structural coverage. It does not replace the actual-loader
tensor checks or inherited full RGB byte verification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

from sampler_seed_contract_v2 import verify_sampler_seed


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cohort = root / 'dataset/complete-optical-target-valid-v4'
    validation = read(cohort / 'validation_report.json')
    groups = ['gt-joint-boundary', 'rec-joint-boundary']
    folders = [cohort / group for group in groups]
    annotation = [folder / 'data/aerialvln-s' for folder in folders]
    for group, folder in zip(groups, annotation):
        for name, digest in validation['annotation_sha256'][group].items():
            assert sha(folder / name) == digest
    identical_aux = ['train_episode2instruction.json', 'train_episodes2idx_merge.json']
    for name in identical_aux:
        assert (annotation[0] / name).read_bytes() == (annotation[1] / name).read_bytes()
    raw = [(folder / 'Raw_data/aerialvln-s').resolve() for folder in folders]
    assert raw[0] == raw[1], 'Both groups must resolve to exactly the same physical RGB root'
    pair = [read(folder / 'train_merged_triple.json') for folder in annotation]
    assert len(pair[0]) == len(pair[1]) == 351983
    prefix_hash = hashlib.sha256()
    changed = 0
    stop = [0, 0]
    episodes = set()
    for first, second in zip(*pair):
        assert set(first) == set(second) == {'episode', 'timestep', 'next_timestep', 'action'}
        a = {key: value for key, value in first.items() if key != 'action'}
        b = {key: value for key, value in second.items() if key != 'action'}
        assert a == b
        prefix_hash.update(json.dumps(a, sort_keys=True, separators=(',', ':')).encode() + b'\n')
        episodes.add(first['episode'])
        changed += first['action'] != second['action']
        for i, row in enumerate([first, second]):
            stop[i] += 'stop' in row['action'].lower()
    assert len(episodes) == 5915 and stop == [5915, 5915] and changed > 0
    # The actual frozen sampler budget and initial data permutation are paired
    # within each training seed, without asserting cross-seed permutations equal.
    seed_proofs = []
    for seed in [10, 11, 12]:
        configs = [read(root / 'training_candidates_v2/paired-seed-candidates-v4/4nodes' /
                        group / ('seed-%d' % seed) / 'attempt-001/training_config.json') for group in groups]
        proofs = [verify_sampler_seed(root, cfg) for cfg in configs]
        assert configs[0]['seed'] == configs[1]['seed'] == seed
        assert proofs[0]['case']['consumed_indices_sha256'] == proofs[1]['case']['consumed_indices_sha256']
        assert configs[0]['training_args'][configs[0]['training_args'].index('--data_seed') + 1] == str(seed)
        assert configs[1]['training_args'][configs[1]['training_args'].index('--data_seed') + 1] == str(seed)
        seed_proofs.append({'seed': seed, 'consumed_indices_sha256': proofs[0]['case']['consumed_indices_sha256'],
                            'sampler_budget_sha256': proofs[0]['sampler_budget_sha256']})
    proof = {'verified': True, 'dataset': cohort.name, 'paired_samples': len(pair[0]),
             'instructions': len(episodes), 'all_input_annotation_fields_equal_except_action': True,
             'ordered_input_metadata_sha256': prefix_hash.hexdigest(),
             'identical_language_and_history_index_files': {name: sha(annotation[0] / name) for name in identical_aux},
             'shared_physical_RGB_root': str(raw[0]), 'changed_action_answers': changed,
             'changed_action_answer_fraction': changed / len(pair[0]), 'Stop_samples_each': stop[0],
             'seed_sampler_proofs': seed_proofs,
             'action_weight_file_byte_identical': (annotation[0] / 'train_action_prob_weight.json').read_bytes() == (annotation[1] / 'train_action_prob_weight.json').read_bytes(),
             'source_dataset_py_sha256': read(root / 'inventory/strict_release_collator_source.json')['files']['llava/data/dataset.py'],
             'dataset_validation_sha256': sha(cohort / 'validation_report.json'),
             'scope': 'Full annotation/language/history/RGB-path and paired sampler metadata equality. Actual input tensor equality remains supported by the separate 24-pair loader check; no new full JPEG decode or tensor pass is claimed.',
             'source_sha256': sha(Path(__file__)), 'created_at': time.time()}
    output = root / 'report/full_paired_input_metadata_verified_001.json'
    with output.open('x') as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
