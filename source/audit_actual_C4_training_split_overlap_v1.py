"""Audit actual retained C4 training exposure without modifying any split.

Full official Train overlap is independently reproduced before measuring the
actual5915 retained instruction IDs. Exact metadata overlap is not RGB pixel
identity, semantic instruction validation, or a physical collision certificate.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import time

from sampler_seed_contract_v2 import personal_path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def number(x):
    x = float(x)
    assert math.isfinite(x)
    return 0.0 if x == 0 else x


def geometry(ep):
    return int(ep['scene_id']), tuple(tuple(number(x) for x in point) for point in ep['reference_path'])


def text(ep):
    result = ep['instruction']['instruction_text']
    assert isinstance(result, str)
    return result


def execution(ep, key):
    acts = tuple(int(x) for x in ep['actions'])
    assert all(x == y and 0 <= y <= 7 for x, y in zip(ep['actions'], acts))
    return key, tuple(number(x) for x in ep['start_position']), tuple(number(x) for x in ep['start_rotation']), acts


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, required=True)
    a = cli.parse_args()
    root = a.root.resolve()
    cohort = root / 'dataset/complete-optical-target-valid-v4'
    common = personal_path(root, '/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
    files = {'train': common / 'inventory/official_train.json',
             'val_seen': common / 'eval/official_annotations/val_seen.json',
             'val_unseen': common / 'eval/official_annotations/val_unseen.json',
             'C4_inventory': cohort / 'route_inventory.json', 'C4_validation': cohort / 'validation_report.json'}
    before = {key: sha(p) for key, p in files.items()}
    original_audit = read(root / 'report/official_split_audit.json')
    verified = read(root / 'report/target_valid_dataset_verification_001.json')['verification']
    assert before['C4_inventory'] == verified['route_inventory_sha256']
    assert before['C4_validation'] == verified['validation_report_sha256']
    for split in ['train', 'val_seen', 'val_unseen']:
        assert before[split] == original_audit[split]['official_file_sha256']
    episodes = {split: read(files[split])['episodes'] for split in ['train', 'val_seen', 'val_unseen']}
    train = {str(ep['episode_id']): ep for ep in episodes['train']}
    assert len(train) == len(episodes['train']) == 10113
    train_keys = {eid: geometry(ep) for eid, ep in train.items()}
    full_keys = set(train_keys.values())
    assert len(full_keys) == original_audit['train']['exact_full_numeric_path_groups'] == 3840
    rows = read(files['C4_inventory'])
    assert len(rows) == verified['routes'] == 2264
    ids = [str(eid) for row in rows for eid in row['episodes']]
    assert len(ids) == len(set(ids)) == verified['instructions'] == 5915
    assert set(ids) <= set(train)
    selected = set(ids)
    selected_geometry = {train_keys[eid] for eid in selected}
    selected_execution = {execution(train[eid], train_keys[eid]) for eid in selected}
    selected_geometry_text = {(train_keys[eid], text(train[eid])) for eid in selected}
    selected_execution_text = {(execution(train[eid], train_keys[eid]), text(train[eid])) for eid in selected}
    geometry_to_route_ids = collections.defaultdict(set)
    multi_execution_route_ids = []
    for row in rows:
        keys = {train_keys[str(eid)] for eid in row['episodes']}
        assert len(keys) == 1 and next(iter(keys))[0] == int(row['scene_id'])
        geometry_to_route_ids[next(iter(keys))].add(row['route_id'])
        if len({execution(train[str(eid)], train_keys[str(eid)]) for eid in row['episodes']}) != 1:
            multi_execution_route_ids.append(row['route_id'])
    results = {}
    field_names = ['reference_path_present_in_C4', 'declared_execution_metadata_present_in_C4',
                   'reference_path_and_verbatim_instruction_present_in_C4',
                   'declared_execution_metadata_and_verbatim_instruction_present_in_C4',
                   'episode_ID_in_C4', 'same_ID_full_episode_payload_equal']
    for split, expected in [('val_seen', 333), ('val_unseen', 531)]:
        source = episodes[split]
        assert len(source) == len({str(ep['episode_id']) for ep in source}) == expected
        keys = {geometry(ep) for ep in source}
        old = original_audit['train__' + split]
        assert len(keys) == original_audit[split]['exact_full_numeric_path_groups']
        assert len(keys & full_keys) == old['shared_exact_numeric_paths']
        assert sum(geometry(ep) in full_keys for ep in source) == old['episodes_in_second']
        records = []
        for ep in source:
            eid = str(ep['episode_id'])
            key, label = geometry(ep), text(ep)
            exe = execution(ep, key)
            records.append({'episode_id': eid, 'scene_id': int(ep['scene_id']),
                            'full_numeric_reference_path_group_sha256': hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest(),
                            'reference_path_present_in_C4': key in selected_geometry,
                            'declared_execution_metadata_present_in_C4': exe in selected_execution,
                            'reference_path_and_verbatim_instruction_present_in_C4': (key, label) in selected_geometry_text,
                            'declared_execution_metadata_and_verbatim_instruction_present_in_C4': (exe, label) in selected_execution_text,
                            'episode_ID_in_C4': eid in selected,
                            'same_ID_full_episode_payload_equal': eid in selected and train[eid] == ep,
                            'matched_C4_route_ids_by_reference_path': sorted(geometry_to_route_ids.get(key, []))})
        by_scene = {}
        for scene in sorted({r['scene_id'] for r in records}):
            subset = [r for r in records if r['scene_id'] == scene]
            by_scene[str(scene)] = {'instructions': len(subset),
                                    **{name: sum(r[name] for r in subset) for name in field_names}}
        counts = {name: sum(r[name] for r in records) for name in field_names}
        assert all(sum(r[name] for r in by_scene.values()) == counts[name] for name in field_names)
        results[split] = {'instructions': expected, 'exact_numeric_reference_path_groups': len(keys),
                          'full_official_Train_reference_path_overlap': {'groups': len(keys & full_keys),
                                                                        'instructions': old['episodes_in_second']},
                          'C4_reference_path_overlap_groups': len(keys & selected_geometry),
                          'C4_reference_path_disjoint_groups': len(keys - selected_geometry),
                          'C4_instruction_exposure_counts': counts, 'by_scene': by_scene,
                          'episode_exposure_records': records}
    assert not set(r['scene_id'] for r in results['val_unseen']['episode_exposure_records']) & {int(r['scene_id']) for r in rows}
    assert not any(results['val_unseen']['C4_instruction_exposure_counts'].values())
    for key, p in files.items():
        assert sha(p) == before[key], 'Input changed during read-only audit'
    out = root / 'report/C4_actual_training_official_split_overlap_001.json'
    assert not out.exists()
    value = {'verified': True, 'created_at': time.time(), 'dataset': 'complete-optical-target-valid-v4',
             'C4_routes': len(rows), 'C4_instructions': len(ids),
             'C4_distinct_full_numeric_reference_paths': len(selected_geometry),
             'C4_route_ids_with_multiple_declared_execution_metadata': multi_execution_route_ids,
             'definitions': {'reference_path': 'scene_id plus complete numeric reference_path, exact float equality with signed zero normalized',
                             'declared_execution_metadata': 'reference_path plus declared start_position, start_rotation, and full integer GT actions',
                             'verbatim_instruction': 'Exact instruction_text string; no lowercasing, stripping, paraphrase or semantic matching',
                             'episode_ID': 'Actual retained5915 IDs only; removed five parent instructions are not included'},
             'splits': results, 'input_files': {k: str(p) for k, p in files.items()}, 'input_sha256': before,
             'original_full_split_audit_sha256': sha(root / 'report/official_split_audit.json'),
             'C4_qualification_sha256': sha(root / 'report/target_valid_dataset_verification_001.json'),
             'source_sha256': sha(Path(__file__)), 'training_or_eval_data_modified': False,
             'official_split_reassignment': False, 'primary_uncertainty_procedure_modified': False,
             'scope': 'Descriptive metadata exposure audit for actual C4 training; no performance filtering, retraining or evaluation change. Declared trajectory metadata equality does not prove pixel-identical rendered observations or semantic instruction quality.'}
    with out.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'report': str(out), 'sha256': sha(out), 'C4_reference_groups': len(selected_geometry),
                      'C4_multi_execution_route_count': len(multi_execution_route_ids),
                      'results': {s: {k: v for k, v in r.items() if k not in ['episode_exposure_records', 'by_scene']} for s, r in results.items()}}))


if __name__ == '__main__':
    main()
