"""Read-only seed/budget qualification shared by future paired-run tools."""
from pathlib import Path
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def personal_path(root, value):
    prefixes = ('/mnt/data/users/kindel/uav', '/mnt/data_20t/users/kindel/uav', '/mnt/data/uav')
    root = str(Path(root).resolve())
    actual = next((p for p in prefixes if root == p or root.startswith(p + '/')), None)
    value = str(value)
    if actual:
        for prefix in prefixes:
            if value == prefix or value.startswith(prefix + '/'):
                return (Path(actual) / value[len(prefix):].lstrip('/')).resolve()
    return Path(value).resolve()


def verify_sampler_seed(root, config):
    root = Path(root).resolve()
    relative = Path(config['sampler_budget_relative'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = (root / relative).resolve()
    assert path.is_relative_to(root / 'report'), 'Sampler receipt must be an experiment report'
    assert sha(path) == config['sampler_budget_sha256']
    report = json.loads(path.read_text())
    assert report['sampler_seed'] == config['seed'], 'Sampler report seed mismatch'
    assert report['sampler_epoch'] == 0 and report['trainer_floor_semantics_verified']
    assert report['dataset_validation_sha256'] == config['dataset_validation_sha256']
    assert report['dataset'] == 'complete-optical-target-valid-v4'
    assert personal_path(root, config['dataset']).parent == root/'dataset/complete-optical-target-valid-v4'
    source = personal_path(root, config['frozen_source']) / 'llava/train/llava_trainer.py'
    assert sha(source) == report['sampler_source_sha256']
    args = config['training_args']
    for flag in ['--seed', '--data_seed']:
        assert args.count(flag) == 1 and int(args[args.index(flag) + 1]) == config['seed'], flag
    assert config['epochs'] == 1 and config['global_batch'] == 256
    assert config['world_size'] == config['nodes'] * 8
    assert config['global_batch'] == config['world_size'] * config['per_GPU_batch'] * config['gradient_accumulation']
    cases = [value for value in report['cases'] if value['group'] == config['group']
             and value['world_size'] == config['world_size']]
    assert len(cases) == 1
    case = cases[0]
    for report_key, config_key in [('annotation_samples', 'samples'),
                                   ('optimizer_steps', 'optimizer_steps'),
                                   ('per_GPU_batch', 'per_GPU_batch'),
                                   ('gradient_accumulation', 'gradient_accumulation'),
                                   ('global_batch', 'global_batch'), ('epoch', 'epochs'),
                                   ('consumed_indices_sha256', 'sampler_consumed_indices_sha256')]:
        assert case[report_key] == config[config_key], (report_key, config_key)
    assert case['samples_contributing_to_optimizer_updates'] == config['optimizer_steps'] * 256
    assert len(case['per_rank_consumed_indices_sha256']) == config['world_size']
    return {'sampler_seed': config['seed'], 'sampler_epoch': 0,
            'sampler_budget_sha256': sha(path), 'case': case}
