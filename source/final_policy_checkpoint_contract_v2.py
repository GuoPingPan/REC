"""CPU-only qualification checks for corrected final-model navigation workers.

Historical checkpoint format can be inspected with verify_tensor_bundle alone.
That inspection does not qualify it as a new corrected training result.
"""
from pathlib import Path
import hashlib
import json
import math
import struct
from sampler_seed_contract_v2 import personal_path, verify_sampler_seed
from target_valid_data_contract_v1 import verify_target_valid_cohort

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_tensor_bundle(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    headers = {}
    elements = 0
    for path in sorted(checkpoint.rglob('*.safetensors')):
        rel = path.relative_to(checkpoint)
        if any(part.startswith('checkpoint-') for part in rel.parts):
            continue
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < size < 64 * 1024**2
            header = json.loads(stream.read(size))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        assert tensors and path.stat().st_size == 8 + size + max(v['data_offsets'][1] for v in tensors.values())
        elements += sum(math.prod(v['shape']) for v in tensors.values())
        headers[path] = tensors
    assert len(headers) == 6 and elements == 8102301504, (len(headers), elements)
    for path in checkpoint.rglob('*.safetensors.index.json'):
        if any(part.startswith('checkpoint-') for part in path.relative_to(checkpoint).parts):
            continue
        for name, shard in read(path)['weight_map'].items():
            assert name in headers[(path.parent / shard).resolve()]
    return {'weight_files': len(headers), 'tensor_elements': elements,
            'files': [{'relative_path': str(p.relative_to(checkpoint)), 'bytes': p.stat().st_size}
                      for p in headers], 'scope': 'Shape/header/index/size integrity; not a full weight checksum or inference test.'}

def verify_final_training(root, plan, group):
    root = Path(root).resolve()
    def relative(name):
        p = (root / name).resolve()
        assert p.is_relative_to(root), name
        return p
    def host_path(name):
        return personal_path(root, name)
    assert plan['kind'] == 'corrected_final_checkpoint_navigation'
    assert plan['temperature'] == .2 and plan['sampling_seed'] == 10 and plan['max_actions'] == 500
    source_proof = root / 'inventory/strict_release_collator_source.json'
    assert sha(source_proof) == plan['source_manifest_sha256']
    entry = plan['groups'][group]
    cfg_path = relative(entry['training_config_relative'])
    assert sha(cfg_path) == entry['training_config_sha256']
    cfg = read(cfg_path)
    assert cfg['group'] == group and cfg['seed'] == entry['training_seed']
    assert cfg['optimizer_steps'] == entry['optimizer_steps']
    assert cfg['epochs'] == 1 and cfg['global_batch'] == 256
    for name, field in [('sampler_seed_contract_v2.py', 'sampler_contract_sha256'),
                        ('target_valid_data_contract_v1.py', 'target_valid_contract_sha256'),
                        ('formal_training_bootstrap_v6.py', 'bootstrap_sha256'),
                        ('formal_training_entry.py', 'entry_sha256'),
                        ('rotary_compat.py', 'rotary_helper_sha256')]:
        assert sha(root / 'source' / name) == cfg[field], ('Training source changed', name)
    sampler_proof = verify_sampler_seed(root, cfg)
    data_proof = verify_target_valid_cohort(root)
    target_receipt = root / 'report/target_valid_dataset_verification_001.json'
    assert sha(target_receipt) == cfg['target_valid_verification_sha256']
    assert read(target_receipt)['verification'] == data_proof
    assert cfg['rotary_mode'] == 'stable_fp32_positions'
    assert cfg['frozen_source_manifest_sha256'] == plan['source_manifest_sha256']
    run = host_path(cfg['run_root'])
    assert run == cfg_path.parent and run.is_relative_to(root / 'training_candidates_v2')
    assert host_path(cfg['root']) == root
    checkpoint = host_path(cfg['output_dir'])
    assert checkpoint.is_relative_to(run) and checkpoint.name == 'final'
    completion_path = relative(entry['completion_receipt_relative'])
    assert sha(completion_path) == entry['completion_receipt_sha256']
    completion = read(completion_path)
    platform_proof = verify_platform_receipt(root, run, cfg, entry['training_config_sha256'], completion)
    assert completion['training_config_sha256'] == entry['training_config_sha256']
    assert completion['optimizer_steps'] == cfg['optimizer_steps']
    assert host_path(completion['final_checkpoint_path']) == checkpoint
    done_path = run / 'debug/training-process-complete.json'
    assert sha(done_path) == entry['training_complete_receipt_sha256']
    done = read(done_path)
    assert done['all_nodes_exit_zero'] and done['ranks_completed'] == cfg['world_size']
    assert done['optimizer_steps'] == cfg['optimizer_steps'] and done['rotary_mode'] == cfg['rotary_mode']
    assert host_path(done['final_checkpoint']) == checkpoint
    protocol = read(run / 'protocol.json')
    for key in ['world_size', 'samples', 'optimizer_steps', 'seed', 'rotary_mode']:
        assert protocol[key] == cfg[key]
    for node in range(cfg['nodes']):
        assert read(run / 'debug' / ('node-exit-%d.json' % node))['exit_code'] == 0
        assert not (run / 'debug' / ('bootstrap-error-node-%d.json' % node)).exists()
        launch = read(run / 'debug' / ('launch-node-%d.json' % node))
        assert launch['node'] == node and launch['config_sha256'] == entry['training_config_sha256']
        assert launch['dataset_validation_sha256'] == cfg['dataset_validation_sha256']
        assert launch['source_manifest_sha256'] == cfg['frozen_source_manifest_sha256']
        assert launch['rotary_mode'] == cfg['rotary_mode']
    for rank in range(cfg['world_size']):
        assert read(run / 'debug' / ('trainer-ended-rank-%d.json' % rank))['steps'] == cfg['optimizer_steps']
        assert read(run / 'debug' / ('process-complete-rank-%d.json' % rank))['rank'] == rank
        participation = read(run / 'debug' / ('participation-rank-%d.json' % rank))
        assert participation['rank'] == rank and participation['world'] == cfg['world_size']
        assert participation['max_steps'] == cfg['optimizer_steps']
        assert participation['global_batch'] == cfg['global_batch']
        assert participation['rotary_mode'] == cfg['rotary_mode']
        loaded = read(run / 'debug' / ('dataset-loaded-rank-%d.json' % rank))
        assert loaded['samples'] == cfg['samples']
        assert loaded['dataset_py_sha256'] == read(source_proof)['files']['llava/data/dataset.py']
    optimizer_path = run / 'logs/optimizer-steps.jsonl'
    steps = [json.loads(line) for line in optimizer_path.read_text().splitlines()]
    assert [x['step'] for x in steps] == list(range(1, cfg['optimizer_steps'] + 1))
    assert all('loss' in row for row in steps)
    assert all(math.isfinite(float(row[key])) for row in steps
               for key in ('loss', 'grad_norm', 'learning_rate') if key in row)
    assert read(checkpoint / 'trainer_state.json')['global_step'] == cfg['optimizer_steps']
    assert read(checkpoint / 'config.json')['num_video_frames'] == 8
    assert (checkpoint / 'data_args.json').is_file()
    cohort = host_path(cfg['dataset']).parent
    assert cohort == root / 'dataset/complete-optical-target-valid-v4'
    assert not (cohort / 'TRAINING_USE_HOLD.json').exists()
    ready = read(cohort / 'READY.json')
    assert ready['status'] == 'ready'
    assert sha(cohort / 'validation_report.json') == cfg['dataset_validation_sha256'] == ready['validation_report_sha256']
    assert read(cohort / 'validation_report.json')['sample_counts'][group] == cfg['samples']
    bundle = verify_tensor_bundle(checkpoint)
    assert bundle['weight_files'] == done['weight_files'] and bundle['tensor_elements'] == done['tensor_elements']
    return checkpoint, host_path(cfg['model']), {
        'training_config_sha256': entry['training_config_sha256'],
        'completion_receipt_sha256': entry['completion_receipt_sha256'],
        'training_complete_receipt_sha256': entry['training_complete_receipt_sha256'],
        'optimizer_steps': cfg['optimizer_steps'], 'training_seed': cfg['seed'],
        'platform_job_id': completion['job_id'], 'checkpoint_bundle': bundle,
        'new_corrected_final_checkpoint_qualified': True,
        'platform_proof': platform_proof,
        'sampler_budget_sha256': sampler_proof['sampler_budget_sha256'],
        'target_valid_data_proof': data_proof,
        'optimizer_log_sha256': sha(optimizer_path),
        'qualification_source_sha256': sha(Path(__file__))}


def verify_platform_receipt(root, run, cfg, config_sha256, completion):
    """Bind terminal status to the exact submission and original platform JSON."""
    root, run = Path(root).resolve(), Path(run).resolve()
    assert completion['platform_status'] == 'Succeeded' and completion['job_id']
    submission = read(run / 'submission.json')
    assert submission['status'] == 'submitted'
    assert submission['job_id'] == completion['job_id']
    assert submission['training_config_sha256'] == config_sha256
    relative = Path(completion['platform_snapshot_relative'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = (root / relative).resolve()
    assert path.is_relative_to(run / 'debug')
    assert sha(path) == completion['platform_snapshot_sha256']
    platform = read(path)
    assert platform['status'] == 'Succeeded' and platform['job_id'] == submission['job_id']
    assert str(platform['workspace_id']) == str(submission['workspace_id'])
    assert platform['resource_id'] == submission['resource_id']
    assert platform['display_name'] == submission['display_name']
    assert platform['gmt_successed_time'] and platform['gmt_finish_time']
    specs = platform['job_specs']
    assert len(specs) == 1 and specs[0]['type'] == 'Worker'
    assert specs[0]['pod_count'] == cfg['nodes'] and int(specs[0]['resource_config']['gpu']) == 8
    import shlex
    command = shlex.split(platform['user_command'])
    assert len(command) == 4 and command[2] == '--config'
    assert personal_path(root, command[1]) == root / 'source/formal_training_bootstrap_v6.py'
    assert personal_path(root, command[3]) == run / 'training_config.json'
    return {'job_id': platform['job_id'], 'status': platform['status'],
            'platform_snapshot_sha256': sha(path), 'submission_sha256': sha(run / 'submission.json')}
