"""Frozen paired formal-training entry with stable RoPE and actual-rank audit."""
from pathlib import Path
import functools, hashlib, json, math, os, runpy, socket, time
import torch
from transformers import TrainerCallback
from rotary_compat import install_stable_qwen2_rotary
install_stable_qwen2_rotary()
from llava.train.llava_trainer import LLaVATrainer
from llava.data.dataset import LazyUpdateTripleMergeAirVLNDataset

ROOT = Path(os.environ['PAPER_TRAIN_ROOT']).resolve()
SOURCE = Path(os.environ['PAPER_FROZEN_SOURCE']).resolve()
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
RANK = int(os.environ['RANK'])
WORLD = int(os.environ['WORLD_SIZE'])
assert WORLD == PROTOCOL['world_size']
assert PROTOCOL['rotary_mode']=='stable_fp32_positions'

def write(name, value):
    p = ROOT / 'debug' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.partial')
    q.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    q.replace(p)

class Audit(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        assert state.max_steps == PROTOCOL['optimizer_steps']
        assert args.per_device_train_batch_size == PROTOCOL['per_GPU_batch']
        assert args.gradient_accumulation_steps == PROTOCOL['gradient_accumulation']
        assert args.seed == PROTOCOL['seed'] and args.learning_rate == PROTOCOL['learning_rate']
        assert args.tune_language_model and args.tune_mm_projector and args.tune_vision_tower
        model.config.rotary_position_mode='stable_fp32_positions'
        model.llm.config.rotary_position_mode='stable_fp32_positions'
        x = {'rotary_mode':'stable_fp32_positions','rotary_source_sha256':hashlib.sha256((Path(__file__).parent/'rotary_compat.py').read_bytes()).hexdigest(),'rank': RANK, 'world': WORLD, 'host': socket.gethostname(),
             'GPU': torch.cuda.get_device_name(), 'max_steps': state.max_steps,
             'global_batch': WORLD*args.per_device_train_batch_size*args.gradient_accumulation_steps,
             'scope': PROTOCOL['scope'],
             'trainable_parameters': sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters() if p.requires_grad)}
        write('participation-rank-%d.json' % RANK, x)
        print('PAPER_FORMAL_BEGIN', json.dumps(x), flush=True)

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        for key in ['loss', 'grad_norm', 'learning_rate']:
            if key in logs:
                assert math.isfinite(float(logs[key]))
        if RANK == 0 and 'loss' in logs:
            x = {'step': state.global_step, 'epoch': state.epoch, 'time': time.time(), **logs}
            with (ROOT/'logs/optimizer-steps.jsonl').open('a') as f:
                f.write(json.dumps(x, allow_nan=False)+'\n')
            print('PAPER_FORMAL_STEP', json.dumps(x), flush=True)

    def on_step_end(self, args, state, control, **kw):
        write('progress-rank-%d.json' % RANK, {'rank': RANK, 'step': state.global_step,
                                            'expected': PROTOCOL['optimizer_steps'], 'time': time.time()})

    def on_train_end(self, args, state, control, **kw):
        assert state.global_step == PROTOCOL['optimizer_steps']
        write('trainer-ended-rank-%d.json' % RANK, {'rank': RANK, 'steps': state.global_step,
                                                  'time': time.time(), 'checkpoint_validation_pending': True})

original_init = LLaVATrainer.__init__
@functools.wraps(original_init)
def init(self, *args, **kwargs):
    kwargs['callbacks'] = list(kwargs.get('callbacks', [])) + [Audit()]
    original_init(self, *args, **kwargs)
    assert len(self.train_dataset) == PROTOCOL['samples']
    write('dataset-loaded-rank-%d.json' % RANK,
          {'samples': len(self.train_dataset), 'source': str(SOURCE),
           'dataset_py_sha256': hashlib.sha256((SOURCE/'llava/data/dataset.py').read_bytes()).hexdigest()})
LLaVATrainer.__init__ = init

original_loss = LLaVATrainer.compute_loss
@functools.wraps(original_loss)
def loss(self, model, inputs, *args, **kwargs):
    assert 'sample_weight' not in inputs
    assert bool((inputs['labels'] != -100).any())
    if not getattr(self, '_paper_first_batch_saved', False):
        self._paper_first_batch_saved = True
        labels = inputs['labels'].detach().cpu()
        targets = [self.tokenizer.decode(row[row != -100]) for row in labels]
        images = inputs['images']
        tensors = images if isinstance(images, list) else [images]
        write('actual-first-batch-rank-%d.json' % RANK,
              {'supervised_texts': targets, 'image_shapes': [list(x.shape) for x in tensors],
               'image_means': [float(x.detach().float().mean()) for x in tensors],
               'unweighted_released_collator': True})
    value = original_loss(self, model, inputs, *args, **kwargs)
    ll = value[0] if isinstance(value, tuple) else value
    assert bool(torch.isfinite(ll).all())
    return value
LLaVATrainer.compute_loss = loss

original_load = LazyUpdateTripleMergeAirVLNDataset._load_video
def load(*args, **kwargs):
    images, valid = original_load(*args, **kwargs)
    assert valid, 'No silent RGB failure'
    return images, valid
LazyUpdateTripleMergeAirVLNDataset._load_video = staticmethod(load)

runpy.run_path(str(SOURCE/'llava/train/train_mem.py'), run_name='__main__')
write('process-complete-rank-%d.json' % RANK, {'rank': RANK, 'time': time.time(),
                                            'checkpoint_shard_integrity_pending': True})
