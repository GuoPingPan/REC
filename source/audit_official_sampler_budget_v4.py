"""Execute the frozen sampler classes without allocating a model or GPU."""
from pathlib import Path
import argparse, ast, hashlib, inspect, json, random, time
from typing import Optional
import torch
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--dataset', default='complete-optical-screened-metric-v3')
ap.add_argument('--seed', type=int, required=True)
a = ap.parse_args()
R = Path(a.root).resolve()
S = Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source']).resolve()
source = S/'llava/train/llava_trainer.py'
tree = ast.parse(source.read_text(errors='replace'))
classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name in ['VILADistributedSampler', 'LongVILADistributedSampler']]
ns = {'DistributedSampler': DistributedSampler, 'Optional': Optional, 'torch': torch, 'random': random}
exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), ns)
inner = inspect.getsource(Trainer._inner_training_loop)
assert 'len_dataloader // args.gradient_accumulation_steps' in inner
class SizeOnlyDataset:
    def __init__(self, size): self.size = size
    def __len__(self): return self.size

validation_path=R/'dataset'/a.dataset/'validation_report.json'
counts = json.loads(validation_path.read_text())['sample_counts']
assert a.seed>=0
output=R/'report'/('official_sampler_budget_audit_'+a.dataset+'_seed-%d_v4.json'%a.seed)
assert not output.exists(), 'Preserve immutable per-seed sampler reports'
cases = []
for group, size in counts.items():
    for world, batch, accumulation in [(16,4,4)]:
        global_batch = world*batch*accumulation
        samples = []
        consumed = []
        lengths = []
        per_rank_consumed_hashes = []
        for rank in range(world):
            sampler = ns['LongVILADistributedSampler'](SizeOnlyDataset(size), num_replicas=world, rank=rank,
                                                      seed=a.seed, batch_size=batch, sample_len_list=[size], sp_degree=1)
            indices = list(sampler)
            assert len(indices) == len(sampler)
            lengths.append(len(indices))
            samples.extend(indices)
            updates = (len(indices)//batch)//accumulation
            consumed.extend(indices[:updates*batch*accumulation])
            per_rank_consumed_hashes.append(hashlib.sha256(json.dumps(indices[:updates*batch*accumulation]).encode()).hexdigest())
        assert len(set(lengths)) == 1
        assert len(samples) == len(set(samples))
        assert len(consumed) == len(set(consumed)) == updates*global_batch
        case = {'group': group, 'annotation_samples': size, 'world_size': world, 'per_GPU_batch': batch,
                'gradient_accumulation': accumulation, 'global_batch': global_batch, 'epoch': 1,
                'per_rank_sampler_samples': lengths[0], 'per_rank_micro_batches': lengths[0]//batch,
                'optimizer_steps': updates, 'sampler_unique_samples': len(samples),
                'samples_contributing_to_optimizer_updates': len(consumed),
                'dropped_annotation_samples': size-len(consumed),
                'sampler_indices_sha256': hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
                'consumed_indices_sha256': hashlib.sha256(json.dumps(consumed).encode()).hexdigest(),
                'per_rank_consumed_indices_sha256':per_rank_consumed_hashes}
        cases.append(case)
for world in [16]:
    pair = [x for x in cases if x['world_size']==world and x['group'].endswith('joint-boundary')]
    assert len(pair)==2 and pair[0]['consumed_indices_sha256']==pair[1]['consumed_indices_sha256']
report = {'scope': 'Actual frozen sampler class execution and installed Transformers update-floor semantics; no model, GPU allocation, or training submission.',
          'sampler_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'trainer_floor_semantics_verified': True, 'cases': cases,
          'sampler_seed':a.seed,'sampler_epoch':0,'dataset':a.dataset,
          'dataset_validation_sha256':hashlib.sha256(validation_path.read_bytes()).hexdigest(),
          'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'LongVILA_unused_batch_shuffle_calls':sum(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='batch_shuffle' for cls in classes if cls.name=='LongVILADistributedSampler' for n in ast.walk(cls)),
          'index_semantics':'Indices address the separately permuted dataset. Identical sampler index hashes across seeds can reflect the released unused shuffle return; this does not prove identical training data order.',
          'caveat': 'Dataset initialization also applies a seeded permutation. Matched GT and REC must use the same frozen input order and data seed. Different node counts can discard different positions.',
          'time': time.time()}
with output.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps({'report':str(output),'sampler_seed':a.seed,'cases':len(cases),'optimizer_steps_by_case':[(c['group'],c['world_size'],c['optimizer_steps']) for c in cases]}))
