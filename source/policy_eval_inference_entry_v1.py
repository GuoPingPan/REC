"""Single-GPU runtime compatibility; execute the frozen official eval.py unchanged."""
import contextlib
import os
from pathlib import Path
import runpy
import sys


@contextlib.contextmanager
def single_process_inference(workspace):
    import torch
    import torch.distributed as dist
    assert not dist.is_initialized()
    # DeepSpeed 0.9.5 calls init_distributed even inside zero.Init(enabled=False).
    # Initialize a real one-rank backend to avoid its irrelevant MPI discovery.
    os.environ.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1')
    torch.cuda.set_device(0)
    store = Path(workspace).resolve() / 'single_process_distributed_store'
    assert not store.exists()
    dist.init_process_group(backend='nccl', init_method='file://' + str(store), rank=0, world_size=1)
    print('AEROACT_SINGLE_GPU_INFERENCE_RUNTIME', torch.cuda.get_device_name(0),
          'rank', dist.get_rank(), 'world_size', dist.get_world_size(), flush=True)
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    target = Path.cwd() / 'eval.py'
    assert target.is_file()
    sys.argv[0] = str(target)
    with single_process_inference(Path.cwd().parent):
        import random
        import numpy as np
        import torch
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        torch.cuda.manual_seed_all(10)
        print('PAIRED_EVAL_SAMPLING_SEED', 10, flush=True)
        runpy.run_path(str(target), run_name='__main__')
