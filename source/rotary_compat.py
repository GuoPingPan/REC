"""Explicit Qwen2 rotary modes for this experiment's pinned runtime.

stable_fp32_positions is for new training and inference. The legacy reconstruction
is exclusively a diagnostic/compatibility mode for checkpoints trained under the
verified installed ZeRO-3 BF16 construction behavior. Neither changes parameters.
"""
import contextlib
import torch

def stable_set_cos_sin_cache(self, seq_len, device, dtype):
    device = torch.device(device)
    self.max_seq_len_cached = seq_len
    # ZeRO-3 0.9.5 coerces floating arange outputs, even with dtype=float32.
    # Integer constructors survive that wrapper; convert after construction.
    with torch.autocast(device_type=device.type, enabled=False):
        positions = torch.arange(seq_len, device=device, dtype=torch.int64).to(torch.float32)
        exponents = torch.arange(0, self.dim, 2, device=device, dtype=torch.int64).to(torch.float32)/self.dim
        frequency = 1.0/(self.base**exponents)
        angles = torch.outer(positions, frequency)
        phases = torch.cat((angles, angles), dim=-1)
        self.register_buffer('inv_freq', frequency, persistent=False)
        self.register_buffer('cos_cached', phases.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', phases.sin().to(dtype), persistent=False)

def install_stable_qwen2_rotary():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    Qwen2RotaryEmbedding._set_cos_sin_cache = stable_set_cos_sin_cache

def rebuild_legacy_zero3_bf16_buffers(model):
    """Use only with explicitly identified historical ZeRO-3 BF16 checkpoints."""
    before=[(name,id(p),p._version,tuple(p.shape)) for name,p in model.named_parameters()]
    for layer in model.llm.model.layers:
        rope=layer.self_attn.rotary_emb
        device=rope.inv_freq.device
        dtype=next(layer.parameters()).dtype
        with torch.autocast(device_type=device.type,enabled=False):
            frequency=1.0/(rope.base**(torch.arange(0,rope.dim,2,device=device,dtype=torch.int64).float()/rope.dim))
            positions=torch.arange(rope.max_seq_len_cached,device=device,dtype=torch.int64).float().bfloat16()
            phases=torch.cat([torch.outer(positions.float(),frequency)]*2,dim=-1)
            rope.register_buffer('inv_freq',frequency.bfloat16().to(dtype),persistent=False)
            rope.register_buffer('cos_cached',phases.cos().bfloat16().to(dtype),persistent=False)
            rope.register_buffer('sin_cached',phases.sin().bfloat16().to(dtype),persistent=False)
    assert before==[(name,id(p),p._version,tuple(p.shape)) for name,p in model.named_parameters()]
