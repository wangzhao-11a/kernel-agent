"""Level 2 — single-token decode attention over a KV cache (the real serving shape).

q is one token (S_q = 1); K/V are the cached context. Memory-bound: every call streams
2 * B*H*S*D elements of cache, so the win comes from streaming K/V efficiently, not from
compute (the 1 x S score row is tiny). This is what decode-time kernels look like.

Optimization targets: a single pass over the cache, vectorized aligned loads, no
per-head temporaries, keeping the softmax reduction on device.
"""
import torch
import torch.nn.functional as F

B, H, S, D = 2, 8, 4096, 64

TASK = {
    "name": "decode_attention",
    "level": 2,
    "description": (
        "Decode-step attention with a KV cache: q (2,8,1,64), k/v (2,8,4096,64) fp32; "
        "out = softmax(q K^T / sqrt(64)) V, out shape (2,8,1,64)."
    ),
    "tolerance": {"atol": 2e-4, "rtol": 2e-4},
    "num_trials": 3,
    "warmup": 10,
    "rep": 50,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.2,
    "tags": ["attention", "decode", "kv-cache", "memory-bound", "fp32"],
    "op_class": "attention",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(6060 + trial)
    q = torch.randn(B, H, 1, D, generator=gen, dtype=torch.float32)
    k = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
    v = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
    return (q.to(device), k.to(device), v.to(device))


def reference(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)