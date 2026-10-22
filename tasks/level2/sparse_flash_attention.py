"""Level 2 — sparse flash attention (top-k keys per query; Ascend 910B target).

Each query attends only TOPK keys, whose indices are given. This is the shape of the
Ascend SparseFlashAttention workload: the optimization target is skipping work for
unselected keys and keeping the index path cheap, not just fusing softmax.

Optimization targets: avoid materializing the dense S x S score matrix, vectorize the
gather of K/V, keep fp32 accumulation, tile over the selected keys.
"""
import math

import torch

B, H, S, D = 2, 4, 256, 64
TOPK = 32

TASK = {
    "name": "sparse_flash_attention",
    "level": 2,
    "description": (
        "Top-k sparse attention: for every query only TOPK=32 keys are attended; indices are "
        "given as int64 (2,4,256,32). out = softmax(Q K_sel^T / sqrt(D)) V_sel. "
        "Q/K/V are fp32 (2,4,256,64)."
    ),
    "tolerance": {"atol": 2e-4, "rtol": 2e-4},
    "num_trials": 2,
    "warmup": 5,
    "rep": 30,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.5,
    "tags": ["attention", "sparse", "flash", "memory-bound", "fp32"],
    "op_class": "attention",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026 + trial)
    q = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
    k = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
    v = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
    idx = torch.randint(0, S, (B, H, S, TOPK), generator=gen, dtype=torch.int64)
    return (q.to(device), k.to(device), v.to(device), idx.to(device))


def reference(q, k, v, idx):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.gather(q @ k.transpose(-1, -2) * scale, -1, idx)     # (B,H,S,TOPK)
    probs = torch.softmax(scores, dim=-1)
    idx_expand = idx.unsqueeze(-1).expand(B, H, S, TOPK, D)
    v_sel = torch.gather(v.unsqueeze(2).expand(B, H, S, S, D), 3, idx_expand)
    return (probs.unsqueeze(-1) * v_sel).sum(dim=-2)