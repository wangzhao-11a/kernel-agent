"""Level 2 — RMSNorm followed by RoPE (a real fusion pattern in modern stacks).

Reference: normalize each (token, head) vector, then apply split-half rotary embedding —
two separate passes over the data plus a concat. The win comes from fusing both into a
single pass (compute rms, rotate, write once) and avoiding the concat allocation.

Optimization targets: fused single-pass norm+rotate, reused output buffer, vectorized
aligned access, keeping the reduction on device.
"""
import torch

S_TOK, HEADS, D = 1024, 16, 64
EPS = 1e-5

TASK = {
    "name": "rmsnorm_rope_fused",
    "level": 2,
    "description": (
        "x (1024,16,64) fp32, rms weight w (64,), cos/sin (1024,1,32): first "
        "rn = x * rsqrt(mean(x^2, last)+1e-5) * w, then split-half RoPE on rn: "
        "out = cat([r1*cos - r2*sin, r2*cos + r1*sin]) with r1 = rn[..., :32]."
    ),
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 3,
    "warmup": 10,
    "rep": 50,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.3,
    "tags": ["rmsnorm", "rope", "fusion", "elementwise", "reduction", "memory-bound", "fp32"],
    "op_class": "reduction",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7070 + trial)
    x = torch.randn(S_TOK, HEADS, D, generator=gen, dtype=torch.float32)
    w = torch.rand(D, generator=gen, dtype=torch.float32)
    cos = torch.rand(S_TOK, 1, D // 2, generator=gen, dtype=torch.float32)
    sin = torch.rand(S_TOK, 1, D // 2, generator=gen, dtype=torch.float32)
    return (x.to(device), w.to(device), cos.to(device), sin.to(device))


def reference(x, w, cos, sin):
    rn = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * w
    half = D // 2
    r1, r2 = rn[..., :half], rn[..., half:]
    return torch.cat([r1 * cos - r2 * sin, r2 * cos + r1 * sin], dim=-1)