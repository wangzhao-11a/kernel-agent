"""Level 2 — rotary position embedding (RoPE, split-half/NeoX style).

Every token's head vector is rotated by cos/sin. This is the classic memory-bound op in
attention stacks (and part of the audio8 RoPE work): the eager reference materializes both
halves and concatenates them; the agent should do it in one pass without the extra
allocation, with vectorized aligned access.

Optimization targets: single-pass multiply-add without `cat`, avoid temporaries,
vectorized loads/stores, merge with adjacent elementwise ops where the pipeline allows.
"""
import torch

B, H, S, D = 4, 8, 1024, 64

TASK = {
    "schema_version": 1,
    "policy": {
        "forbid_native_compute": False,
        "require_kernel": False,
        "torch_fused_exempt": True,
        "forbid_reference": True,
        "allow_nonfinite": False,
        "reason": "legacy torch reference / replay fixture exemption",
    },
    "name": "rope",
    "level": 2,
    "description": (
        "Rotary embedding on (4,8,1024,64) fp32 with cos/sin of shape (1024,32): "
        "with x1 = x[..., :32], x2 = x[..., 32:], "
        "out = cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)."
    ),
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 3,
    "warmup": 10,
    "rep": 50,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.3,
    "tags": ["attention", "rope", "elementwise", "memory-bound", "fusion", "fp32"],
    "op_class": "elementwise",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2020 + trial)
    x = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
    cos = torch.rand(S, D // 2, generator=gen, dtype=torch.float32)
    sin = torch.rand(S, D // 2, generator=gen, dtype=torch.float32)
    return (x.to(device), cos.to(device), sin.to(device))


def reference(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)