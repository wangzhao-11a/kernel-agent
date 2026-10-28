"""Level 1 — RMSNorm (reduction + normalized scaling).

Reference: x * rsqrt(mean(x^2) + eps) * w over the last dim.
Optimization target: keep x in registers/*SRAM* for the second pass instead of
re-reading it, and avoid separate pow/mean/rsqrt passes.
"""
import torch

EPS = 1e-5

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
    "name": "rmsnorm",
    "level": 1,
    "description": "RMSNorm over the last dim: out = x * rsqrt(mean(x^2) + 1e-5) * w, x=(2048, 4096) fp32, w=(4096,) fp32.",
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 3,
    "warmup": 10,
    "rep": 50,
    "input_specs": [
        {"kind": "randn", "shape": [2048, 4096], "dtype": "float32"},
        {"kind": "rand", "shape": [4096], "dtype": "float32"},
    ],
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.1,
}


def reference(x, w):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * w