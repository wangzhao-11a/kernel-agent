"""Level 2 — scaled dot-product attention (flash-attention style).

Reference: torch.nn.functional.scaled_dot_product_attention on (B, H, S, D) fp32.
Optimization target: online softmax + tiling, never materializing the S x S score
matrix; this is the shape of optimization that matters for real inference serving.
"""
import torch
import torch.nn.functional as F

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
    "name": "sdpa_attention",
    "level": 2,
    "description": (
        "Non-causal scaled dot-product attention: out = softmax(Q K^T / sqrt(D)) V, "
        "Q/K/V of shape (4, 8, 512, 64) fp32."
    ),
    "tolerance": {"atol": 2e-4, "rtol": 2e-4},
    "num_trials": 2,
    "warmup": 5,
    "rep": 30,
    "input_specs": [
        {"kind": "randn", "shape": [4, 8, 512, 64], "dtype": "float32"},
        {"kind": "randn", "shape": [4, 8, 512, 64], "dtype": "float32"},
        {"kind": "randn", "shape": [4, 8, 512, 64], "dtype": "float32"},
    ],
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.2,
}


def reference(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)