"""Level 1 — row-wise softmax (reduction + elementwise).

Reference: torch.softmax over the last dim of a (2048, 4096) fp32 tensor.
Optimization target: fusing the max-subtract / exp / sum / divide chain, or a
single-pass online softmax that avoids materializing intermediate tensors.
"""
import torch

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
    "name": "softmax",
    "level": 1,
    "description": "Row-wise softmax over the last dimension of a (2048, 4096) float32 tensor.",
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 3,
    "warmup": 10,
    "rep": 50,
    "input_specs": [
        {"kind": "randn", "shape": [2048, 4096], "dtype": "float32", "scale": 2.0},
    ],
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.1,
}


def reference(x):
    return torch.softmax(x, dim=-1)