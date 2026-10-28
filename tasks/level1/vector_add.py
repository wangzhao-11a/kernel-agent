"""Level 1 — elementwise add (memory-bound).

Reference: out = x + y on two 1M-element fp32 tensors.
Optimization target: vectorized loads/stores, avoiding extra passes (already 1 pass),
so realistic gains come from better memory access or a kernel-level implementation.
"""
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
    "name": "vector_add",
    "level": 1,
    "description": "Elementwise add: out[i] = x[i] + y[i] for two contiguous float32 tensors of shape (1048576,).",
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 3,
    "warmup": 25,
    "rep": 100,
    "input_specs": [
        {"kind": "randn", "shape": [1048576], "dtype": "float32"},
        {"kind": "randn", "shape": [1048576], "dtype": "float32"},
    ],
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.1,
}


def reference(x, y):
    return x + y