"""Level 1 — fp32 GEMM (compute-bound).

Reference: x @ w for (1024, 1024) fp32 tensors.
Optimization target: on GPU this is where tiling / shared-memory staging / split-K
matter; on CPU the baseline already uses a tuned BLAS, so only fused epilogues or
better parallelism can win. Good "can the agent avoid regressing" task.
"""
TASK = {
    "name": "matmul",
    "level": 1,
    "description": "fp32 matrix multiply: out = x @ w, x=(1024,1024), w=(1024,1024).",
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "num_trials": 2,
    "warmup": 5,
    "rep": 20,
    "input_specs": [
        {"kind": "randn", "shape": [1024, 1024], "dtype": "float32"},
        {"kind": "randn", "shape": [1024, 1024], "dtype": "float32"},
    ],
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.0,
}


def reference(x, w):
    return x @ w