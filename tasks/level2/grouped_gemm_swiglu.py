"""Level 2 — grouped GEMM with a fused SwiGLU epilogue (MoE fc1; VeOmni #1289 shape).

out_g = silu(a_g @ w1_g) * (a_g @ w3_g) per group with variable token counts. This is the
fc1+gating pair in a SwiGLU MoE; the eager reference computes groups separately and
materializes both intermediates — exactly the pattern #1289 complains about.

Optimization targets: one grouped GEMM launch over all experts + fused epilogue (no
intermediate write), balanced tiling for variable group sizes, fp32 accumulation.
"""
import torch
import torch.nn.functional as F

N_GROUPS, K_DIM, N_DIM = 8, 256, 256
SIZES = (96, 32, 160, 64, 128, 48, 80, 112)
TOTAL = sum(SIZES)

TASK = {
    "name": "grouped_gemm_swiglu",
    "level": 2,
    "description": (
        "Grouped GEMM + SwiGLU epilogue: 8 groups with variable token counts %s (total %d); "
        "out[start:end] = silu(a[start:end] @ w1[g]) * (a[start:end] @ w3[g]). a=(%d,256), "
        "w1/w3=(8,256,256), group_list int64 (8,), fp32."
        % (list(SIZES), TOTAL, TOTAL)
    ),
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "num_trials": 2,
    "warmup": 5,
    "rep": 20,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 2.0,
    "tags": ["gemm", "moe", "grouped", "fusion", "compute-bound", "fp32"],
    "op_class": "gemm",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(9090 + trial)
    a = torch.randn(TOTAL, K_DIM, generator=gen, dtype=torch.float32)
    w1 = torch.randn(N_GROUPS, K_DIM, N_DIM, generator=gen, dtype=torch.float32) * 0.05
    w3 = torch.randn(N_GROUPS, K_DIM, N_DIM, generator=gen, dtype=torch.float32) * 0.05
    group_list = torch.tensor(SIZES, dtype=torch.int64)
    return (a.to(device), w1.to(device), w3.to(device), group_list.to(device))


def reference(a, w1, w3, group_list):
    outs = []
    start = 0
    for g, size in enumerate(group_list.tolist()):
        size = int(size)
        if size:
            a_g = a[start:start + size]
            outs.append(F.silu(a_g @ w1[g]) * (a_g @ w3[g]))
        start += size
    if not outs:
        return a.new_zeros((0, w1.shape[-1]))
    return torch.cat(outs, dim=0)