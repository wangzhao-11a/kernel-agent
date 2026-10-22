"""Level 2 — grouped matmul with variable group sizes (Ascend GroupedMatmul shape).

out[start:end] = a[start:end] @ b[g] for each group g, where group sizes come from
group_list. This is the MoE fc1/fc2 shape: variable-length groups, which the eager
per-group python loop handles badly on accelerators.

Optimization targets: one grouped GEMM over all experts (single launch, balanced
tiling), avoid the per-group slice overhead; equivalent to VeOmni's grouped-GEMM work.
"""
import torch

N_GROUPS, K_DIM, N_DIM = 8, 512, 512
SIZES = (96, 32, 160, 64, 128, 48, 80, 112)
TOTAL = sum(SIZES)

TASK = {
    "name": "grouped_matmul",
    "level": 2,
    "description": (
        "Grouped matmul: 8 groups with variable token counts %s (total %d); "
        "out[start:end] = a[start:end] @ b[g]. a=(%d,512), b=(8,512,512), group_list int64 (8,), fp32."
        % (list(SIZES), TOTAL, TOTAL)
    ),
    "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    "num_trials": 2,
    "warmup": 5,
    "rep": 20,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 2.0,
    "tags": ["gemm", "moe", "grouped", "compute-bound", "fp32"],
    "op_class": "gemm",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(4242 + trial)
    a = torch.randn(TOTAL, K_DIM, generator=gen, dtype=torch.float32)
    b = torch.randn(N_GROUPS, K_DIM, N_DIM, generator=gen, dtype=torch.float32)
    group_list = torch.tensor(SIZES, dtype=torch.int64)
    return (a.to(device), b.to(device), group_list.to(device))


def reference(a, b, group_list):
    outs = []
    start = 0
    for g, size in enumerate(group_list.tolist()):
        size = int(size)
        if size:
            outs.append(a[start:start + size] @ b[g])
        start += size
    if not outs:
        return a.new_zeros((0, b.shape[-1]))
    return torch.cat(outs, dim=0)