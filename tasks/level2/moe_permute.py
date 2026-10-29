"""Level 2 — MoE token permutation (vLLM/VeOmni `moe_permute` shape).

Sort (token, k) pairs by routed expert so the grouped GEMM sees contiguous rows, and return
the permutation + per-expert counts. VeOmni #1288 is exactly about this layer's per-layer host
syncs and permute-order rebuilds: the eager reference re-scans the id tensor once per expert
and does `.max().item()` (a host sync) — the agent should do it in one pass, device-resident.

Optimization targets: single stable sort pass, keep counts on device, avoid per-expert scans,
cheap gather for the permuted rows.
"""
import torch

T_TOK, K_TOP, HID = 1024, 8, 256

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
    "name": "moe_permute",
    "level": 2,
    "description": (
        "MoE token permutation: x (1024,256) fp32, topk expert ids (1024,8) int64 in [0,64). "
        "Return (permuted_x (8192,256), perm (8192,) int64, counts (64,) int64): perm sorts all "
        "(token,k) pairs by expert, stable within an expert; permuted_x = x[perm // 8]; "
        "counts = per-expert pair counts."
    ),
    "tolerance": {"atol": 0.0, "rtol": 0.0},
    "num_trials": 2,
    "warmup": 3,
    "rep": 15,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 2.0,
    "tags": ["moe", "permute", "gather", "layout", "memory-bound", "fp32"],
    "op_class": "moe",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(31337 + trial)
    x = torch.randn(T_TOK, HID, generator=gen, dtype=torch.float32)
    ids = torch.randint(0, 64, (T_TOK, K_TOP), generator=gen, dtype=torch.int64)
    return (x.to(device), ids.to(device))


def reference(x, topk_ids):
    t, k = topk_ids.shape
    e = int(topk_ids.max().item()) + 1  # host sync — one of the things the agent should remove
    parts = []
    for g in range(e):
        pos = (topk_ids == g).nonzero(as_tuple=False)  # (n, 2), row-major order = stable
        if pos.numel():
            parts.append(pos[:, 0] * k + pos[:, 1])
    if parts:
        perm = torch.cat(parts)
    else:
        perm = torch.empty(0, dtype=torch.int64, device=topk_ids.device)
    row_ids = torch.div(perm, k, rounding_mode="floor")
    permuted_x = x[row_ids]
    counts = torch.bincount(topk_ids.reshape(-1), minlength=e)
    return permuted_x, perm, counts