"""Starting-point solution for grouped_gemm_swiglu (per-group loop; the agent must beat it)."""
import torch
import torch.nn.functional as F


def run(a, w1, w3, group_list):
    sizes = [int(s) for s in group_list.tolist()]
    outs = []
    start = 0
    for g, size in enumerate(sizes):
        if size:
            a_g = a[start:start + size]
            outs.append(F.silu(a_g @ w1[g]) * (a_g @ w3[g]))
        start += size
    if not outs:
        return a.new_zeros((0, w1.shape[-1]))
    return torch.cat(outs, dim=0)