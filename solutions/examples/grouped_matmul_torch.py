"""Starting-point solution for grouped_matmul (torch; the agent must beat this on Ascend)."""
import torch


def run(a, b, group_list):
    sizes = [int(s) for s in group_list.tolist()]
    outs = []
    start = 0
    for g, size in enumerate(sizes):
        if size:
            outs.append(a[start:start + size] @ b[g])
        start += size
    if not outs:
        return a.new_zeros((0, b.shape[-1]))
    return torch.cat(outs, dim=0)