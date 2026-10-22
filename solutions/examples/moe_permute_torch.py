"""Starting-point solution for moe_permute (single pass; the agent must beat this on Ascend)."""
import torch


def run(x, topk_ids):
    t, k = topk_ids.shape
    e = int(topk_ids.max()) + 1
    flat = topk_ids.reshape(-1)
    perm = torch.argsort(flat, stable=True)
    row_ids = torch.div(perm, k, rounding_mode="floor")
    permuted_x = x[row_ids]
    counts = torch.bincount(flat, minlength=e)
    return permuted_x, perm, counts