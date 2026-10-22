"""Starting-point solution for sparse_flash_attention (torch; the agent must beat this on Ascend)."""
import math

import torch


def run(q, k, v, idx):
    scale = 1.0 / math.sqrt(q.shape[-1])
    b, h, s, d = q.shape
    topk = idx.shape[-1]
    idx_expand = idx.unsqueeze(-1).expand(b, h, s, topk, d)
    k_sel = torch.gather(k.unsqueeze(2).expand(b, h, s, k.shape[2], d), 3, idx_expand)
    v_sel = torch.gather(v.unsqueeze(2).expand(b, h, s, v.shape[2], d), 3, idx_expand)
    scores = torch.einsum("bhsd,bhskd->bhsk", q, k_sel) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhsk,bhskd->bhsd", probs, v_sel)