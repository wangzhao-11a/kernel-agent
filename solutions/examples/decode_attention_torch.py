"""Starting-point solution for decode_attention (torch; the agent must beat it on Ascend)."""
import math

import torch


def run(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # (B,H,1,S)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)