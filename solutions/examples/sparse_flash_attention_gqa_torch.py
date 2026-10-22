"""Starting-point solution for the cann-bench SparseFlashAttention task (torch; beat it on Ascend).

Dense-mask route, mirroring the official golden (fine for the S1=1 decode shape): expand the KV
heads to N1 with repeat_interleave (n1 -> n1//G mapping), scatter the sparse indices into a bool
mask, then softmax over masked scores and multiply by the expanded V.
"""
import torch


def run(query, key, value, sparseIndices, scaleValue):
    b, s1, n1, dk = query.shape
    _, s2, n2, dv = key.shape
    groups = n1 // n2
    topk = sparseIndices.shape[-1]

    q = query.to(torch.float32)
    k = key.to(torch.float32).repeat_interleave(groups, dim=2)      # (B,S2,N1,Dk)
    v = value.to(torch.float32).repeat_interleave(groups, dim=2)    # (B,S2,N1,Dv)

    scores = torch.einsum("bsnd,btnd->bnst", q, k) * scaleValue      # (B,N1,S1,S2)
    idx = sparseIndices.to(torch.int64).repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
    mask = torch.zeros(b, n1, s1, s2, dtype=torch.bool, device=query.device)
    mask.scatter_(3, idx, True)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    y = torch.einsum("bnst,btnd->bsnd", probs, v)
    return y.to(query.dtype)