"""Starting-point solution for mla_attention (torch; the agent must beat this on Ascend).

Uses the concat trick (q_cat = [q_nope | q_pe], k_cat = [k_nope | k_pe]) so the score path
is one batched matmul — but still materializes k_nope/v, so absorption is left as the
optimization for the agent.
"""
import math

import torch


def run(x, w_q_nope, w_q_pe, w_dkv, w_kr, w_uk, w_uv):
    heads, dn, dv = 8, 64, 64
    dr = w_kr.shape[-1]
    q_nope = (x @ w_q_nope).view(-1, heads, dn)
    q_pe = (x @ w_q_pe).view(-1, heads, dr)
    c_kv = x @ w_dkv
    k_pe = x @ w_kr
    k_nope = (c_kv @ w_uk).view(-1, heads, dn)
    v = (c_kv @ w_uv).view(-1, heads, dv)

    q_cat = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)                    # (H,T,dn+dr)
    k_cat = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, heads, -1)], dim=-1).transpose(0, 1)
    scores = (q_cat @ k_cat.transpose(-1, -2)) / math.sqrt(dn + dr)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hij,jhd->ihd", probs, v)