"""Starting-point solution for moe_finalize_routing (torch; the agent must beat this on Ascend).

Per-k python loop: correct but K separate passes over the big (T,K,H) tensor — exactly the
kind of eager pattern the task exists to eliminate.
"""
import torch


def run(expert_out, routing_weights, expert_ids, expert_bias):
    out = torch.zeros(expert_out.shape[0], expert_out.shape[-1],
                      dtype=expert_out.dtype, device=expert_out.device)
    for k in range(expert_out.shape[1]):
        bias_k = expert_bias[expert_ids[:, k]]
        out = out + expert_out[:, k] * routing_weights[:, k:k + 1] + bias_k
    return out