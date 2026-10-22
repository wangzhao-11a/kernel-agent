"""Level 2 — MLA (multi-head latent attention, DeepSeek-V2/V3 style); Ascend 910B target.

The K/V are decoded from a shared latent c_kv; the known strong optimization is the
"absorption" trick (fold W_uk into the query projection and W_uv into the output
projection so k_nope/v are never materialized). The reference uses the explicit
(non-absorbed) form; finding absorption is part of the task.

Optimization targets: absorption, fused projections, tiling the H x T x T score path,
fp32 accumulation.
"""
import math

import torch

T, HID, HEADS = 128, 512, 8
DN, DR, DKV, DV = 64, 32, 96, 64

TASK = {
    "name": "mla_attention",
    "level": 2,
    "description": (
        "MLA (single call, no KV cache): q_nope/q_pe from x; k_nope/v decoded from a shared "
        "latent c_kv = x @ W_dkv; k_pe = x @ W_kr. "
        "scores = (q_nope @ k_nope^T + q_pe @ k_pe^T) / sqrt(DN+DR); out = softmax(scores) @ v, "
        "per head. x=(128,512) fp32, weights supplied; out shape (128,8,64)."
    ),
    "tolerance": {"atol": 2e-4, "rtol": 2e-4},
    "num_trials": 2,
    "warmup": 5,
    "rep": 20,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.3,
    "tags": ["attention", "mla", "compute-bound", "fusion", "fp32"],
    "op_class": "attention",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(77 + trial)

    def rand(shape, scale=0.05):
        return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(device)

    x = rand((T, HID), scale=1.0)
    w_q_nope = rand((HID, HEADS * DN))
    w_q_pe = rand((HID, HEADS * DR))
    w_dkv = rand((HID, DKV))
    w_kr = rand((HID, DR))
    w_uk = rand((DKV, HEADS * DN))
    w_uv = rand((DKV, HEADS * DV))
    return (x, w_q_nope, w_q_pe, w_dkv, w_kr, w_uk, w_uv)


def reference(x, w_q_nope, w_q_pe, w_dkv, w_kr, w_uk, w_uv):
    q_nope = (x @ w_q_nope).view(T, HEADS, DN)
    q_pe = (x @ w_q_pe).view(T, HEADS, DR)
    c_kv = x @ w_dkv
    k_pe = x @ w_kr
    k_nope = (c_kv @ w_uk).view(T, HEADS, DN)
    v = (c_kv @ w_uv).view(T, HEADS, DV)
    scores = torch.einsum("ihd,jhd->hij", q_nope, k_nope) + torch.einsum("ihd,jd->hij", q_pe, k_pe)
    probs = torch.softmax(scores * (1.0 / math.sqrt(DN + DR)), dim=-1)
    return torch.einsum("hij,jhd->ihd", probs, v)