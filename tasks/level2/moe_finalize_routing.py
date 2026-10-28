"""Level 2 — MoE finalize routing (Ascend MoeFinalizeRouting shape).

Combine top-k expert outputs per token with routing weights and add the selected
experts' bias: out[t] = sum_k w[t,k]*expert_out[t,k] + bias[expert_ids[t,k]].

Optimization targets: one fused gather + FMA + reduction pass instead of a per-k
python loop, cheap index path, fp32 accumulation.
"""
import torch

T_TOK, K_TOP, HID, N_EXPERTS = 512, 8, 512, 64

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
    "name": "moe_finalize_routing",
    "level": 2,
    "description": (
        "MoE finalize routing: expert_out (512,8,512), routing_weights (512,8) summing to 1, "
        "expert_ids int64 (512,8) in [0,64), expert_bias (64,512). "
        "out[t] = sum_k routing_weights[t,k]*expert_out[t,k] + expert_bias[expert_ids[t,k]]; out (512,512) fp32."
    ),
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 3,
    "warmup": 10,
    "rep": 50,
    "entrypoint": "run",
    "baseline": "eager",
    "expected_speedup": 1.8,
    "tags": ["moe", "routing", "fusion", "memory-bound", "fp32"],
    "op_class": "moe",
    "dtypes": ["float32"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(555 + trial)
    expert_out = torch.randn(T_TOK, K_TOP, HID, generator=gen, dtype=torch.float32)
    weights = torch.rand(T_TOK, K_TOP, generator=gen, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = torch.stack([torch.randperm(N_EXPERTS, generator=gen)[:K_TOP] for _ in range(T_TOK)]).to(torch.int64)
    bias = torch.randn(N_EXPERTS, HID, generator=gen, dtype=torch.float32) * 0.1
    return (expert_out.to(device), weights.to(device), ids.to(device), bias.to(device))


def reference(expert_out, routing_weights, expert_ids, expert_bias):
    scaled = expert_out * routing_weights.unsqueeze(-1)
    selected_bias = expert_bias[expert_ids]
    return (scaled + selected_bias).sum(dim=1)