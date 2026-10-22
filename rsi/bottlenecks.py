"""Bottleneck -> action playbook.

Seeded from profiles gathered while developing this harness: when the profiler's
top ops match a trigger, the matching advice is appended to the feedback so the
agent knows what to do about it. Extend SEEDS with your own profile signatures.
"""
from __future__ import annotations

SEEDS = (
    {
        "match": ("_local_scalar_dense", ".item", "item()"),
        "advice": "Host sync dominates: remove device->host scalar reads from the hot path "
                  "(replace .max().item() with a fixed/device-side bound such as bincount(minlength=N)).",
    },
    {
        "match": ("empty_tensor", "aten::empty", "new_empty"),
        "advice": "Allocation overhead dominates: allocate output/intermediate buffers once and reuse "
                  "them instead of torch.empty*/new_* inside the hot path.",
    },
    {
        "match": ("scatter_add_", "aclnnScatterAdd"),
        "advice": "scatter_add is slow for this shape on Ascend: prefer permute-once + contiguous writes, "
                  "index_add into a preallocated buffer, or fold the accumulation into a matmul epilogue.",
    },
    {
        "match": ("aten::cat",),
        "advice": "cat materializes a new tensor: write into a preallocated output (two half-writes) "
                  "or keep the halves separate and avoid the concat.",
    },
    {
        "match": ("aten::index", "aten::gather", "aclnnIndex", "aclnnGather"),
        "advice": "Gather/index dominates: pack indices once (single int32 tensor), keep accesses contiguous, "
                  "or replace per-token gather with a permute-once layout.",
    },
    {
        "match": ("aclnnMax", "aten::max"),
        "advice": "Device max is cheap; only the host-side .item() on it is expensive — keep it on device.",
    },
    {
        "match": ("aten::repeat_interleave", "aten::expand", "aten::as_strided"),
        "advice": "View/expand ops are normally metadata-only; real self time means materialization is "
                  "happening — restructure to avoid it.",
    },
    {
        "match": ("silu", "aten::sigmoid", "aten::gelu"),
        "advice": "Activation shows up separately: fuse it into the producer's epilogue instead of a "
                  "second pass over the tensor.",
    },
    {
        "match": ("aten::bmm", "aten::mm", "aten::matmul", "aclnnBatchMatMul"),
        "advice": "Matmul dominates (compute-bound): try one wide GEMM + gather instead of many small GEMMs, "
                  "tiling/split-K, or fp16 inputs with fp32 accumulation.",
    },
    {
        "match": ("aten::sort", "aten::argsort"),
        "advice": "Sort cost is usually unavoidable for permutes, but make sure the sort key packs "
                  "(expert, index) into one integer so a single stable sort suffices.",
    },
)


def lookup(profile_summary: dict, top_n: int = 3) -> list:
    """Advice strings whose triggers match any of the profiler's top ops."""
    ops = [str(row.get("op", "")) for row in (profile_summary or {}).get("top_ops") or []][:top_n]
    out, seen = [], set()
    for seed in SEEDS:
        if any(trigger.lower() in op.lower() for op in ops for trigger in seed["match"]):
            if seed["advice"] not in seen:
                out.append(seed["advice"])
                seen.add(seed["advice"])
    return out


def render(advice: list) -> str:
    if not advice:
        return ""
    return "Bottleneck playbook (matched from the profile above):\n" + "\n".join(
        "  - " + item for item in advice)