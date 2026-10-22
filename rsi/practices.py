"""Seed practice library: canonical, hardware-aware tuning practices.

Each practice carries:
  statement        model-facing description of the practice
  tags/op_class    used for condition matching against a task
  keywords         used to attribute a solution to this practice (fallback signal)
  expected_speedup rough prior; the store recalibrates it from measured evidence

The strategy store bootstraps from this list; from then on it evolves with evidence.
"""

PRACTICES = [
    {
        "key": "online_softmax",
        "statement": "Fuse max-subtraction, exp and sum into one pass with a running accumulator (online softmax) instead of materializing the score matrix.",
        "tags": ["attention", "softmax", "reduction", "fusion", "memory-bound"],
        "op_class": ["attention", "reduction"],
        "keywords": ["online softmax", "running max", "rescale", "single pass", "flash", "avoid materializ"],
        "expected_speedup": 1.35,
    },
    {
        "key": "block_tiling",
        "statement": "Tile the operator so each block's working set fits on-chip memory (UB/L1/shared) and loop over tiles instead of touching the whole tensor at once.",
        "tags": ["attention", "gemm", "tiling", "memory-bound", "compute-bound"],
        "op_class": ["attention", "gemm", "elementwise", "reduction", "moe"],
        "keywords": ["tile", "block size", "on-chip", "shared memory", "ub buffer", "chunk"],
        "expected_speedup": 1.3,
    },
    {
        "key": "multi_core_split",
        "statement": "Split work across all cores with balanced, aligned block sizes; avoid idle cores caused by a heavy tail block.",
        "tags": ["parallelism", "load-balance"],
        "op_class": ["attention", "gemm", "elementwise", "reduction", "moe"],
        "keywords": ["multi-core", "block_dim", "core", "grid", "load balance", "tail", "split"],
        "expected_speedup": 1.25,
    },
    {
        "key": "double_buffer_pipeline",
        "statement": "Overlap data movement with compute via double buffering / software pipelining instead of serial copy-then-compute.",
        "tags": ["pipeline", "performance"],
        "op_class": ["attention", "gemm", "elementwise", "reduction", "moe"],
        "keywords": ["double buffer", "pipel", "overlap", "async copy", "prefetch"],
        "expected_speedup": 1.2,
    },
    {
        "key": "vectorized_access",
        "statement": "Make global memory access contiguous, aligned and wide (vectorized loads/stores); avoid strided or unaligned access.",
        "tags": ["memory", "bandwidth", "memory-bound"],
        "op_class": ["elementwise", "reduction", "attention"],
        "keywords": ["vectoriz", "align", "contiguous", "coalesc", "128", "mask"],
        "expected_speedup": 1.15,
    },
    {
        "key": "fused_epilogue",
        "statement": "Fold bias / scaling / activation into the producer's epilogue so the intermediate tensor is never written back to global memory.",
        "tags": ["fusion", "gemm", "moe", "memory-bound"],
        "op_class": ["gemm", "moe", "elementwise"],
        "keywords": ["epilogue", "fuse", "fusion", "bias", "activation"],
        "expected_speedup": 1.25,
    },
    {
        "key": "avoid_host_sync",
        "statement": "Remove device-to-host synchronization from the hot path (.item(), bool(tensor), .cpu()); keep the dataflow device-resident.",
        "tags": ["sync", "latency"],
        "op_class": ["moe", "gemm", "attention"],
        "keywords": ["synchronize", "item()", ".cpu()", "host sync", "non_blocking"],
        "expected_speedup": 1.2,
    },
    {
        "key": "mixed_precision_accum",
        "statement": "Run the heavy math in fp16/bf16 but accumulate in fp32 to keep accuracy while gaining throughput.",
        "tags": ["precision", "throughput", "compute-bound"],
        "op_class": ["gemm", "attention", "moe"],
        "keywords": ["fp32 accum", "accumul", "cast", "fp16", "bf16", "half"],
        "expected_speedup": 1.4,
    },
    {
        "key": "mla_absorption",
        "statement": "For MLA, absorb the K up-projection into the query projection and the V up-projection into the output projection, so k_nope/v are never materialized; compute scores against the latent c_kv.",
        "tags": ["attention", "mla", "fusion", "compute-bound"],
        "op_class": ["attention"],
        "keywords": ["absorb", "latent", "c_kv", "up-projection", "w_uk", "w_uv"],
        "expected_speedup": 1.3,
    },
    {
        "key": "grouped_gemm_single_launch",
        "statement": "For MoE/grouped workloads use one grouped GEMM launch over all experts instead of per-expert python loops (removes launch overhead and enables balanced tiling).",
        "tags": ["moe", "gemm", "grouped", "launch"],
        "op_class": ["moe", "gemm"],
        "keywords": ["grouped", "group_list", "per-expert", "single launch", "batched gemm", "bmm"],
        "expected_speedup": 1.6,
    },
    {
        "key": "permute_once",
        "statement": "In MoE, permute/sort tokens once by expert, run contiguous computation, then invert the permutation once — instead of gather/scatter per token.",
        "tags": ["moe", "gather", "layout"],
        "op_class": ["moe"],
        "keywords": ["permute", "sort", "gather", "scatter", "reorder", "unsort"],
        "expected_speedup": 1.3,
    },
    {
        "key": "split_k",
        "statement": "For tall-skinny GEMMs, split the K dimension across cores and reduce partials instead of wasting cores on a thin tiling.",
        "tags": ["gemm", "parallelism", "compute-bound"],
        "op_class": ["gemm", "attention"],
        "keywords": ["split-k", "split k", "partial", "atomic", "reduce partial"],
        "expected_speedup": 1.2,
    },
    {
        "key": "sparse_skip_blocks",
        "statement": "In sparse attention, skip fully-masked/empty blocks with an early-out instead of multiplying them by zero; keep sparse index math packed and cheap.",
        "tags": ["attention", "sparse", "skip"],
        "op_class": ["attention"],
        "keywords": ["skip", "mask", "sparse", "empty block", "early out"],
        "expected_speedup": 1.5,
    },
    {
        "key": "reduce_launch_overhead",
        "statement": "Merge chains of tiny operators into fewer kernels; on accelerators per-op launch overhead often dominates for small tensors.",
        "tags": ["fusion", "launch", "latency"],
        "op_class": ["elementwise", "reduction", "moe"],
        "keywords": ["launch", "merge", "fewer kernel", "overhead", "tiny"],
        "expected_speedup": 1.3,
    },
]


def practice_by_key(key: str):
    for p in PRACTICES:
        if p["key"] == key:
            return p
    return None