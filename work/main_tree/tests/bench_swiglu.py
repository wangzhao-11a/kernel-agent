#!/usr/bin/env python3
"""SwiGlu device benchmark: CannBench AscendC kernel vs the PyTorch NPU baseline.

The official test only proves correctness.  This script gives the agent a
performance signal in the same units as `metric_key: speedup`:

    speedup = geometric mean of per-case (baseline_ms / candidate_ms)
    total_speedup = sum(baseline_ms) / sum(candidate_ms)

The primary metric is the geometric mean so one huge shape cannot dominate.
Both paths allocate their output on every call, so the comparison includes the
same allocator behaviour.  Timings use NPU events and a fixed warmup; each case
is measured RUNS times and the median is reported (BENCH_RUNS, default 3).
"""
import json
import math
import os
import statistics
import sys

import torch
import torch_npu

import cann_bench

CASES = [
    # aligned tile path
    ((1024, 2048), torch.float16, -1),
    ((2048, 4096), torch.float32, -1),
    ((4096, 8192), torch.bfloat16, -1),
    ((512, 4096), torch.float32, -1),
    ((8192, 1024), torch.float16, -1),
    ((256, 8192), torch.bfloat16, -1),
    # non-last dims / 4D / 5D
    ((1022, 2047), torch.bfloat16, 0),
    ((2049, 1024), torch.float16, 1),
    ((11, 13, 16, 67), torch.float32, 2),
    ((3, 4, 7, 1024), torch.bfloat16, 1),
    ((2, 3, 5, 7, 64), torch.float16, -1),
    ((3, 2, 8, 11, 128), torch.bfloat16, 2),
    # prime / odd / large-outer
    ((1000003, 2), torch.bfloat16, 1),
    ((999983, 4), torch.float16, 1),
    ((363, 367, 14), torch.bfloat16, -1),
    ((17, 257, 1022), torch.float32, -1),
    ((65537, 4), torch.float32, -1),
    ((100000, 8), torch.bfloat16, 1),
    ((50000, 16), torch.float16, 1),
    ((32768, 32), torch.bfloat16, 1),
    ((262144, 6), torch.float16, 0),
    ((4097, 1022), torch.float16, 1),
]

WARMUP = 5
REPS = 20
try:
    RUNS = max(1, min(9, int(os.environ.get("BENCH_RUNS", "3"))))
except ValueError:
    RUNS = 3


def baseline(x, dim):
    x0, x1 = x.chunk(2, dim=dim)
    return torch.nn.functional.silu(x0) * x1


def ref(x, dim):
    """Golden in fp32, matching tests/test_swiglu_official.py."""
    xf = x.to(torch.float32)
    x0, x1 = xf.chunk(2, dim=dim)
    return (torch.nn.functional.silu(x0) * x1).to(x.dtype)


THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}


def time_ms(fn, x, dim):
    for _ in range(WARMUP):
        fn(x, dim)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fn(x, dim)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPS


def median_ms(fn, x, dim):
    # Median of RUNS independent full measurements; this suppresses the
    # run-to-run frequency/noise swings seen on the shared NPU.
    return statistics.median(time_ms(fn, x, dim) for _ in range(RUNS))


def main():
    rows = []
    base_total = 0.0
    cand_total = 0.0
    ok_all = True
    for shape, dtype, dim in CASES:
        x = (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dtype).npu()
        gold = ref(x, dim)
        got = cann_bench.swi_glu(x, dim)
        ok = (tuple(got.shape) == tuple(gold.shape)) and (got.dtype == dtype)
        if ok:
            a = got.cpu().to(torch.float32).reshape(-1)
            g = gold.cpu().to(torch.float32).reshape(-1)
            rel = (a - g).abs() / (g.abs() + 1e-7)
            thr = THR[dtype]
            ok = float(rel.mean()) < thr and float(rel.max()) < 10 * thr
        base_ms = median_ms(baseline, x, dim)
        cand_ms = median_ms(lambda t, d: cann_bench.swi_glu(t, d), x, dim)
        speedup = base_ms / cand_ms if cand_ms > 0 else 0.0
        base_total += base_ms
        cand_total += cand_ms
        ok_all = ok_all and ok
        rows.append({
            "shape": list(shape), "dtype": str(dtype).split(".")[-1], "dim": dim,
            "baseline_ms": round(base_ms, 6), "candidate_ms": round(cand_ms, 6),
            "speedup": round(speedup, 4), "correct": bool(ok),
        })
        print("%-22s %-9s dim=%-3d base=%8.4fms cand=%8.4fms speedup=%6.3fx %s" % (
            shape, str(dtype).split(".")[-1], dim, base_ms, cand_ms, speedup,
            "OK" if ok else "BAD"))

    total_speedup = (base_total / cand_total) if cand_total > 0 else 0.0
    speedups = [row["speedup"] for row in rows]
    geomean = math.exp(sum(math.log(max(1e-6, s)) for s in speedups) / len(speedups)) if speedups else 0.0
    worst = min(speedups) if speedups else 0.0
    print("SWIGLU_BENCH", "PASS" if ok_all else "FAIL")
    print("CANNBENCH_METRICS " + json.dumps({
        # primary metric: balanced across cases, not dominated by one large shape
        "speedup": round(geomean, 6),
        "total_speedup": round(total_speedup, 6),
        "worst_speedup": round(worst, 6),
        "baseline_ms": round(base_total, 6),
        "candidate_ms": round(cand_total, 6),
        "metric_key": "speedup",
        "runs": RUNS,
        "cases": rows,
    }, separators=(",", ":")))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())