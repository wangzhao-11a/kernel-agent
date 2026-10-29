#!/usr/bin/env python3
"""RMSNorm device benchmark: CannBench kernel vs a PyTorch NPU baseline."""
import json
import math
import os
import statistics
import sys

import torch
import torch_npu

import cann_bench

CASES = [
    ((1, 128), torch.float32, 1e-6),
    ((17, 257), torch.float32, 1e-5),
    ((1024, 1024), torch.float16, 1e-6),
    ((2048, 4096), torch.float32, 1e-6),
    ((4096, 1024), torch.float16, 1e-6),
    ((512, 8192), torch.bfloat16, 1e-6),
    ((3, 5, 127), torch.float16, 1e-6),
    ((2, 3, 5, 257), torch.bfloat16, 1e-6),
    ((65536, 64), torch.float16, 1e-6),
    ((100000, 128), torch.bfloat16, 1e-6),
    ((33, 4095), torch.float32, 1e-6),
    ((7, 8191), torch.float16, 1e-6),
    ((1024, 8192), torch.bfloat16, 1e-6),
    ((8, 16384), torch.float32, 1e-6),
]

WARMUP = 5
REPS = 20
try:
    RUNS = max(1, min(9, int(os.environ.get("BENCH_RUNS", "3"))))
except ValueError:
    RUNS = 3

THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}


def baseline(x, g, eps):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * g


def ref(x, g, eps):
    xf = x.to(torch.float32)
    gf = g.to(torch.float32)
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gf).to(x.dtype)


def time_ms(fn, x, g, eps):
    for _ in range(WARMUP):
        fn(x, g, eps)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fn(x, g, eps)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPS


def median_ms(fn, x, g, eps):
    return statistics.median(time_ms(fn, x, g, eps) for _ in range(RUNS))


def main():
    ok_all = True
    rows = []
    base_total = cand_total = 0.0
    for shape, dtype, eps in CASES:
        x = torch.randn(*shape, dtype=dtype).npu()
        g = torch.randn(shape[-1], dtype=dtype).npu()
        got = cann_bench.rms_norm(x, g, eps)
        gold = ref(x.cpu(), g.cpu(), eps)
        ok = tuple(got.shape) == tuple(gold.shape) and got.dtype == dtype
        if ok:
            a = got.cpu().to(torch.float32).reshape(-1)
            b = gold.to(torch.float32).reshape(-1)
            rel = (a - b).abs() / (b.abs() + 1e-7)
            thr = THR[dtype]
            ok = float(rel.mean()) < thr and float(rel.max()) < 10 * thr
        base_ms = median_ms(baseline, x, g, eps)
        cand_ms = median_ms(cann_bench.rms_norm, x, g, eps)
        speedup = base_ms / cand_ms if cand_ms > 0 else 0.0
        base_total += base_ms
        cand_total += cand_ms
        ok_all = ok_all and ok
        rows.append({
            "shape": list(shape), "dtype": str(dtype).split(".")[-1], "dim": -1,
            "baseline_ms": round(base_ms, 6), "candidate_ms": round(cand_ms, 6),
            "speedup": round(speedup, 4), "correct": bool(ok),
        })
        print("%-22s %-9s base=%8.4fms cand=%8.4fms speedup=%6.3fx %s" % (
            shape, str(dtype).split(".")[-1], base_ms, cand_ms, speedup,
            "OK" if ok else "BAD"))
    total_speedup = base_total / cand_total if cand_total > 0 else 0.0
    speedups = [row["speedup"] for row in rows]
    geomean = math.exp(sum(math.log(max(1e-6, s)) for s in speedups) / len(speedups)) if speedups else 0.0
    worst = min(speedups) if speedups else 0.0
    print("RMSNORM_BENCH", "PASS" if ok_all else "FAIL")
    print("CANNBENCH_METRICS " + json.dumps({
        "speedup": round(geomean, 6),
        "total_speedup": round(total_speedup, 6),
        "worst_speedup": round(worst, 6),
        "baseline_ms": round(base_total, 6),
        "candidate_ms": round(cand_total, 6),
        "metric_key": "speedup",
        "runs": RUNS, "cases": rows,
    }, separators=(",", ":")))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())