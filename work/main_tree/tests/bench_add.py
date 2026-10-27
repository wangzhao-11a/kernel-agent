#!/usr/bin/env python3
"""Device benchmark: CannBench AscendC add vs the PyTorch NPU baseline."""
import json
import sys

import torch
import torch_npu

import cann_bench

CASES = [
    ((1024,), torch.float32),
    ((10000,), torch.float16),
    ((256, 512), torch.float16),
    ((1000, 1000), torch.float32),
    ((4, 3, 64, 64), torch.float16),
    ((1000003,), torch.float32),
    ((256, 512), torch.int32),
]

WARMUP = 5
REPS = 20


def baseline(x, y):
    return x + y


def time_ms(fn, x, y):
    for _ in range(WARMUP):
        fn(x, y)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fn(x, y)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPS


def main():
    rows = []
    base_total = 0.0
    cand_total = 0.0
    ok_all = True
    for shape, dtype in CASES:
        if dtype == torch.int32:
            x = torch.randint(-1000, 1000, shape, dtype=dtype)
            y = torch.randint(-1000, 1000, shape, dtype=dtype)
            gold = x + y
        else:
            x = torch.randn(*shape, dtype=dtype)
            y = torch.randn(*shape, dtype=dtype)
            gold = x + y
        xn, yn = x.npu(), y.npu()
        got = cann_bench.add(xn, yn)
        ok = tuple(got.shape) == tuple(gold.shape) and got.dtype == dtype
        if ok and dtype != torch.int32:
            a = got.cpu().to(torch.float32).reshape(-1)
            g = gold.to(torch.float32).reshape(-1)
            rel = (a - g).abs() / (g.abs() + 1e-7)
            ok = float(rel.mean()) < 2 ** -10 and float(rel.max()) < 10 * 2 ** -10
        elif ok:
            ok = torch.equal(got.cpu(), gold)
        base_ms = time_ms(baseline, xn, yn)
        cand_ms = time_ms(lambda a, b: cann_bench.add(a, b), xn, yn)
        speedup = base_ms / cand_ms if cand_ms > 0 else 0.0
        base_total += base_ms
        cand_total += cand_ms
        ok_all = ok_all and ok
        rows.append({
            "shape": list(shape), "dtype": str(dtype).split(".")[-1], "dim": 0,
            "baseline_ms": round(base_ms, 6), "candidate_ms": round(cand_ms, 6),
            "speedup": round(speedup, 4), "correct": bool(ok),
        })
        print("%-22s %-9s base=%8.4fms cand=%8.4fms speedup=%6.3fx %s" % (
            shape, str(dtype).split(".")[-1], base_ms, cand_ms, speedup,
            "OK" if ok else "BAD"))

    speedup = (base_total / cand_total) if cand_total > 0 else 0.0
    print("ADD_BENCH", "PASS" if ok_all else "FAIL")
    print("CANNBENCH_METRICS " + json.dumps({
        "speedup": round(speedup, 6),
        "baseline_ms": round(base_total, 6),
        "candidate_ms": round(cand_total, 6),
        "metric_key": "speedup",
        "cases": rows,
    }, separators=(",", ":")))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())