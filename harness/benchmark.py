"""Timing: accelerator events when available, perf_counter otherwise.

Anti-noise measures: warmup, median over `rep` iterations, optional L2 flush between
timed iterations, and a second pass (run by the worker) to report stability.
"""
from __future__ import annotations

import statistics
import time

from . import device as dev
from .policy import policy_guard


def benchmark(fn, inputs, device: str, warmup: int = 25, rep: int = 100, flush_l2: bool = True,
              policy=None, policy_exempt: bool = False) -> dict:
    mod = dev.backend_module(device)
    buf = dev.flush_buffer(device) if flush_l2 else None

    def call():
        with policy_guard(policy, policy_exempt):
            return fn(*inputs)

    for _ in range(max(0, int(warmup))):
        call()
    dev.sync(device)

    if mod is not None:
        events = []
        for _ in range(int(rep)):
            if buf is not None:
                buf.zero_()
            start = mod.Event(enable_timing=True)
            end = mod.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            events.append((start, end))
        mod.synchronize()
        times = [float(s.elapsed_time(e)) for s, e in events]
    else:
        times = []
        for _ in range(int(rep)):
            t0 = time.perf_counter()
            call()
            times.append((time.perf_counter() - t0) * 1000.0)

    ordered = sorted(times)
    n = len(ordered)
    return {
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.fmean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "p10_ms": float(ordered[max(0, int(0.10 * (n - 1)))]),
        "std_ms": float(statistics.pstdev(ordered)) if n > 1 else 0.0,
        "reps": n,
        # Raw samples feed the bootstrap CI and ablation gate.  Keep them rounded
        # to keep result JSON readable.
        "samples_ms": [round(float(x), 6) for x in times],
    }