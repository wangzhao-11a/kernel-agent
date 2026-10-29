"""Roofline-style utilization from a workload's bytes/flops and elapsed time.

The harness only uses this when a task declares a workload (or the caller passes
one).  Peak values are optional and come from the task metadata or the
KERNEL_AGENT_PEAK_BW_GBPS / KERNEL_AGENT_PEAK_TFLOPS environment variables.
Without peaks the result still reports achieved bandwidth/throughput.
"""
from __future__ import annotations

import os


def _peak(meta, key, env):
    value = (meta or {}).get(key)
    if value in (None, ""):
        value = os.environ.get(env)
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def estimate(workload: dict, elapsed_ms: float, meta: dict | None = None) -> dict:
    workload = workload or {}
    try:
        nbytes = float(workload.get("bytes") or 0.0)
        flops = float(workload.get("flops") or 0.0)
        ms = float(elapsed_ms or 0.0)
    except (TypeError, ValueError):
        return {}
    if ms <= 0 or (nbytes <= 0 and flops <= 0):
        return {}
    seconds = ms / 1000.0
    bw = nbytes / seconds / 1e9 if nbytes > 0 else 0.0
    tflops = flops / seconds / 1e12 if flops > 0 else 0.0
    peak_bw = _peak(meta, "peak_bw_gbps", "KERNEL_AGENT_PEAK_BW_GBPS")
    peak_tf = _peak(meta, "peak_tflops", "KERNEL_AGENT_PEAK_TFLOPS")
    util_bw = (bw / peak_bw) if peak_bw else None
    util_tf = (tflops / peak_tf) if peak_tf else None
    if util_bw is None and util_tf is None:
        bottleneck = "unknown"
    elif nbytes <= 0:
        bottleneck = "compute"
    elif flops <= 0:
        bottleneck = "memory"
    else:
        bottleneck = "memory" if (util_bw or 0.0) >= (util_tf or 0.0) else "compute"
    return {
        "achieved_bw_gbps": round(bw, 4),
        "achieved_tflops": round(tflops, 6),
        "peak_bw_gbps": peak_bw,
        "peak_tflops": peak_tf,
        "bw_utilization": round(util_bw, 4) if util_bw is not None else None,
        "flops_utilization": round(util_tf, 4) if util_tf is not None else None,
        "bottleneck": bottleneck,
        "elapsed_ms": round(ms, 6),
    }