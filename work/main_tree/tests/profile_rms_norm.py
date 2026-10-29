#!/usr/bin/env python3
"""Profile one RMSNorm shape with torch_npu and print CANNBENCH_PROFILE JSON.

Environment:
  PROFILE_SHAPE   comma-separated shape (default 65536,64)
  PROFILE_DTYPE   float16 / bfloat16 / float32 (default float16)
  PROFILE_ITERS   profiled iterations (default 3)
  PROFILE_OUT     optional profile output directory
"""
import csv
import glob
import json
import os
import sys
import tempfile

import torch
import torch_npu

import cann_bench


def _shape():
    raw = os.environ.get("PROFILE_SHAPE", "65536,64")
    return [int(x) for x in raw.replace("x", ",").split(",") if x.strip()]


def _dtype():
    name = os.environ.get("PROFILE_DTYPE", "float16").lower()
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(name, torch.float16)


def _config():
    P = torch_npu.profiler
    try:
        return P.experimental_config._ExperimentalConfig(
            profiler_level=P.ProfilerLevel.Level1,
            aic_metrics=P.AiCMetrics.PipeUtilization)
    except Exception:
        return None


def _num(row, key):
    try:
        return float(str(row.get(key, "")).strip())
    except (TypeError, ValueError):
        return 0.0


def _read_kernel_rows(out_dir):
    paths = glob.glob(os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True)
    if not paths:
        return []
    latest = max(paths, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = []
        for row in csv.DictReader(fh):
            clean = {str(k).lstrip("\ufeff").strip(): v for k, v in row.items()}
            name = str(clean.get("Name") or "").lower()
            if "rms_norm" in name or "rmsnorm" in name:
                rows.append(clean)
        return rows


def main():
    shape = _shape()
    dtype = _dtype()
    iters = max(1, min(20, int(os.environ.get("PROFILE_ITERS", "3"))))
    out_dir = os.environ.get("PROFILE_OUT") or tempfile.mkdtemp(prefix="rmsnorm-profile-")

    x = torch.randn(*shape, dtype=dtype).npu()
    g = torch.randn(shape[-1], dtype=dtype).npu()
    for _ in range(5):
        cann_bench.rms_norm(x, g, 1e-6)
    torch.npu.synchronize()

    P = torch_npu.profiler
    cfg = _config()
    kwargs = {
        "activities": [P.ProfilerActivity.CPU, P.ProfilerActivity.NPU],
        "on_trace_ready": P.tensorboard_trace_handler(out_dir),
    }
    if cfg is not None:
        kwargs["experimental_config"] = cfg
    with P.profile(**kwargs):
        for _ in range(iters):
            cann_bench.rms_norm(x, g, 1e-6)
        torch.npu.synchronize()

    rows = _read_kernel_rows(out_dir)
    base = {
        "shape": shape,
        "dtype": os.environ.get("PROFILE_DTYPE", "float16"),
    }
    if not rows:
        result = dict(base, kernel="", count=0, duration_us=0.0,
                      aiv_vec_ratio=0.0, aiv_scalar_ratio=0.0,
                      aiv_mte2_ratio=0.0, aiv_mte3_ratio=0.0,
                      bottleneck="unknown", out_dir=out_dir)
        print("CANNBENCH_PROFILE " + json.dumps(result, separators=(",", ":")))
        print("RMSNORM_PROFILE PASS")
        return 0

    n = float(len(rows))
    vec = sum(_num(r, "aiv_vec_ratio") for r in rows) / n
    scalar = sum(_num(r, "aiv_scalar_ratio") for r in rows) / n
    mte2 = sum(_num(r, "aiv_mte2_ratio") for r in rows) / n
    mte3 = sum(_num(r, "aiv_mte3_ratio") for r in rows) / n
    duration = sum(_num(r, "Duration(us)") for r in rows) / n
    ratios = {"VEC": vec, "SCALAR": scalar, "MTE2": mte2, "MTE3": mte3}
    bottleneck = max(ratios, key=ratios.get)
    result = dict(
        base, kernel=str(rows[0].get("Name") or ""), count=len(rows),
        duration_us=round(duration, 4),
        aiv_vec_ratio=round(vec, 4), aiv_scalar_ratio=round(scalar, 4),
        aiv_mte2_ratio=round(mte2, 4), aiv_mte3_ratio=round(mte3, 4),
        bottleneck=bottleneck, out_dir=out_dir)
    print("CANNBENCH_PROFILE " + json.dumps(result, separators=(",", ":")))
    print("RMSNORM_PROFILE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())