"""Parse Nsight Compute CSV exports into a compact, agent-friendly summary.

Produce the CSV:

    ncu --csv --target-processes all -f -o /tmp/prof \\
        python -m harness.profile_target --task tasks/level1/softmax.py --solution sol.py --device cuda
    ncu --import /tmp/prof.ncu-rep --csv > /tmp/prof.csv

Then:  python -m harness.profilers.ncu --file /tmp/prof.csv
"""
from __future__ import annotations

import argparse
import csv
import json

KEY_METRICS = (
    "Duration", "Elapsed Cycles", "SM Frequency",
    "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput",
    "L1/TEX Hit Rate", "L2 Hit Rate", "Achieved Occupancy",
    "Registers Per Thread", "Waves Per SM", "Executed Ipc Active",
)


def _to_float(value):
    v = (value or "").strip().replace(",", "")
    if v.endswith("%"):
        v = v[:-1].strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    return ordered[n // 2]


def summarize_csv(path: str) -> dict:
    order = []
    metrics = {}  # kernel name -> metric name -> [(unit, value)]

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        idx = None
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            if row[0].startswith("=="):
                continue
            if idx is None:
                if "Metric Name" in row and "Metric Value" in row:
                    idx = {name: i for i, name in enumerate(row)}
                continue

            def cell(name):
                i = idx.get(name)
                return row[i].strip() if i is not None and i < len(row) else ""

            kernel = cell("Kernel Name") or cell("Function Name") or "kernel"
            metric = cell("Metric Name")
            if not metric:
                continue
            value = _to_float(cell("Metric Value"))
            if value is None:
                continue
            if kernel not in metrics:
                metrics[kernel] = {}
                order.append(kernel)
            metrics[kernel].setdefault(metric, []).append((cell("Metric Unit"), value))

    kernels = []
    for name in order:
        agg = {}
        for metric, samples in metrics[name].items():
            unit = samples[0][0]
            agg[metric] = {"value": _median([s[1] for s in samples]), "unit": unit, "n": len(samples)}
        kernels.append({"name": name, "metrics": agg})

    primary = None
    if kernels:
        with_duration = [k for k in kernels if k["metrics"].get("Duration", {}).get("value") is not None]
        primary = max(with_duration, key=lambda k: k["metrics"]["Duration"]["value"]) if with_duration else kernels[0]
    return {"n_kernels": len(kernels), "kernels": kernels, "primary": primary}


def format_summary(summary: dict) -> str:
    if not summary.get("primary"):
        return "no kernel metrics found (is this an `ncu --csv` export?)"
    lines = ["kernels profiled: %d" % summary.get("n_kernels", 0)]
    primary = summary["primary"]
    lines.append("primary kernel: %s" % primary["name"])
    shown = 0
    for metric in KEY_METRICS:
        m = primary["metrics"].get(metric)
        if m is None or m["value"] is None:
            continue
        lines.append("  %s: %s %s" % (metric, ("%.4g" % m["value"]), m["unit"]))
        shown += 1
        if shown >= 10:
            break
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.profilers.ncu", description="Summarize an ncu --csv export.")
    ap.add_argument("--file", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    summary = summarize_csv(args.file)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary))
    return 0 if summary.get("primary") else 1


if __name__ == "__main__":
    raise SystemExit(main())