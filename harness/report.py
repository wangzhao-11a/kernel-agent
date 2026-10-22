"""Summarize results.jsonl: pass@1, speedup table, fast_p (geometric mean).

    python -m harness.report runs/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict


def load_results(paths) -> list:
    rows = []
    for path in paths:
        if not os.path.isfile(path):
            print("WARNING: results file not found: %s" % path, file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def latest_per_task(rows: list) -> OrderedDict:
    latest = OrderedDict()
    for row in rows:
        name = row.get("task") or os.path.basename(str(row.get("task_file", "unknown")))
        latest[name] = row
    return latest


def summarize(rows: list) -> dict:
    latest = latest_per_task(rows)
    passed = [r for r in latest.values() if r.get("status") == "pass"]
    speedups = []
    for r in passed:
        t = r.get("timing") or {}
        s = t.get("speedup")
        if isinstance(s, (int, float)) and s > 0:
            speedups.append(float(s))
    fast_p = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else None
    median = None
    if speedups:
        ordered = sorted(speedups)
        median = ordered[len(ordered) // 2]
    return {
        "tasks": len(latest),
        "passed": len(passed),
        "pass_rate": (len(passed) / len(latest)) if latest else 0.0,
        "fast_p": fast_p,
        "median_speedup": median,
        "latest": latest,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.report", description="Summarize evaluation runs.")
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    paths = args.paths or [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "results.jsonl")]
    rows = load_results(paths)
    if not rows:
        print("no results yet in %s" % ", ".join(paths))
        return 1
    s = summarize(rows)

    if args.json:
        print(json.dumps({
            "tasks": s["tasks"], "passed": s["passed"], "pass_rate": s["pass_rate"],
            "fast_p": s["fast_p"], "median_speedup": s["median_speedup"],
            "per_task": {
                name: {
                    "status": r.get("status"),
                    "speedup": (r.get("timing") or {}).get("speedup"),
                    "solution": os.path.basename(str(r.get("solution", ""))),
                    "timestamp": r.get("timestamp"),
                }
                for name, r in s["latest"].items()
            },
        }, ensure_ascii=False, indent=2))
        return 0

    print("%-24s %-16s %10s %12s" % ("task", "status", "speedup", "solution"))
    print("-" * 68)
    for name, r in s["latest"].items():
        speedup = (r.get("timing") or {}).get("speedup")
        speed = "%.3fx" % speedup if isinstance(speedup, (int, float)) else "-"
        print("%-24s %-16s %10s %12s" % (name, r.get("status", "?"), speed, os.path.basename(str(r.get("solution", "")))))
    print("-" * 68)
    fast = "%.3fx" % s["fast_p"] if s["fast_p"] else "-"
    med = "%.3fx" % s["median_speedup"] if s["median_speedup"] else "-"
    print("pass@1: %d/%d (%.0f%%) | fast_p (geomean over passing): %s | median speedup: %s"
          % (s["passed"], s["tasks"], 100.0 * s["pass_rate"], fast, med))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())