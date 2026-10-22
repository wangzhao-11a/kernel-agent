"""Parse msprof text exports into an op-duration summary (Ascend / CANN).

Produce the export:

    msprof --output=/tmp/prof --type=text --summary-format=csv --export=on \\
        --application="python -m harness.profile_target --task tasks/level1/softmax.py --solution sol.py --device npu"

Then:  python -m harness.profilers.msprof --dir /tmp/prof

The parser is defensive about column names: it looks for any name-like column
(Op Name / Kernel Name / ...) and any duration-like column (Task Duration(us) / ...).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

NAME_COLS = ("op name", "op_name", "opname", "kernel name", "kernel_name", "name")
DUR_COLS = ("task duration(us)", "task_duration(us)", "duration(us)", "total time(us)",
            "aicore_time(us)", "task duration", "duration")


def _norm(col: str) -> str:
    return col.strip().lower().replace(" ", "")


def _to_float(value):
    v = (value or "").strip().replace(",", "")
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _pick(header, candidates, fuzzy_keywords=()):
    normed = [_norm(h) for h in header]
    for cand in candidates:
        c = _norm(cand)
        if c in normed:
            return normed.index(c)
    for i, h in enumerate(normed):
        if any(k in h for k in fuzzy_keywords):
            return i
    return None


def summarize_dir(path: str) -> dict:
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))

    ops = {}
    used = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                rows = list(csv.reader(fh))
        except OSError:
            continue
        name_i = dur_i = None
        found_header = False
        for row in rows:
            if not row or all(not c.strip() for c in row):
                continue
            if not found_header:
                name_i = _pick(row, NAME_COLS, fuzzy_keywords=("opname", "kernelname"))
                dur_i = _pick(row, DUR_COLS, fuzzy_keywords=("duration", "time(us)"))
                if name_i is not None and dur_i is not None:
                    found_header = True
                else:
                    name_i = dur_i = None
                continue
            name = row[name_i].strip() if name_i < len(row) else ""
            value = _to_float(row[dur_i]) if dur_i < len(row) else None
            if not name or value is None:
                continue
            ops.setdefault(name, []).append(value)
        if found_header:
            used.append(file_path)

    top = []
    for name, values in ops.items():
        top.append({
            "name": name,
            "count": len(values),
            "total_us": round(sum(values), 2),
            "mean_us": round(sum(values) / len(values), 2),
        })
    top.sort(key=lambda item: -item["total_us"])
    return {
        "n_ops": len(top),
        "total_us": round(sum(item["total_us"] for item in top), 2),
        "top_ops": top[:30],
        "files_used": used,
    }


def format_summary(summary: dict, top: int = 8) -> str:
    if not summary.get("top_ops"):
        return "no op duration data found (did msprof export CSVs? check --type=text --summary-format=csv)"
    lines = ["files parsed: %d | ops: %d | total: %.1f us" % (len(summary.get("files_used", [])), summary.get("n_ops", 0), summary.get("total_us", 0.0))]
    lines.append("top ops by total time:")
    for op in summary["top_ops"][:top]:
        lines.append("  %-40s total=%9.1f us  count=%-5d mean=%8.1f us" % (op["name"][:40], op["total_us"], op["count"], op["mean_us"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.profilers.msprof", description="Summarize an msprof text export.")
    ap.add_argument("--dir", required=True, help="msprof output directory (or a single CSV)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    summary = summarize_dir(args.dir)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary, top=args.top))
    return 0 if summary.get("top_ops") else 1


if __name__ == "__main__":
    raise SystemExit(main())