"""Torch-profiler based op-level profiling (CPU / CUDA / NPU) for agent feedback.

    python -m harness.profilers.torch_prof --task tasks/level1/vector_add.py \\
        --solution sol.py --device cpu --out prof.json

This is the cross-platform profiling backend used by the agent loop (`--profiler torch`).
On Ascend switch to `--profiler msprof` for hardware counters; on NVIDIA `--profiler ncu`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness import device as dev  # noqa: E402
from harness.tasks import load_task  # noqa: E402
from harness.worker import _wrap_inference, import_solution  # noqa: E402


def _activities(device: str) -> list:
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    kind = dev.backend_of(device)
    if kind == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif kind == "npu":
        npu_activity = getattr(torch.profiler.ProfilerActivity, "NPU", None)
        if npu_activity is not None:
            activities.append(npu_activity)
    return activities


def _sort_key(device: str) -> str:
    return "self_device_time_total" if dev.is_accelerator(device) else "self_cpu_time_total"


def profile_fn(fn, inputs, device: str, iters: int = 3, top: int = 10) -> dict:
    import torch

    with torch.profiler.profile(activities=_activities(device), record_shapes=False) as prof:
        for _ in range(max(1, int(iters))):
            fn(*inputs)
        dev.sync(device)

    rows = []
    for evt in prof.key_averages():
        self_device = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        self_cpu = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
        total = self_device if self_device > 0 else self_cpu
        if total <= 0:
            continue
        rows.append({
            "op": evt.key,
            "count": int(evt.count),
            "self_us": round(total, 2),
            "self_device_us": round(self_device, 2),
            "self_cpu_us": round(self_cpu, 2),
        })
    rows.sort(key=lambda r: -r["self_us"])
    total_us = sum(r["self_us"] for r in rows) or 1.0
    for row in rows[:top]:
        row["pct"] = round(100.0 * row["self_us"] / total_us, 1)

    table = ""
    try:
        table = prof.key_averages().table(sort_by=_sort_key(device), row_limit=top)
    except Exception:
        try:
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top)
        except Exception:
            table = ""

    return {
        "backend": "torch",
        "device": device,
        "iters": int(iters),
        "total_us": round(total_us, 2),
        "n_ops": len(rows),
        "top_ops": rows[:top],
        "table": table,
    }


def format_summary(summary: dict, top: int = 8) -> str:
    if not summary.get("top_ops"):
        return "profiling produced no ops (did the solution run torch ops in that process?)"
    lines = ["profiling backend=%s device=%s (%s iters, self time aggregated; total %.2f ms)"
             % (summary.get("backend"), summary.get("device"), summary.get("iters"),
                summary.get("total_us", 0.0) / 1000.0)]
    lines.append("top ops by self time:")
    for row in summary["top_ops"][:top]:
        lines.append("  %-44s %5.1f%%  %9.1f us  x%d"
                     % (row["op"][:44], row.get("pct", 0.0), row["self_us"], row["count"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.profilers.torch_prof",
                                 description="Profile a solution with torch.profiler and write a JSON summary.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--case", default="", help="shape_set case name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = dev.resolve_device(args.device)
    task = load_task(args.task)
    solution = import_solution(args.solution)
    entry = getattr(solution, task.entrypoint)
    run = _wrap_inference(entry)
    if args.case:
        cases = {c["name"]: c for c in task.cases()}
        if args.case not in cases:
            raise SystemExit("unknown case %r (choose %s)" % (args.case, ", ".join(cases) or "none"))
        inputs = task.case_inputs(cases[args.case], device, args.trial)
    else:
        inputs = task.make_inputs(device, args.trial)
    for _ in range(2):  # warmup outside the profiled region
        run(*inputs)
    dev.sync(device)

    summary = profile_fn(run, inputs, device, iters=args.iters, top=args.top)
    summary["task"] = task.name
    summary["solution"] = os.path.abspath(args.solution)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(format_summary(summary, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())