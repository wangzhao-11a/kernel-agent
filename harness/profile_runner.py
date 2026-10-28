"""Run a profiling pass over a solution and return a compact summary.

Backends:
  torch  — torch.profiler op-level self time; works on CPU / CUDA / NPU (default)
  msprof — Ascend hardware profiling via the msprof CLI (parsed by profilers.msprof)
  ncu    — NVIDIA Nsight Compute via the ncu CLI (parsed by profilers.ncu)

Used by the agent loop to feed profiling back into the conversation, and available
standalone:

    python -m harness.profile_runner --task tasks/level2/mla_attention.py \\
        --solution sol.py --device npu --backend msprof
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.profilers import msprof as msprof_mod  # noqa: E402
from harness.profilers import ncu as ncu_mod  # noqa: E402
from harness.profilers import torch_prof as torch_prof_mod  # noqa: E402

BACKENDS = ("torch", "msprof", "ncu")


def profile(task: str, solution: str, device: str = "auto", backend: str = "torch",
            out_dir: str = None, timeout: float = 900.0, iters: int = 3, top: int = 10,
            trial: int = 0, case: str = "") -> dict:
    out_dir = out_dir or os.path.join(ROOT, "runs", "profiling")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result = {"status": "error", "backend": backend, "device": device,
              "error": None, "summary": None, "text": None, "out_dir": None}

    if backend == "torch":
        out_path = os.path.join(out_dir, "torch_%s.json" % stamp)
        cmd = [sys.executable, "-m", "harness.profilers.torch_prof",
               "--task", task, "--solution", solution, "--device", device,
               "--iters", str(int(iters)), "--top", str(int(top)), "--out", out_path,
               "--trial", str(int(trial))]
        if case:
            cmd += ["--case", str(case)]
        proc = _run(cmd, timeout)
        if proc is None:
            result["error"] = "profiling timed out after %ss" % timeout
            return result
        if os.path.isfile(out_path):
            with open(out_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            result["summary"] = summary
            result["text"] = torch_prof_mod.format_summary(summary, top=top)
            result["out_dir"] = out_path
            result["status"] = "ok" if summary.get("top_ops") else "empty"
            if result["status"] == "empty":
                result["error"] = "no ops recorded (solution may not run torch ops in that process)"
            return result
        result["error"] = "profiler worker exited rc=%s without output: %s" % (proc.returncode, _tail(proc.stderr))
        return result

    if backend == "msprof":
        target = os.path.join(out_dir, "msprof_%s" % stamp)
        os.makedirs(target, exist_ok=True)
        application = "%s -m harness.profile_target --task %s --solution %s --device %s --iters %d" % (
            sys.executable, task, solution, device, max(1, int(iters)))
        cmd = ["msprof", "--output=%s" % target, "--type=text", "--summary-format=csv",
               "--export=on", "--application=%s" % application]
        try:
            proc = _run(cmd, timeout)
        except FileNotFoundError:
            result["error"] = "msprof not found in PATH (source the CANN environment first)"
            return result
        if proc is None:
            result["error"] = "msprof timed out after %ss" % timeout
            return result
        summary = msprof_mod.summarize_dir(target)
        result.update(summary=summary, text=msprof_mod.format_summary(summary, top=top), out_dir=target)
        result["status"] = "ok" if summary.get("top_ops") else "empty"
        if result["status"] == "empty":
            result["error"] = "msprof produced no op data (rc=%s): %s" % (proc.returncode, _tail(proc.stdout + proc.stderr))
        return result

    if backend == "ncu":
        target = os.path.join(out_dir, "ncu_%s" % stamp)
        os.makedirs(target, exist_ok=True)
        report = os.path.join(target, "profile")
        cmd = ["ncu", "--csv", "--target-processes", "all", "--launch-count", str(max(1, int(iters))),
               "-f", "-o", report, sys.executable, "-m", "harness.profile_target",
               "--task", task, "--solution", solution, "--device", device, "--iters", "1"]
        try:
            proc = _run(cmd, timeout)
        except FileNotFoundError:
            result["error"] = "ncu not found in PATH"
            return result
        if proc is None:
            result["error"] = "ncu timed out after %ss" % timeout
            return result
        csv_path = os.path.join(target, "profile.csv")
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write(proc.stdout or "")
        summary = ncu_mod.summarize_csv(csv_path)
        result.update(summary=summary, text=ncu_mod.format_summary(summary), out_dir=csv_path)
        result["status"] = "ok" if summary.get("primary") else "empty"
        if result["status"] == "empty":
            result["error"] = "ncu produced no metrics (rc=%s): %s" % (proc.returncode, _tail(proc.stderr))
        return result

    result["error"] = "unknown backend %r (choose from %s)" % (backend, ", ".join(BACKENDS))
    return result


def _tail(text, n: int = 500) -> str:
    text = text or ""
    return text[-n:] if len(text) > n else text


def _run(cmd, timeout):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.profile_runner", description="Profile one solution (compact summary).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="torch", choices=list(BACKENDS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = profile(args.task, args.solution, device=args.device, backend=args.backend,
                     out_dir=args.out_dir, timeout=args.timeout, iters=args.iters, top=args.top)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("text") or ("profiling failed (%s): %s" % (result["status"], result["error"])))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())