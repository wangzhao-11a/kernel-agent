"""Run one (task, solution) evaluation in an isolated subprocess with a timeout.

This is THE interface any agent (this project's minimal loop, OpenHands, Claude Agent
SDK, ...) should use:

    python -m harness.runner --task tasks/level1/vector_add.py --solution sol.py

Exit code 0 when the solution passes, 1 otherwise. Every run is appended to
runs/results.jsonl unless --no-append is given.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(ROOT, "runs", "results.jsonl")

from .worker import format_summary  # noqa: E402  (kept after ROOT for clarity)


def _tail(text, n: int = 1200) -> str:
    text = text or ""
    return text[-n:] if len(text) > n else text


def run_evaluation(
    task: str,
    solution: str,
    device: str = "auto",
    timeout: float = 600,
    warmup=None,
    rep=None,
    trials=None,
    baseline=None,
    flush_l2=None,
    fail_on_suspicious: bool = False,
    policy_exempt: bool = False,
    ignore_require: bool = False,
    require: str = None,
    run_id: str = "",
    iteration: int = 0,
    out_dir: str = None,
    results_file: str = None,
    append: bool = True,
    quiet: bool = False,
) -> dict:
    out_dir = out_dir or os.path.join(ROOT, "runs", "single")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_path = os.path.join(out_dir, "result_%s.json" % stamp)

    cmd = [
        sys.executable, "-m", "harness.worker",
        "--task", task, "--solution", solution, "--out", out_path,
        "--device", device, "--run-id", run_id, "--iter", str(iteration),
    ]
    if warmup is not None:
        cmd += ["--warmup", str(int(warmup))]
    if rep is not None:
        cmd += ["--rep", str(int(rep))]
    if trials is not None:
        cmd += ["--trials", str(int(trials))]
    if baseline:
        cmd += ["--baseline", baseline]
    if flush_l2 is True:
        cmd.append("--flush-l2")
    elif flush_l2 is False:
        cmd.append("--no-flush-l2")
    if fail_on_suspicious:
        cmd.append("--fail-on-suspicious")
    if policy_exempt:
        cmd.append("--policy-exempt")
    if ignore_require:
        cmd.append("--ignore-require")
    if require:
        cmd += ["--require", require]

    t0 = time.time()
    stdout, stderr, rc = "", "", 0
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        stdout, stderr, rc = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        result = {
            "task_file": os.path.abspath(task),
            "solution": os.path.abspath(solution),
            "status": "timeout",
            "error": "evaluation exceeded %ss timeout" % timeout,
            "device": device,
            "wall_time_s": round(time.time() - t0, 2),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": run_id,
            "iter": iteration,
            "worker_stdout_tail": _tail(stdout, 800),
            "worker_stderr_tail": _tail(stderr),
        }
        return _postprocess(result, append, results_file, quiet)

    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
        except (OSError, ValueError) as exc:
            result = {
                "task_file": os.path.abspath(task),
                "solution": os.path.abspath(solution),
                "status": "crash",
                "error": "worker result unreadable: %s" % exc,
                "device": device,
                "wall_time_s": round(time.time() - t0, 2),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "run_id": run_id,
                "iter": iteration,
            }
    else:
        result = {
            "task_file": os.path.abspath(task),
            "solution": os.path.abspath(solution),
            "status": "crash",
            "error": "worker exited with code %s without writing a result" % rc,
            "device": device,
            "wall_time_s": round(time.time() - t0, 2),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": run_id,
            "iter": iteration,
            "worker_stderr_tail": _tail(stderr),
        }

    result["worker_stdout_tail"] = _tail(stdout, 800)
    return _postprocess(result, append, results_file, quiet)


def _postprocess(result: dict, append: bool, results_file: str, quiet: bool) -> dict:
    if not quiet:
        print(format_summary(result))
    if append:
        path = results_file or DEFAULT_RESULTS
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        except OSError as exc:
            print("WARNING: could not append to %s: %s" % (path, exc))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.runner", description="Evaluate one solution (isolated subprocess + timeout).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | npu")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--baseline", default=None, help="eager | compile")
    ap.add_argument("--flush-l2", dest="flush_l2", action="store_true", default=None)
    ap.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    ap.add_argument("--fail-on-suspicious", action="store_true")
    ap.add_argument("--policy-exempt", action="store_true",
                    help="legacy/reference/replay run: skip strict task policy checks")
    ap.add_argument("--ignore-require", action="store_true")
    ap.add_argument("--require", default=None, help="run-level requirements, e.g. 'triton'")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    result = run_evaluation(
        task=args.task, solution=args.solution, device=args.device, timeout=args.timeout,
        warmup=args.warmup, rep=args.rep, trials=args.trials, baseline=args.baseline,
        flush_l2=args.flush_l2, fail_on_suspicious=args.fail_on_suspicious,
        policy_exempt=args.policy_exempt,
        ignore_require=args.ignore_require,
        require=args.require,
        run_id=args.run_id, iteration=args.iter, out_dir=args.out_dir,
        results_file=args.results_file, append=not args.no_append, quiet=args.quiet,
    )
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())