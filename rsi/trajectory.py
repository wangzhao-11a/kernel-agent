"""Execution trajectories: the structured record of one agent run, plus replay.

Build:  python -m rsi.trajectory build runs/agent/<run_dir>
Replay: python -m rsi.trajectory replay runs/agent/<run_dir> --device npu

The agent loop writes one line per attempt into <run_dir>/attempts.jsonl; this module
merges attempts + meta.json into <run_dir>/trajectory.json. `replay` re-evaluates every
recorded solution through the harness, so a trajectory can be verified for
reproducibility before it is allowed to update the strategy pool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        try:
            return json.load(fh)
        except ValueError:
            return {}


def read_jsonl(path: str) -> list:
    rows = []
    if not os.path.isfile(path):
        return rows
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


def write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source(path: str, cap: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[:cap]
    except OSError:
        return ""


def _attempts_from_results(run_dir: str) -> list:
    """Fallback for runs that predate attempts.jsonl: rebuild from result_*.json."""
    results = []
    for name in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        if not name.startswith("result_") or not name.endswith(".json"):
            continue
        row = read_json(os.path.join(run_dir, name))
        if not row:
            continue
        results.append({
            "iter": row.get("iter"),
            "solution": row.get("solution"),
            "status": row.get("status"),
            "speedup": (row.get("timing") or {}).get("speedup"),
            "error": row.get("error"),
            "strategies_declared": [],
            "expected_speedup": None,
        })
    results.sort(key=lambda r: (r.get("iter") is None, r.get("iter") or 0))
    return results


def build_trajectory(run_dir: str, hardware: str = "") -> dict:
    meta = read_json(os.path.join(run_dir, "meta.json"))
    attempts = read_jsonl(os.path.join(run_dir, "attempts.jsonl"))
    if not attempts:
        attempts = _attempts_from_results(run_dir)
    for attempt in attempts:
        solution = attempt.get("solution") or ""
        if solution and os.path.isfile(solution) and not attempt.get("solution_source"):
            attempt["solution_source"] = _source(solution)
    prev = read_json(os.path.join(run_dir, "trajectory.json"))
    task_file = meta.get("task_file") or ""
    if not task_file:
        for attempt in attempts:
            if attempt.get("task_file"):
                task_file = attempt["task_file"]
                break
    traj = {
        "run_dir": os.path.abspath(run_dir),
        "task": meta.get("task") or (attempts[0].get("task") if attempts else ""),
        "task_file": task_file,
        "hardware": hardware or meta.get("hardware") or "",
        "device": meta.get("device", ""),
        "features": meta.get("features") or {},
        "best": meta.get("best"),
        "usage": meta.get("usage", {}),
        "attempts": attempts,
        "distilled": bool(prev.get("distilled")),
        "replay": prev.get("replay"),
        "built_at": _now(),
    }
    with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as fh:
        json.dump(traj, fh, ensure_ascii=False, indent=2)
    return traj


def replay_trajectory(run_dir: str, device: str = "auto", timeout: float = 600.0,
                      results_file: str = None) -> dict:
    from harness.runner import run_evaluation

    traj = read_json(os.path.join(run_dir, "trajectory.json")) or build_trajectory(run_dir)
    replay_dir = os.path.join(run_dir, "replay")
    results = []
    matches = 0
    if not traj.get("task_file"):
        replay = {"at": _now(), "device": device, "matches": 0, "n": 0, "results": [],
                  "error": "trajectory has no task_file (re-run the agent loop to regenerate it)"}
        traj["replay"] = replay
        with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as fh:
            json.dump(traj, fh, ensure_ascii=False, indent=2)
        return replay
    for attempt in traj.get("attempts", []):
        solution = attempt.get("solution")
        if not solution or not os.path.isfile(solution):
            results.append({"iter": attempt.get("iter"), "replay_status": "missing_solution", "match": False})
            continue
        res = run_evaluation(
            task=traj.get("task_file"), solution=solution, device=device, timeout=timeout,
            out_dir=replay_dir, results_file=results_file, quiet=True,
        )
        same = res.get("status") == attempt.get("status")
        matches += 1 if same else 0
        results.append({
            "iter": attempt.get("iter"),
            "recorded_status": attempt.get("status"),
            "replay_status": res.get("status"),
            "recorded_speedup": attempt.get("speedup"),
            "replay_speedup": (res.get("timing") or {}).get("speedup"),
            "match": same,
        })
    replay = {"at": _now(), "device": device, "matches": matches, "n": len(results), "results": results}
    traj["replay"] = replay
    with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as fh:
        json.dump(traj, fh, ensure_ascii=False, indent=2)
    return replay


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.trajectory", description="Build / replay execution trajectories.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("run_dir")
    b.add_argument("--hardware", default="")
    r = sub.add_parser("replay")
    r.add_argument("run_dir")
    r.add_argument("--device", default="auto")
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--results-file", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        traj = build_trajectory(args.run_dir, hardware=args.hardware)
        print("trajectory: %s" % os.path.join(args.run_dir, "trajectory.json"))
        print("task: %s | attempts: %d | best: %s" % (traj["task"], len(traj["attempts"]), traj.get("best")))
        return 0

    replay = replay_trajectory(args.run_dir, device=args.device, timeout=args.timeout,
                               results_file=args.results_file)
    print("replay: %d/%d attempts reproduced the recorded status (device=%s)"
          % (replay["matches"], replay["n"], replay["device"]))
    for row in replay["results"]:
        print("  iter=%-3s recorded=%-16s replay=%-16s match=%s" % (
            row.get("iter"), row.get("recorded_status"), row.get("replay_status"), row.get("match")))
    return 0 if replay["matches"] == replay["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())