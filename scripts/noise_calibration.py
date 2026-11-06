"""Calibrate the per-device noise amplitude epsilon and validate the G1 gate.

    python scripts/noise_calibration.py --device cpu --runs 20 --validate-runs 20
    python scripts/noise_calibration.py --device npu --runs 20 --validate-runs 20

Runs an honest solution whose work equals the reference (tests/adversarial/honest.py)
against tasks/level1/softmax.py N times.  epsilon is the 95th percentile of
|speedup - 1| over those runs; it is written to configs/noise.json.  A second,
independent batch is classified with that epsilon and reports how many honest
runs would be counted as "success" (G1 requires <= 1 out of 20).

Every number comes from an actual run on the given device; this script never
estimates or fills in a device it did not execute on.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness import device as dev  # noqa: E402
from harness import noise as noise_mod  # noqa: E402
from harness import stats as stats_mod  # noqa: E402
from harness.runner import run_evaluation  # noqa: E402

DEFAULT_TASK = os.path.join(ROOT, "tasks", "level1", "softmax.py")
DEFAULT_SOLUTION = os.path.join(ROOT, "tests", "adversarial", "honest.py")


def _env(device: str) -> dict:
    t = dev.torch_module()
    env = {
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "torch": getattr(t, "__version__", None),
    }
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401

            env["torch_npu"] = getattr(torch_npu, "__version__", None)
            env["device_name"] = t.npu.get_device_name(0)
            env["device_count"] = t.npu.device_count()
        except Exception as exc:
            env["npu_error"] = "%s: %s" % (type(exc).__name__, exc)
    return env


def _write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def run_batch(args, count: int, label: str) -> list:
    rows = []
    tmp = tempfile.mkdtemp(prefix="kernel_agent_noise_")
    for i in range(int(count)):
        res = run_evaluation(
            task=args.task, solution=args.solution, device=args.device,
            timeout=args.timeout, warmup=args.warmup, rep=args.rep, rounds=args.rounds,
            out_dir=os.path.join(tmp, "out"), results_file=os.path.join(tmp, "results.jsonl"),
            append=False, quiet=True,
        )
        timing = res.get("timing") or {}
        row = {
            "status": res.get("status"),
            "speedup": float(timing.get("speedup") or 0.0),
            "speedup_ci": timing.get("speedup_ci"),
            "stability_delta": timing.get("stability_delta"),
            "epsilon_at_run": timing.get("epsilon"),
            "error": res.get("error"),
        }
        rows.append(row)
        print("  %s %2d/%d: status=%s speedup=%.4f ci=%s stability=%s" % (
            label, i + 1, count, row["status"], row["speedup"],
            row["speedup_ci"], row["stability_delta"]), flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="noise_calibration",
                                 description="Measure device epsilon and validate the success gate.")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | npu")
    ap.add_argument("--runs", type=int, default=20, help="calibration runs")
    ap.add_argument("--validate-runs", type=int, default=20, help="independent validation runs")
    ap.add_argument("--max-success", type=int, default=1,
                    help="G1: allowed success count in the validation batch (default 1)")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--solution", default=DEFAULT_SOLUTION)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default=noise_mod.DEFAULT_PATH, help="config path to write")
    args = ap.parse_args(argv)

    device = dev.resolve_device(args.device)
    key = noise_mod.device_key(device=device)
    if not key:
        print("ERROR: cannot map device %r to a noise key" % device)
        return 2
    print("device=%s key=%s task=%s solution=%s" % (
        device, key, os.path.relpath(args.task, ROOT), os.path.relpath(args.solution, ROOT)))

    print("\n== calibration ==")
    calibration = run_batch(args, args.runs, "cal")
    ok = [r for r in calibration if r["status"] in ("pass", "pass_unstable")]
    if not ok:
        print("ERROR: no passing calibration runs; nothing to calibrate")
        return 2
    deviations = [abs(r["speedup"] - 1.0) for r in ok]
    epsilon = round(float(stats_mod.percentile(deviations, 95)), 4)

    data = noise_mod.load_noise(args.out)
    entry = {
        "epsilon": epsilon,
        "method": "p95(|speedup-1|) over honest runs whose work equals the reference",
        "n_runs": len(ok),
        "task": os.path.relpath(args.task, ROOT).replace("\\", "/"),
        "solution": os.path.relpath(args.solution, ROOT).replace("\\", "/"),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "rep": args.rep,
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": _env(device),
        "speedups": [round(r["speedup"], 4) for r in ok],
        "speedup_ci": [r["speedup_ci"] for r in ok],
        "statuses": [r["status"] for r in ok],
    }
    data.setdefault("schema_version", 1)
    data.setdefault("devices", {})[key] = entry
    # Write the calibration before validation so the workers actually apply the
    # pass_unstable gate during the validation batch.
    _write(args.out, data)

    print("\n== validation (epsilon=%.4f) ==" % epsilon)
    validation = run_batch(args, args.validate_runs, "val")
    classified = []
    for row in validation:
        if row["status"] == "pass_unstable":
            outcome = "unstable"  # correct, but not evidence: never a success
        elif row["status"] != "pass":
            outcome = "not_pass"
        else:
            outcome = noise_mod.classify(row["speedup_ci"], epsilon)
        classified.append(outcome)
    successes = [i for i, outcome in enumerate(classified) if outcome == "success"]
    entry["validation"] = {
        "n_runs": len(validation),
        "epsilon": epsilon,
        "outcomes": classified,
        "success_count": len(successes),
        "success_indices": successes,
        "max_success": args.max_success,
        "passed": len(successes) <= args.max_success,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _write(args.out, data)
    print("\nwrote %s" % args.out)
    print("epsilon(%s)=%.4f | validation: %d/%d classified success (allowed <= %d)" % (
        key, epsilon, len(successes), len(validation), args.max_success))
    print("validation outcomes: %s" % ", ".join(classified))
    return 0 if len(successes) <= args.max_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
