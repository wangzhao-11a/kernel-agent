"""Causal-attribution / calibration checks for Phase 2.

1. bootstrap CI: clear separation is significant, identical distributions are not.
2. calibration: a persistently over-optimistic strategy ranks below a calibrated one.
3. hitchhiker: old evidence logic counts both attempts; the ablation gate skips
   the non-significant one and only promotes the controlled win.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness.stats import bootstrap_ratio_ci, ci_excludes_one  # noqa: E402
from rsi.distill import distill_run  # noqa: E402
from rsi.retrieve import _quality  # noqa: E402
from rsi.store import StrategyStore  # noqa: E402

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))
        traceback.print_exc()
        return
    RESULTS["pass"].append(name)
    print("PASS  %s" % name)


def check_bootstrap():
    control = [2.0] * 20
    candidate = [1.0] * 20
    ci = bootstrap_ratio_ci(control, candidate)
    assert ci and ci_excludes_one(ci), ci
    same = bootstrap_ratio_ci([1.0] * 20, [1.0] * 20)
    assert same and not ci_excludes_one(same), same


def check_calibration(tmp):
    store = StrategyStore(os.path.join(tmp, "calib"))
    optimistic = store.find_by_key("block_tiling")
    calibrated = store.find_by_key("multi_core_split")
    store.record_evidence(optimistic["id"], "t/a", "hitch", 1.0, "success",
                          expected_speedup=3.0, promote=False)
    store.record_evidence(calibrated["id"], "t/b", "hitch", 1.0, "success",
                          expected_speedup=1.0, promote=False)
    assert float(optimistic["stats"]["calibration_bias"]) > 0, optimistic["stats"]
    assert float(calibrated["stats"]["calibration_bias"]) == 0.0, calibrated["stats"]
    assert _quality(optimistic) < _quality(calibrated), (
        _quality(optimistic), _quality(calibrated))


def check_hitchhiker(tmp):
    run_dir = os.path.join(tmp, "hitch_run")
    os.makedirs(run_dir, exist_ok=True)
    attempts = [
        {
            "schema_version": 1, "iter": 1, "status": "pass", "speedup": 1.0,
            "strategies_declared": ["block tiling"], "solution_source": "",
            "ablation": {"control_speedup": 1.0, "ci95": [0.95, 1.02],
                         "significant": False},
            "evidence_eligible": False, "promote": False, "distilled": False,
        },
        {
            "schema_version": 1, "iter": 2, "status": "pass", "speedup": 1.25,
            "strategies_declared": ["block tiling"], "solution_source": "",
            "ablation": {"control_speedup": 1.0, "ci95": [1.08, 1.31],
                         "significant": True},
            "evidence_eligible": True, "promote": True, "distilled": False,
        },
    ]
    traj = {
        "task": "hitch", "hardware": "ascend-910b", "device": "npu",
        "features": {"tags": ["elementwise"], "op_class": "elementwise",
                     "dtypes": ["float32"], "name": "hitch"},
        "attempts": attempts, "distilled": False,
    }
    with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as fh:
        json.dump(traj, fh, ensure_ascii=False)

    # Old logic: every passing attempt records evidence and can promote.
    old = StrategyStore(os.path.join(tmp, "old_knowledge"))
    st_old = old.find_by_key("block_tiling")
    for attempt in attempts:
        old.record_evidence(st_old["id"], "t/%d" % attempt["iter"], "hitch",
                            attempt["speedup"], "success", expected_speedup=1.0,
                            promote=True)
    assert st_old["stats"]["applied"] == 2, st_old["stats"]
    assert st_old["stats"]["calibration_bias"] is not None, st_old["stats"]

    new = StrategyStore(os.path.join(tmp, "new_knowledge"))
    report = distill_run(run_dir, store=new)
    st_new = new.find_by_key("block_tiling")
    assert st_new["stats"]["applied"] == 1, st_new["stats"]
    assert any(row.get("iter") == 1 for row in report["skipped"]), report["skipped"]


def main():
    tmp = tempfile.mkdtemp(prefix="kernel-agent-causality-")
    print("temp dir: %s" % tmp)
    check("causal: bootstrap CI separates real gains from noise", check_bootstrap)
    check("causal: over-optimistic strategy loses retrieval weight",
          lambda: check_calibration(tmp))
    check("causal: hitchhiker is skipped by the ablation gate",
          lambda: check_hitchhiker(tmp))
    print("\n==================================================================")
    print("passed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    print("note: this exercises the gate logic on CPU; hardware conclusions still need device runs")
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())