"""Self-test: weighted shape aggregation and the worst-shape regression gate."""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness.runner import run_evaluation  # noqa: E402
from harness.stats import bootstrap_weighted_ratio  # noqa: E402
from harness.worker import format_summary  # noqa: E402

TASK = '''
TASK = {
    "schema_version": 1,
    "name": "shape_probe",
    "level": 1,
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 1,
    "warmup": 5,
    "rep": 20,
    "baseline": "eager",
    "shape_gate_min_speedup": %s,
    "shape_set": [
        {"name": "small", "shape": [1024], "dtype": "float32", "weight": 0.5,
         "heldout": False, "input_specs": [{"kind": "randn", "shape": [1024], "dtype": "float32"}]},
        {"name": "large", "shape": [8192], "dtype": "float32", "weight": 0.5,
         "heldout": False, "input_specs": [{"kind": "randn", "shape": [8192], "dtype": "float32"}]},
    ],
}

def reference(x):
    return x * 2
'''

GOOD = "def run(x):\n    return x * 2\n"
SLOW = """
import time

def run(x):
    if x.numel() > 4096:
        time.sleep(0.02)
    return x * 2
"""


def check_stats():
    ci = bootstrap_weighted_ratio([
        {"weight": 1, "numerator": [10.0] * 20, "denominator": [5.0] * 20},
        {"weight": 3, "numerator": [10.0] * 20, "denominator": [10.0] * 20},
    ])
    assert ci == [1.25, 1.25], ci
    assert bootstrap_weighted_ratio([]) is None


def evaluate(tmp, name, gate, source):
    task = os.path.join(tmp, name + "_task.py")
    sol = os.path.join(tmp, name + "_sol.py")
    with open(task, "w", encoding="utf-8") as fh:
        fh.write(TASK % gate)
    with open(sol, "w", encoding="utf-8") as fh:
        fh.write(source)
    return run_evaluation(task=task, solution=sol, device="cpu", warmup=5, rep=20,
                          trials=1, timeout=180, out_dir=os.path.join(tmp, name + "_out"),
                          quiet=True)


def check_good(tmp):
    result = evaluate(tmp, "good", "0.1", GOOD)
    assert result["status"] == "pass", (result.get("status"), result.get("error"))
    timing = result["timing"]
    assert len(timing["shape_cases"]) == 2, timing
    assert timing["shape_gate"]["passed"] is True, timing["shape_gate"]
    assert timing["speedup"] == timing["weighted_speedup"], timing
    assert timing["worst_case"]["speedup"] >= 0.1, timing["worst_case"]
    summary = format_summary(result)
    assert "shapes: 2 cases" in summary and "weighted=" in summary, summary


def check_regression(tmp):
    result = evaluate(tmp, "slow", "0.9", SLOW)
    assert result["status"] == "fail_shape_gate", (result.get("status"), result.get("error"))
    assert "worst shape large" in (result.get("error") or ""), result.get("error")
    gate = result["timing"]["shape_gate"]
    assert gate["passed"] is False and gate["worst_case"] == "large", gate
    assert gate["worst_speedup"] < 0.9, gate


def main():
    check_stats()
    tmp = tempfile.mkdtemp(prefix="ka_shape_")
    check_good(tmp)
    check_regression(tmp)
    print("passed: 3 | failed: 0")


if __name__ == "__main__":
    main()