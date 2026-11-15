"""F5 self-test: the scheduler's dream pass respects a per-cycle budget.

Checks:
  1. with 10 verified records, one cycle replays only --dream-limit records
  2. four cycles (1 + 3 + 3 + 3) replay all 10 records at least once
  3. every replayed record reported the remaining count

The 10 records come from real `agent.loop --provider replay` runs; each cycle
is a real `python -m rsi.schedule` invocation (budget 0, holdout evaluation
skipped, so the only work in the cycle is the dream pass and the bookkeeping).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi.store import StrategyStore  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS.append(("FAIL", name, exc))
        print("FAIL  %s: %s" % (name, exc))
        import traceback
        traceback.print_exc()
        return
    RESULTS.append(("PASS", name, None))
    print("PASS  %s" % name)


def _write_task(tmp, name):
    src = '''import numpy as np

TASK = {
    "name": "%s",
    "level": 0,
    "tolerance": {"atol": 1e-6, "rtol": 1e-6},
    "num_trials": 1, "warmup": 5, "rep": 60,
    "tags": ["softmax", "reduction"],
}

def make_inputs(device, trial):
    rng = np.random.default_rng(7 + trial)
    return (rng.standard_normal((512, 512)).astype(np.float32) * 2.0,)

def reference(x):
    # fp64: a plain fp32 reference is only ~1.0x (recorded CI razor thin), and
    # an honest replay then lands outside it; fp64 makes the recorded margin
    # unambiguous on any CPU.
    y = x.astype(np.float64)
    e = np.exp(y - y.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)
''' % name
    path = os.path.join(tmp, name + ".py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path


GOOD = ("import numpy as np\n\n"
        "def run(x):\n"
        "    m = x.max(axis=-1, keepdims=True)\n"
        "    e = np.exp(x - m)\n"
        "    return e / e.sum(axis=-1, keepdims=True)\n")


def _make_run(tmp, task_path, run_name, runs_root, before):
    replay = os.path.join(tmp, run_name + ".json")
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": [
            {"action": "write_solution", "content": GOOD, "strategies": [],
             "expected_speedup": 1.0},
            {"action": "finish", "summary": run_name},
        ]}, fh)
    proc = subprocess.run([
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--out-dir", runs_root,
        "--knowledge-root", os.path.join(tmp, run_name + "_kb"),
        "--max-iters", "2", "--target-speedup", "0.5", "--timeout", "300",
        "--profiler", "off", "--no-distill",
    ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1000:]
    new = sorted(set(os.listdir(runs_root)) - before)
    assert len(new) == 1, new
    return os.path.join(runs_root, new[0])


def _run_cycle(tmp, index, knowledge_root, runs_root, extra):
    out = os.path.join(tmp, "cycles", "cycle_%d" % index)
    proc = subprocess.run([
        sys.executable, "-m", "rsi.schedule",
        "--budget-iters", "0", "--approve", "auto", "--provider", "replay",
        "--device", "cpu", "--knowledge-root", knowledge_root,
        "--runs-root", runs_root, "--out", out, "--skip-holdout-eval",
        "--versions-root", os.path.join(tmp, "versions"),
    ] + list(extra), cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)
    # No budget means no agent runs, so main() exits 1; the audit must exist.
    assert proc.returncode in (0, 1), proc.stdout[-1500:] + proc.stderr[-1000:]
    audits = sorted(glob.glob(os.path.join(out, "*", "audit.json")))
    assert audits, proc.stdout[-1000:]
    with open(audits[-1], encoding="utf-8") as fh:
        return json.load(fh)


def check_dream_budget(tmp):
    runs_root = os.path.join(tmp, "runs")
    os.makedirs(runs_root, exist_ok=True)
    run_dirs = []
    for index in range(10):
        task = _write_task(tmp, "budget_task_%d" % index)
        run_dirs.append(_make_run(tmp, task, "budget_run_%02d" % index, runs_root,
                                  set(os.listdir(runs_root))))

    knowledge_root = os.path.join(tmp, "kb")
    store = StrategyStore(knowledge_root)
    sids = [row["id"] for row in list(store.strategies.values())[:10]]
    assert len(sids) == 10, len(sids)
    for sid, run_dir in zip(sids, run_dirs):
        store.record_evidence(sid, os.path.basename(run_dir), "budget_task", 1.0, "neutral",
                              run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
        store.strategies[sid]["confidence"] = "verified"
        # A mature store: every record was last checked long ago.  Without this
        # all ten tie on the bootstrap timestamp and the first three would be
        # picked again by a stable sort.
        store.strategies[sid]["updated"] = "2026-01-01T00:00:00Z"
    store.save()

    reviewed = []
    demoted = 0
    expected_per_cycle = [1, 3, 3, 3]
    for index, expected in enumerate(expected_per_cycle, start=1):
        extra = ["--dream-limit", "1"] if index == 1 else []
        audit = _run_cycle(tmp, index, knowledge_root, runs_root, extra)
        report = audit["knowledge"]["dream"]
        checked = report["checked"]
        assert len(checked) == expected, (index, len(checked), report["candidates"])
        assert report["candidates"] == 10 - demoted, report
        assert report["remaining"] == report["candidates"] - expected, report
        assert report["summary"]["env_error"] == 0, report["summary"]
        demoted += report["summary"]["not_reproduced"]
        reviewed.extend(row["id"] for row in checked)
        assert len(set(reviewed)) == len(reviewed), "a record was checked twice: %s" % reviewed

    assert len(set(reviewed)) == 10, reviewed
    assert set(reviewed) == set(sids), (reviewed, sids)


def main():
    tmp = tempfile.mkdtemp(prefix="f5-schedule-")
    check("dream replays --dream-limit per cycle and covers 10 records in 4 cycles",
          lambda: check_dream_budget(tmp))
    failed = [row for row in RESULTS if row[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())