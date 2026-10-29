"""Structure/parameter separation checks (Phase 4.1)."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsi.params import ParamError, grid, parse_space, parse_tunable, render, validate  # noqa: E402

TASK_SOURCE = '''"""Tunable add task for the params self-test (CPU)."""
import numpy as np

TASK = {
    "name": "tunable_add", "level": 0,
    "tolerance": {"atol": 1e-6, "rtol": 1e-6},
    "num_trials": 1, "warmup": 1, "rep": 5,
}

def make_inputs(device, trial):
    rng = np.random.default_rng(31 + trial)
    return (rng.standard_normal(256).astype(np.float32),
            rng.standard_normal(256).astype(np.float32))

def reference(x, y):
    return x + y
'''

TEMPLATE = "def run(x, y):\n    return x + y + {{OFFSET}}\n"


def main():
    space = parse_space({"OFFSET": "1,0", "BLOCK": "range(64, 257, 64)"})
    assert space["OFFSET"] == [1, 0], space
    assert space["BLOCK"] == [64, 128, 192, 256], space
    parsed = parse_tunable("# TUNABLE: OFFSET=1,0\n" + TEMPLATE)
    assert parsed == {"OFFSET": [1, 0]}, parsed
    check = validate(TEMPLATE, {"OFFSET": [1, 0]})
    assert check["ok"], check
    assert render(TEMPLATE, {"OFFSET": 0}) == "def run(x, y):\n    return x + y + 0\n"
    variants = grid(space, max_variants=3, seed=7)
    assert len(variants) == 3, variants
    try:
        render(TEMPLATE, {})
    except ParamError:
        pass
    else:
        raise AssertionError("unresolved placeholder should fail")
    print("PASS  params: parse/render/grid/validation")

    tmp = tempfile.mkdtemp(prefix="kernel-agent-params-")
    task = os.path.join(tmp, "tasks", "tunable_add.py")
    os.makedirs(os.path.dirname(task), exist_ok=True)
    with open(task, "w", encoding="utf-8") as fh:
        fh.write(TASK_SOURCE)
    replay = os.path.join(tmp, "replay.json")
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": [
            {"action": "tune", "content": TEMPLATE, "params": {"OFFSET": [1, 0]}},
            {"action": "finish", "summary": "params self-test"},
        ]}, fh)
    out_dir = os.path.join(tmp, "runs")
    cmd = [sys.executable, "-m", "agent.loop", "--task", task, "--provider", "replay",
           "--replay-file", replay, "--device", "cpu", "--max-iters", "2",
           "--tune-max-variants", "2", "--target-speedup", "0.5",
           "--out-dir", out_dir, "--timeout", "120"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    meta = json.load(open(metas[0], encoding="utf-8"))
    assert meta["best"]["status"] == "pass", meta["best"]
    transcript = open(os.path.join(os.path.dirname(metas[0]), "transcript.md"),
                      encoding="utf-8").read()
    assert "Tuned 2/2" in transcript, transcript[-1500:]
    print("PASS  params: agent tune action renders/evaluates the grid")
    print("\n==================================================================")
    print("passed: 2 | failed: 0")
    print("note: CPU mechanism test only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())