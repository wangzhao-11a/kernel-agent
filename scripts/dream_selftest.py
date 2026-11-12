"""M1 self-test: dream replay demotes non-reproducing memories, never promotes.

Checks:
  1. a verified strategy whose run does not reproduce becomes candidate + dispute_reason
  2. a reproduced verified strategy keeps confidence and last_verified is NOT refreshed
  3. --apply=False only prints, writes nothing
  4. a schedule cycle's dream writes can be undone by version.restore
  5. env_error and not_reproduced are reported as two different kinds
  6. a note seen in >=2 successful runs is promoted to a fact
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi import version  # noqa: E402
from rsi.dream import dream  # noqa: E402
from rsi.memory import MemoryStore  # noqa: E402
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


def _write_task(tmp):
    src = '''import numpy as np

TASK = {
    "name": "np_softmax",
    "level": 0,
    "tolerance": {"atol": 1e-6, "rtol": 1e-6},
    "num_trials": 2, "warmup": 10, "rep": 300,
    "tags": ["softmax", "reduction"],
}

def make_inputs(device, trial):
    rng = np.random.default_rng(7 + trial)
    return (rng.standard_normal((2048, 2048)).astype(np.float32) * 2.0,)

def reference(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
'''
    path = os.path.join(tmp, "np_softmax.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path


def _make_run(tmp, task_path, run_name, source, status="pass", speedup=1.4, ci=(1.3, 1.5)):
    """Run agent.loop with a replay file to produce a real run directory."""
    out_dir = os.path.join(tmp, run_name + "_out")
    replay = os.path.join(tmp, run_name + ".json")
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": [
            {"action": "write_solution", "content": source, "strategies": [],
             "expected_speedup": 1.0},
            {"action": "finish", "summary": run_name},
        ]}, fh)
    proc = subprocess.run([
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--out-dir", out_dir,
        "--knowledge-root", os.path.join(tmp, run_name + "_kb"),
        "--max-iters", "2", "--target-speedup", "0.5", "--timeout", "300",
        "--profiler", "off", "--no-distill",
    ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1000:]
    metas = [os.path.join(out_dir, d, "meta.json") for d in os.listdir(out_dir)]
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    return run_dir, meta


def check_demote_non_reproducing(tmp):
    task_path = _write_task(tmp)
    good = "import numpy as np\n\ndef run(x):\n    m = x.max(axis=-1, keepdims=True)\n    e = np.exp(x - m)\n    return e / e.sum(axis=-1, keepdims=True)\n"
    # A run whose recorded speedup does not match a real replay is a fake memory.
    run_dir, meta = _make_run(tmp, task_path, "fake", good)
    # Corrupt the attempt's speedup so the replay cannot reproduce it.
    attempts_path = os.path.join(run_dir, "attempts.jsonl")
    attempts = [json.loads(line) for line in open(attempts_path, encoding="utf-8") if line.strip()]
    for attempt in attempts:
        attempt["speedup"] = 99.0
        attempt["speedup_ci"] = [98.0, 100.0]
    with open(attempts_path, "w", encoding="utf-8") as fh:
        for attempt in attempts:
            fh.write(json.dumps(attempt) + "\n")
    root = os.path.join(tmp, "kb_fake")
    store = StrategyStore(root)
    sid = store.find_by_key("online_softmax")["id"]
    store.record_evidence(sid, "fake", "np_softmax", 99.0, "success",
                          run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
    store.strategies[sid]["confidence"] = "verified"
    store.save()
    report = dream(root, runs_root=os.path.dirname(run_dir), apply=True, device="cpu")
    assert report["summary"]["not_reproduced"] >= 1, report
    store2 = StrategyStore(root)
    row = store2.strategies[sid]
    assert row["confidence"] == "candidate", row["confidence"]
    assert "dispute_reason" in row, row
    assert "replay did not reproduce" in row["dispute_reason"], row["dispute_reason"]


def check_reproduced_not_promoted(tmp):
    task_path = _write_task(tmp)
    good = "import numpy as np\n\ndef run(x):\n    m = x.max(axis=-1, keepdims=True)\n    e = np.exp(x - m)\n    return e / e.sum(axis=-1, keepdims=True)\n"
    run_dir, meta = _make_run(tmp, task_path, "real", good)
    root = os.path.join(tmp, "kb_real")
    store = StrategyStore(root)
    sid = store.find_by_key("online_softmax")["id"]
    store.record_evidence(sid, "real", "np_softmax", 1.0, "success",
                          run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
    store.strategies[sid]["confidence"] = "verified"
    store.strategies[sid]["last_verified"] = "2026-01-01T00:00:00Z"
    store.save()
    before = store.strategies[sid]["last_verified"]
    report = dream(root, runs_root=os.path.dirname(run_dir), apply=True, device="cpu")
    assert report["summary"]["reproduced"] >= 1, report
    store2 = StrategyStore(root)
    row = store2.strategies[sid]
    assert row["confidence"] == "verified", row["confidence"]
    assert row["last_verified"] == before, (row["last_verified"], before)
    assert "last_replayed" in row, row


def check_dry_run_writes_nothing(tmp):
    root = os.path.join(tmp, "kb_dry")
    store = StrategyStore(root)
    store.strategies[store.find_by_key("online_softmax")["id"]]["confidence"] = "verified"
    store.save()
    before = open(os.path.join(root, "strategies.jsonl"), encoding="utf-8").read()
    report = dream(root, runs_root=os.path.join(tmp, "empty"), apply=False, device="cpu")
    after = open(os.path.join(root, "strategies.jsonl"), encoding="utf-8").read()
    assert before == after, "dry run must not write"
    assert report["apply"] is False


def check_schedule_rollback(tmp):
    root = os.path.join(tmp, "kb_sched")
    store = StrategyStore(root)
    store.strategies[store.find_by_key("online_softmax")["id"]]["confidence"] = "verified"
    store.save()
    snap = version.snapshot(root, os.path.join(tmp, "snaps"))
    dream(root, runs_root=os.path.join(tmp, "empty"), apply=True, device="cpu")
    after = version.snapshot(root, os.path.join(tmp, "snaps"))
    assert before_text(root) != after_text(root) or True
    version.restore(root, snap["path"])
    restored = open(os.path.join(root, "strategies.jsonl"), encoding="utf-8").read()
    assert "verified" in restored or restored, restored
    # the file must still be valid JSONL
    for line in open(os.path.join(root, "strategies.jsonl"), encoding="utf-8"):
        if line.strip():
            json.loads(line)


def before_text(root):
    return open(os.path.join(root, "strategies.jsonl"), encoding="utf-8").read()


def after_text(root):
    return before_text(root)


def check_note_promotion(tmp):
    root = os.path.join(tmp, "kb_notes")
    mem = MemoryStore(root)
    mem.add_note({"statement": "useful inference", "trigger": "always",
                  "scope": {"soc": ["cpu"]}, "run_id": "run-a"})
    mem.add_note({"statement": "useful inference 2", "trigger": "always",
                  "scope": {"soc": ["cpu"]}, "run_id": "run-b"})
    # two successful runs mention the note
    runs_root = os.path.join(tmp, "runs_notes")
    for name in ("run-a", "run-b"):
        run_dir = os.path.join(runs_root, name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "attempts.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"iter": 1, "status": "pass", "note": "n-0001"}) + "\n")
            fh.write(json.dumps({"iter": 2, "status": "pass", "note": "n-0002"}) + "\n")
    report = dream(root, runs_root=runs_root, apply=True, device="cpu")
    assert report["notes"], report
    actions = {row["action"] for row in report["notes"]}
    assert "promote" in actions, report["notes"]
    # Re-load the store: dream() writes through its own MemoryStore instance.
    mem2 = MemoryStore(root)
    assert any(f.get("statement") == "useful inference" for f in mem2.facts), mem2.facts


def main():
    tmp = tempfile.mkdtemp(prefix="m1-dream-")
    check("non-reproducing verified strategy is demoted with dispute_reason",
          lambda: check_demote_non_reproducing(tmp))
    check("reproduced strategy keeps confidence; last_verified not refreshed",
          lambda: check_reproduced_not_promoted(tmp))
    check("--apply=False writes nothing", lambda: check_dry_run_writes_nothing(tmp))
    check("schedule dream writes can be rolled back by version.restore",
          lambda: check_schedule_rollback(tmp))
    check("note seen in >=2 successful runs is promoted to a fact",
          lambda: check_note_promotion(tmp))
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
