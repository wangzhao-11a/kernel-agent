"""M1 self-test: dream replay demotes non-reproducing memories, never promotes.

Checks:
  1. a verified strategy whose run does not reproduce becomes candidate + dispute_reason
  2. a reproduced verified strategy keeps confidence and last_verified is NOT refreshed
  3. --apply=False only prints, writes nothing
  4. a schedule cycle's dream writes can be undone by version.restore
  5. env_error and not_reproduced are reported as two different kinds
  6. env_error never demotes: the record keeps verified and gets a retry marker
  7. a round with more than 50% env_errors writes no demotion at all (aborted)
  8. a note seen in >=2 successful runs is promoted to a fact
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


def _make_run(tmp, task_path, run_name, source, status="pass", speedup=1.4, ci=(1.3, 1.5),
              out_dir=None):
    """Run agent.loop with a replay file to produce a real run directory."""
    out_dir = out_dir or os.path.join(tmp, run_name + "_out")
    before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
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
    new = sorted(set(os.listdir(out_dir)) - before)
    assert new, "agent.loop produced no new run directory"
    metas = [os.path.join(out_dir, d, "meta.json") for d in new]
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    return run_dir, meta


def _write_small_task(tmp, name="np_softmax_small"):
    """A fast real task, so the 10-run environment-abort check stays affordable."""
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
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
''' % name
    path = os.path.join(tmp, name + ".py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path


def _make_empty_run(tmp, task_path, run_name, out_dir=None):
    """A real agent.loop run that finished without evaluating anything.

    replay_trajectory reports n=0 for it, which dream classifies as env_error
    (the run exists but has no replayable attempt).
    """
    out_dir = out_dir or os.path.join(tmp, run_name + "_out")
    before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    replay = os.path.join(tmp, run_name + ".json")
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": [{"action": "finish", "summary": run_name}]}, fh)
    proc = subprocess.run([
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--out-dir", out_dir,
        "--knowledge-root", os.path.join(tmp, run_name + "_kb"),
        "--max-iters", "2", "--target-speedup", "0.5", "--timeout", "120",
        "--profiler", "off", "--no-distill",
    ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300)
    # No passing attempt means the loop exits 1; the run directory is still real.
    assert proc.returncode in (0, 1), proc.stdout[-1500:] + proc.stderr[-1000:]
    new = sorted(set(os.listdir(out_dir)) - before)
    assert new, proc.stdout[-1500:]
    return os.path.join(out_dir, new[0])


_GOOD_SRC = ("import numpy as np\n\n"
             "def run(x):\n"
             "    m = x.max(axis=-1, keepdims=True)\n"
             "    e = np.exp(x - m)\n"
             "    return e / e.sum(axis=-1, keepdims=True)\n")
# Same math, much slower: a real replay that cannot reproduce the recorded
# speedup, without editing the recorded attempt row.
_SLOW_SRC = ("import numpy as np\n\n"
             "def run(x):\n"
             "    m = x.max(axis=-1, keepdims=True)\n"
             "    e = np.exp(x - m)\n"
             "    y = e / e.sum(axis=-1, keepdims=True)\n"
             "    for _ in range(40):\n"
             "        y = y.copy()\n"
             "    return y\n")


def _replace_solution(run_dir, source):
    """Replace the solution file of the last real attempt (the row stays real)."""
    path = os.path.join(run_dir, "attempts.jsonl")
    attempts = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert attempts, run_dir
    with open(attempts[-1]["solution"], "w", encoding="utf-8") as fh:
        fh.write(source)


def check_env_error_does_not_demote(tmp):
    """A run we cannot replay is not evidence against the memory.

    Mixed round (1 env_error + 1 not_reproduced = exactly 50%): the abort rule
    does not trigger, so this isolates the env_error branch while also checking
    that a real non-reproducing record is still demoted.
    """
    runs_root = os.path.join(tmp, "mixed_runs")
    os.makedirs(runs_root, exist_ok=True)
    env_task = _write_small_task(tmp, "env_task")
    env_dir = _make_empty_run(tmp, env_task, "noeval", out_dir=runs_root)
    slow_task = _write_small_task(tmp, "slow_task")
    slow_dir, _meta = _make_run(tmp, slow_task, "slow", _GOOD_SRC, out_dir=runs_root)
    _replace_solution(slow_dir, _SLOW_SRC)

    root = os.path.join(tmp, "kb_env")
    store = StrategyStore(root)
    env_sid, slow_sid = [row["id"] for row in list(store.strategies.values())[:2]]
    for sid, run_dir in ((env_sid, env_dir), (slow_sid, slow_dir)):
        store.record_evidence(sid, os.path.basename(run_dir), "mixed_task", 1.0, "neutral",
                              run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
        store.strategies[sid]["confidence"] = "verified"
    store.save()

    report = dream(root, runs_root=runs_root, apply=True, device="cpu")
    assert report["summary"]["env_error"] == 1, report
    assert report["summary"]["not_reproduced"] == 1, report
    assert report["aborted"] == "", report
    rows = StrategyStore(root).strategies

    env_row = rows[env_sid]
    assert env_row["confidence"] == "verified", env_row["confidence"]
    assert env_row.get("demote_count") in (None, 0), env_row
    assert "replay_blocked_reason" in env_row, env_row
    assert "replay_blocked_at" in env_row, env_row
    assert "dispute_reason" not in env_row, env_row

    slow_row = rows[slow_sid]
    assert slow_row["confidence"] == "candidate", slow_row["confidence"]
    assert "replay did not reproduce" in slow_row["dispute_reason"], slow_row["dispute_reason"]


def check_blocked_records_retried_first(tmp):
    """A record blocked by an environment error jumps the queue next round."""
    runs_root = os.path.join(tmp, "retry_runs")
    os.makedirs(runs_root, exist_ok=True)
    a_task = _write_small_task(tmp, "retry_task_a")
    b_task = _write_small_task(tmp, "retry_task_b")
    a_dir = _make_empty_run(tmp, a_task, "retry_a", out_dir=runs_root)
    b_dir = _make_empty_run(tmp, b_task, "retry_b", out_dir=runs_root)
    root = os.path.join(tmp, "kb_retry")
    store = StrategyStore(root)
    a_sid, b_sid = [row["id"] for row in list(store.strategies.values())[:2]]
    for sid, run_dir in ((a_sid, a_dir), (b_sid, b_dir)):
        store.record_evidence(sid, os.path.basename(run_dir), "retry_task", 1.0, "neutral",
                              run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
        store.strategies[sid]["confidence"] = "verified"
    # A was blocked recently; B was checked long ago.  The blocked flag must win
    # over recency, otherwise a broken run only retries once everything else has.
    store.strategies[a_sid]["replay_blocked_at"] = "2026-01-02T00:00:00Z"
    store.strategies[a_sid]["replay_blocked_reason"] = "device offline"
    store.strategies[a_sid]["last_replayed"] = "2026-11-14T00:00:00Z"
    store.strategies[b_sid]["last_replayed"] = "2026-01-01T00:00:00Z"
    store.save()
    report = dream(root, runs_root=runs_root, apply=True, device="cpu", limit=1)
    assert [row["id"] for row in report["checked"]] == [a_sid], report["checked"]


def check_env_error_majority_aborts(tmp):
    """More than half the replays failing is an environment problem, not memory."""
    runs_root = os.path.join(tmp, "abort_runs")
    os.makedirs(runs_root, exist_ok=True)
    env_dirs = []
    for i in range(6):
        task = _write_small_task(tmp, "env_task_%d" % i)
        env_dirs.append(_make_empty_run(tmp, task, "env_%d" % i, out_dir=runs_root))
    slow_dirs = []
    for i in range(4):
        task = _write_small_task(tmp, "slow_task_%d" % i)
        run_dir, _meta = _make_run(tmp, task, "slow_%d" % i, _GOOD_SRC, out_dir=runs_root)
        _replace_solution(run_dir, _SLOW_SRC)
        slow_dirs.append(run_dir)

    root = os.path.join(tmp, "kb_abort")
    store = StrategyStore(root)
    sids = [row["id"] for row in list(store.strategies.values())[:10]]
    assert len(sids) == 10
    for sid, run_dir in zip(sids, env_dirs + slow_dirs):
        store.record_evidence(sid, os.path.basename(run_dir), "abort_task", 1.0, "neutral",
                              run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
        store.strategies[sid]["confidence"] = "verified"
    store.save()

    report = dream(root, runs_root=runs_root, apply=True, device="cpu")
    assert report["summary"]["env_error"] == 6, report
    assert report["summary"]["not_reproduced"] == 4, report
    assert report["aborted"] == "environment", report
    assert report["env_error_ratio"] == 0.6, report
    store2 = StrategyStore(root)
    for sid in sids:
        row = store2.strategies[sid]
        assert row["confidence"] == "verified", (sid, row["confidence"])
        assert row.get("demote_count") in (None, 0), (sid, row)
    blocked = [row for row in store2.strategies.values() if row.get("replay_blocked_at")]
    assert len(blocked) == 6, [row["id"] for row in blocked]
    # The 4 real not_reproduced records were measured, so they are not "blocked".
    for rid in sids[6:]:
        assert "replay_blocked_at" not in store2.strategies[rid], store2.strategies[rid]


def check_demote_non_reproducing(tmp):
    task_path = _write_small_task(tmp, "demote_task")
    # A real run whose recorded speedup the replay cannot reproduce is a fake memory.
    run_dir, meta = _make_run(tmp, task_path, "fake", _GOOD_SRC)
    _replace_solution(run_dir, _SLOW_SRC)
    root = os.path.join(tmp, "kb_fake")
    store = StrategyStore(root)
    sid = store.find_by_key("online_softmax")["id"]
    store.record_evidence(sid, "fake", "np_softmax_small", 1.0, "success",
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
    """A dream demotion must be undoable byte-for-byte by version.restore."""
    runs_root = os.path.join(tmp, "rollback_runs")
    os.makedirs(runs_root, exist_ok=True)
    task = _write_small_task(tmp, "rollback_task")
    run_dir, _meta = _make_run(tmp, task, "rollback", _GOOD_SRC, out_dir=runs_root)
    # A real record the replay cannot reproduce, so dream really writes.
    _replace_solution(run_dir, _SLOW_SRC)
    root = os.path.join(tmp, "kb_sched")
    store = StrategyStore(root)
    sid = store.find_by_key("online_softmax")["id"]
    store.record_evidence(sid, os.path.basename(run_dir), "rollback_task", 1.0, "neutral",
                          run_id=os.path.basename(run_dir), hardware="cpu", dtype="float32")
    store.strategies[sid]["confidence"] = "verified"
    store.save()

    path = os.path.join(root, "strategies.jsonl")
    before = open(path, "rb").read()
    snap = version.snapshot(root, os.path.join(tmp, "snaps"))
    report = dream(root, runs_root=runs_root, apply=True, device="cpu")
    assert report["summary"]["not_reproduced"] == 1, report
    after = open(path, "rb").read()
    assert after != before, "dream did not write the demotion this test relies on"

    version.restore(root, snap["path"])
    restored = open(path, "rb").read()
    assert restored == before, "version.restore did not restore the pre-dream bytes"
    # the file must still be valid JSONL
    for line in open(path, encoding="utf-8"):
        if line.strip():
            json.loads(line)


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
    check("env_error keeps verified and marks the record for a retry",
          lambda: check_env_error_does_not_demote(tmp))
    check("blocked records are retried before recently checked ones",
          lambda: check_blocked_records_retried_first(tmp))
    check("majority env_error aborts the round without demoting",
          lambda: check_env_error_majority_aborts(tmp))
    check("note seen in >=2 successful runs is promoted to a fact",
          lambda: check_note_promotion(tmp))
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
