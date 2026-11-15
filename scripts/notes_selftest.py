"""M3 self-test: the `note` action writes process-level experience into memory.

Checks:
  1. replay e2e writes a note; the NEXT run's memory index shows it
  2. a 4th note in the same run is rejected with a clear cap message
  3. a note without a trigger is rejected
  4. notes.jsonl is part of version.snapshot() output
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

from rsi.memory import MemoryStore  # noqa: E402
from rsi import version  # noqa: E402

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


def _run_loop(tmp, root, name, responses, extra=()):
    task_path = os.path.join(ROOT, "tasks", "level1", "softmax.py")
    replay = os.path.join(tmp, name + ".json")
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": responses}, fh)
    out_dir = os.path.join(tmp, name + "_runs")
    proc = subprocess.run([
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--knowledge-root", root,
        "--out-dir", out_dir, "--max-iters", "8", "--target-speedup", "0.5",
        "--timeout", "300", "--profiler", "off", "--no-distill",
    ] + list(extra), cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    return proc, out_dir


def check_note_visible_next_run(tmp):
    root = os.path.join(tmp, "visible", "kb")
    good = "import torch\n\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n"
    responses = [
        {"action": "note", "statement": "Row-major softmax is launch-bound on cpu",
         "trigger": "small rows"},
        {"action": "write_solution", "content": good, "strategies": [],
         "expected_speedup": 1.0},
        {"action": "finish", "summary": "note e2e"},
    ]
    proc, out_dir = _run_loop(tmp, root, "note1", responses)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1000:]
    mem = MemoryStore(root)
    assert mem.notes, "notes.jsonl should have one note"
    note = mem.notes[0]
    assert note["statement"].startswith("Row-major softmax"), note
    assert note["trigger"] == "small rows", note
    assert note["confidence"] == "candidate", note
    # the next run's memory index must include it
    feats = {"op": "softmax", "op_family": "reduction", "soc": "cpu",
             "dtype": ["float32"], "task": "softmax", "form": ["triton"]}
    index = mem.context_index(feats)
    assert note["id"] in index, index
    assert "NOTE" in index, index


def check_fourth_note_rejected(tmp):
    root = os.path.join(tmp, "cap", "kb")
    good = "import torch\n\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n"
    responses = [
        {"action": "note", "statement": "n1", "trigger": "t"},
        {"action": "note", "statement": "n2", "trigger": "t"},
        {"action": "note", "statement": "n3", "trigger": "t"},
        {"action": "note", "statement": "n4", "trigger": "t"},
        {"action": "write_solution", "content": good, "strategies": [],
         "expected_speedup": 1.0},
        {"action": "finish", "summary": "cap"},
    ]
    proc, out_dir = _run_loop(tmp, root, "cap", responses)
    assert proc.returncode == 0, proc.stdout[-1500:]
    metas = [os.path.join(out_dir, d, "meta.json") for d in os.listdir(out_dir)]
    transcript = [json.loads(line) for line in
                  open(os.path.join(os.path.dirname(metas[0]), "transcript.jsonl"),
                       encoding="utf-8") if line.strip()]
    user_msgs = [r.get("content") or "" for r in transcript if r.get("role") == "user"]
    assert any("at most 3 notes per run" in msg for msg in user_msgs), user_msgs[-4:]
    mem = MemoryStore(root)
    assert len(mem.notes) == 3, len(mem.notes)


def check_missing_trigger_rejected(tmp):
    root = os.path.join(tmp, "trigger", "kb")
    good = "import torch\n\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n"
    responses = [
        {"action": "note", "statement": "no trigger here"},
        {"action": "write_solution", "content": good, "strategies": [],
         "expected_speedup": 1.0},
        {"action": "finish", "summary": "trigger"},
    ]
    proc, out_dir = _run_loop(tmp, root, "trigger", responses)
    assert proc.returncode == 0, proc.stdout[-1500:]
    metas = [os.path.join(out_dir, d, "meta.json") for d in os.listdir(out_dir)]
    transcript = [json.loads(line) for line in
                  open(os.path.join(os.path.dirname(metas[0]), "transcript.jsonl"),
                       encoding="utf-8") if line.strip()]
    user_msgs = [r.get("content") or "" for r in transcript if r.get("role") == "user"]
    assert any("trigger" in msg for msg in user_msgs), user_msgs[-4:]
    mem = MemoryStore(root)
    assert len(mem.notes) == 0, len(mem.notes)


def check_snapshot_includes_notes(tmp):
    root = os.path.join(tmp, "snap", "kb")
    mem = MemoryStore(root)
    mem.add_note({"statement": "snapshot me", "trigger": "always",
                  "scope": {"soc": ["cpu"]}, "run_id": "r1"})
    snap = version.snapshot(root, os.path.join(tmp, "snaps"))
    names = set((snap.get("files") or {}).keys())
    assert "notes.jsonl" in names, names
    assert os.path.isfile(os.path.join(snap["path"], "notes.jsonl")), snap


def main():
    tmp = tempfile.mkdtemp(prefix="m3-notes-")
    check("note written in one run is visible in the next run's index",
          lambda: check_note_visible_next_run(tmp))
    check("4th note in one run is rejected with a cap message",
          lambda: check_fourth_note_rejected(tmp))
    check("note without a trigger is rejected", lambda: check_missing_trigger_rejected(tmp))
    check("notes.jsonl is part of version.snapshot()", lambda: check_snapshot_includes_notes(tmp))
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
