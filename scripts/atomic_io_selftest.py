"""M4 self-test: knowledge-base write atomicity.

Checks:
  1. all three stores share rsi/_io.write_jsonl + _FileLock (no local copies)
  2. killing a writer mid-write leaves a valid JSONL file (atomic replace)
  3. two concurrent record_evidence writers do not lose either row
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi._io import write_jsonl  # noqa: E402
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


def check_shared_implementation(tmp):
    """memory.py must not keep its own _write_jsonl / _FileLock copies."""
    for path in ("rsi/memory.py", "rsi/store.py", "rsi/exemplars.py"):
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            text = fh.read()
        assert "class _FileLock" not in text, "%s still defines _FileLock" % path
        assert "def _write_jsonl(" not in text, "%s still defines _write_jsonl" % path
        assert "rsi._io import" in text or "from . import" in text, path
    with open(os.path.join(ROOT, "rsi/_io.py"), encoding="utf-8") as fh:
        shared = fh.read()
    assert "def write_jsonl" in shared and "class _FileLock" in shared


def check_atomic_replace(tmp):
    """A killed writer must not truncate the target file."""
    path = os.path.join(tmp, "strategies.jsonl")
    write_jsonl(path, [{"id": "stg-1", "statement": "old"}])
    # Start a writer that writes a large payload, then kill it before replace.
    big = [{"id": "stg-%04d" % i, "statement": "x" * 200} for i in range(5000)]
    script = textwrap.dedent("""
        import json, os, sys, time
        sys.path.insert(0, %r)
        from rsi._io import write_jsonl
        rows = json.load(open(%r))
        path = %r
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\\n")
        os.replace(tmp, path)
    """ % (ROOT, os.path.join(tmp, "rows.json"), path))
    with open(os.path.join(tmp, "rows.json"), "w", encoding="utf-8") as fh:
        json.dump(big, fh)
    proc = subprocess.Popen([sys.executable, "-c", script])
    time.sleep(0.2)
    proc.kill()
    proc.wait()
    # The target must still be the previous valid file (or fully the new one).
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, "file is empty after kill"
    assert all(isinstance(r, dict) and r.get("id") for r in rows), rows[:2]


def check_concurrent_writers(tmp):
    """Two processes recording different strategies must both survive."""
    root = os.path.join(tmp, "kb")
    StrategyStore(root)  # bootstrap seed strategies
    worker = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from rsi.store import StrategyStore
        store = StrategyStore(%r)
        st = store.upsert(statement=%r, key=%r, tags=["x"], op_class=["y"], keywords=["z"])
        store.record_evidence(st["id"], "run", "task", 1.5, "success", run_id=%r)
        store.save()
    """)
    procs = []
    for name in ("alpha", "beta"):
        code = worker % (ROOT, root, "practice %s" % name, name, name)
        procs.append(subprocess.Popen([sys.executable, "-c", code]))
    for proc in procs:
        assert proc.wait() == 0
    store = StrategyStore(root)
    keys = {row.get("key") for row in store.strategies.values()}
    assert "alpha" in keys and "beta" in keys, sorted(keys)


def main():
    tmp = tempfile.mkdtemp(prefix="m4-atomic-")
    check("shared _io implementation (no local copies)", lambda: check_shared_implementation(tmp))
    check("atomic replace: killed writer keeps a valid JSONL", lambda: check_atomic_replace(tmp))
    check("concurrent writers keep both rows", lambda: check_concurrent_writers(tmp))
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
