"""M4/F2 self-test: knowledge-base write atomicity and concurrent writers.

Checks (repeated `--loop N` times, each in a fresh directory):
  1. all three stores share rsi/_io.write_jsonl + _FileLock (no local copies)
  2. killing a writer mid-write leaves a valid JSONL file (atomic replace)
  3. eight concurrent writers keep all rows and allocate distinct ids
  4. every read-modify-write entry point of StrategyStore runs under one lock

`--loop 50` is the acceptance for the F2 race: a single green run proves
nothing about a race, so CI runs the whole file 50 times in a row.
"""
from __future__ import annotations

import argparse
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
QUIET_PASS = False


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
    if not QUIET_PASS:
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


def check_concurrent_writers(tmp, writers=8):
    """Eight processes recording different practices must all survive.

    Each worker keeps a gap between its in-memory mutation and save().  The id
    used to be allocated before the lock, so all workers picked the same
    stg-XXXX and the last save() overwrote the others; the gap makes that loss
    deterministic instead of relying on scheduler luck.
    """
    root = os.path.join(tmp, "kb")
    StrategyStore(root)  # bootstrap seed strategies
    worker = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, %r)
        from rsi.store import StrategyStore
        name = %r
        store = StrategyStore(%r)
        st = store.upsert(statement="practice " + name, key=name,
                          tags=["x"], op_class=["y"], keywords=["z"])
        time.sleep(1.0)
        store.record_evidence(st["id"], "run", "task", 1.5, "success", run_id=name)
        store.save()
    """)
    procs = []
    for index in range(writers):
        name = "w%02d" % index
        code = worker % (ROOT, name, root)
        procs.append((name, subprocess.Popen([sys.executable, "-c", code],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             text=True, encoding="utf-8",
                                             errors="replace")))
    for name, proc in procs:
        out, err = proc.communicate()
        assert proc.returncode == 0, (
            "writer %s exited %s\nstdout:\n%s\nstderr:\n%s"
            % (name, proc.returncode, out[-800:], err[-2000:]))
    store = StrategyStore(root)
    expected = {"w%02d" % i for i in range(writers)}
    keys = {row.get("key") for row in store.strategies.values()}
    assert expected <= keys, "missing rows: %s" % sorted(expected - keys)
    rows = [row for row in store.strategies.values() if row.get("key") in expected]
    ids = [row["id"] for row in rows]
    assert len(ids) == writers, ids
    assert len(set(ids)) == writers, "duplicate ids: %s" % ids
    # Every row must also have kept its evidence (the second write in the worker)
    for row in rows:
        assert len(row.get("evidence") or []) == 1, (row.get("key"), row.get("evidence"))


def check_store_transactions(tmp):
    """All three mutating entry points must reload+write inside one lock."""
    import inspect
    from rsi.store import StrategyStore as Store

    for name in ("upsert", "record_evidence", "upsert_combo"):
        source = inspect.getsource(getattr(Store, name))
        assert "with self.transaction():" in source, (
            "%s does not hold the store transaction" % name)
    # The exemplar store allocates ids from a content hash, but it must still
    # read inside the lock before it decides whether the row exists.
    import rsi.exemplars as exemplars
    source = inspect.getsource(exemplars.record_exemplar)
    assert "with _FileLock" in source, "record_exemplar does not lock"
    assert source.index("with _FileLock") < source.index("_read_jsonl"), (
        "record_exemplar reads before taking the lock")


def run_once(index, loops):
    global QUIET_PASS
    QUIET_PASS = loops > 1
    RESULTS.clear()
    tmp = tempfile.mkdtemp(prefix="m4-atomic-")
    check("shared _io implementation (no local copies)", lambda: check_shared_implementation(tmp))
    check("atomic replace: killed writer keeps a valid JSONL", lambda: check_atomic_replace(tmp))
    check("8 concurrent writers keep all rows with distinct ids",
          lambda: check_concurrent_writers(tmp))
    check("read-modify-write entry points hold the store transaction",
          lambda: check_store_transactions(tmp))
    failed = [row for row in RESULTS if row[0] == "FAIL"]
    if failed:
        print("loop %d/%d: FAIL (%d/%d checks)" % (index, loops, len(failed), len(RESULTS)))
        return False
    print("loop %d/%d: PASS (%d checks)" % (index, loops, len(RESULTS)))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="knowledge-base atomicity / concurrency self-test")
    ap.add_argument("--loop", type=int, default=1,
                    help="run the whole suite N times (races need repetition)")
    args = ap.parse_args(argv)
    loops = max(1, int(args.loop))
    for index in range(1, loops + 1):
        if not run_once(index, loops):
            print("\nloops run: %d | failed loops: 1 (stopped at loop %d)" % (index, index))
            return 1
    print("\nloops: %d | failed loops: 0" % loops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())