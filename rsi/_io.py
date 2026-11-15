"""Shared JSONL I/O: one lock, one atomic writer, for every knowledge store.

`rsi/store.py` (strategies/combos) and `rsi/exemplars.py` used to write with a
plain `open(path, "w")`, which truncates the file first: a crash mid-write, or
two writers racing, leaves a half-written knowledge base.  `rsi/memory.py` had
the right pattern; this module is the single home for it.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryLockError(RuntimeError):
    pass


class _FileLock:
    """Cross-process lock for the knowledge JSONL files.

    Uses an OS advisory lock on a persistent lock file: `msvcrt.locking` on
    Windows, `flock` on POSIX.  The kernel releases it when the process exits,
    so there is no stale-lock detection, no pid parsing and no unlink race --
    the previous "create an empty file, then write the holder pid" scheme let a
    second process see the empty window, treat the lock as stale and enter it
    too, which is exactly the lost-update bug this lock exists to prevent.
    The lock file itself is never deleted (deleting it would reintroduce the
    same race); it is a zero-byte dotfile next to the data.
    """

    def __init__(self, path: str, timeout: float = 20.0, poll: float = 0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self.acquired = False
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fh = open(self.path, "a+b")
        deadline = time.time() + self.timeout
        while True:
            try:
                fh.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fh
            except OSError:
                if time.time() >= deadline:
                    fh.close()
                    raise MemoryLockError("memory lock is busy: %s" % self.path)
                time.sleep(self.poll)
                continue
            self.acquired = True
            return self

    def __exit__(self, *exc):
        fh = self._fh
        if fh is None:
            return
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()
            self._fh = None
            self.acquired = False


def write_jsonl(path: str, rows: list) -> None:
    """Atomic JSONL write: temp file + os.replace, never a truncated target."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Per-process temp name: two writers must never share the scratch path even
    # if a lock is (wrongly) lost, otherwise one os.replace moves the other's
    # file and the second replace fails with FileNotFoundError.
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # On Windows os.replace fails with PermissionError while another process
    # (a reader that does not take the lock) has the target open.  The write is
    # still atomic; retry briefly instead of losing the update.
    last = None
    for _attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(0.05)
    raise last