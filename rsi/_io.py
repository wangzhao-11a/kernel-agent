"""Shared JSONL I/O: one lock, one atomic writer, for every knowledge store.

`rsi/store.py` (strategies/combos) and `rsi/exemplars.py` used to write with a
plain `open(path, "w")`, which truncates the file first: a crash mid-write, or
two writers racing, leaves a half-written knowledge base.  `rsi/memory.py` had
the right pattern; this module is the single home for it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryLockError(RuntimeError):
    pass


class _FileLock:
    """Cross-process lock for the memory JSONL files.

    Multiple agent runs (different cards/containers) may distill at the same time.
    Without a lock each process does load -> mutate -> save and silently drops the
    others' updates.  The lock file contains the holder pid; a stale lock (crashed
    process) is removed automatically.
    """

    def __init__(self, path: str, timeout: float = 20.0, poll: float = 0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self.acquired = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            try:
                out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                                     capture_output=True, text=True, timeout=10)
                return str(pid) in out.stdout
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    with open(self.path, "r", encoding="utf-8") as fh:
                        held = json.load(fh)
                except (OSError, ValueError):
                    held = {}
                pid = int(held.get("pid") or 0)
                if not self._pid_alive(pid):
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                if time.time() >= deadline:
                    raise MemoryLockError("memory lock held by pid %s: %s" % (pid, self.path))
                time.sleep(self.poll)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "created": utc_now()}, fh)
            self.acquired = True
            return self

    def __exit__(self, *exc):
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False


def write_jsonl(path: str, rows: list) -> None:
    """Atomic JSONL write: temp file + os.replace, never a truncated target."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
