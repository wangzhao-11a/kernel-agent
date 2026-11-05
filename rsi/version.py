"""Knowledge versioning + holdout score for the outer loop.

A curriculum cycle snapshots rsi/knowledge before and after it runs. The audit
records both version ids and the holdout score; if the score drops by more than
the allowed epsilon, the scheduler restores the previous snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VERSIONS = os.path.join(ROOT, "runs", "knowledge_versions")
KNOWLEDGE_FILES = (
    "strategies.jsonl", "combos.jsonl", "exemplars.jsonl",
    "facts.jsonl", "failures.jsonl", "skills.jsonl",
)


def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def knowledge_version(root: str) -> str:
    h = hashlib.sha256()
    for name in KNOWLEDGE_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            h.update(name.encode("utf-8"))
            h.update(open(path, "rb").read())
    return h.hexdigest()[:12]


def snapshot(root: str, dest_root: str = None) -> dict:
    version = knowledge_version(root)
    version_id = "%s-%s" % (_now(), version)
    dest = os.path.join(dest_root or DEFAULT_VERSIONS, version_id)
    os.makedirs(dest, exist_ok=True)
    files = {}
    for name in KNOWLEDGE_FILES:
        source = os.path.join(root, name)
        if not os.path.isfile(source):
            continue
        shutil.copy2(source, os.path.join(dest, name))
        files[name] = {"sha256": _sha(source), "bytes": os.path.getsize(source)}
    with open(os.path.join(dest, "version.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": version_id, "version": version, "root": os.path.abspath(root),
                   "files": files, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  fh, ensure_ascii=False, indent=2)
    return {"id": version_id, "version": version, "path": dest, "files": files}


def restore(root: str, version_dir: str) -> dict:
    for name in KNOWLEDGE_FILES:
        source = os.path.join(version_dir, name)
        if os.path.isfile(source):
            os.makedirs(root, exist_ok=True)
            shutil.copy2(source, os.path.join(root, name))
    return {"restored_from": version_dir, "version": knowledge_version(root)}


def holdout_score(runs_root, holdout_tasks) -> dict:
    """Mean best speedup over passing runs of the configured holdout tasks."""
    wanted = {str(name) for name in (holdout_tasks or []) if str(name)}
    roots = runs_root if isinstance(runs_root, (list, tuple)) else [runs_root]
    values = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            run_dir = os.path.join(root, name)
            meta_path = os.path.join(run_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                continue
            if str(meta.get("task") or "") not in wanted:
                continue
            best = meta.get("best") or {}
            if best.get("status") not in ("pass", "pass_unstable"):
                continue
            try:
                values.append(float(best.get("speedup") or 0.0))
            except (TypeError, ValueError):
                continue
    return {
        "score": (sum(values) / len(values)) if values else None,
        "n_runs": len(values),
        "tasks": sorted(wanted),
    }


def rollback_if_worse(root: str, before: dict, before_score: dict, after_score: dict,
                      epsilon: float = 0.01) -> dict:
    """Restore `before` when the holdout score dropped by more than epsilon."""
    old = (before_score or {}).get("score")
    new = (after_score or {}).get("score")
    if old is None or new is None:
        return {"rolled_back": False, "reason": "holdout score unavailable"}
    if float(new) >= float(old) - float(epsilon):
        return {"rolled_back": False,
                "reason": "holdout score %.4f -> %.4f (within epsilon %.3f)" % (old, new, epsilon)}
    restored = restore(root, before["path"])
    return {"rolled_back": True,
            "reason": "holdout score %.4f -> %.4f (drop > %.3f)" % (old, new, epsilon),
            "restored": restored}
