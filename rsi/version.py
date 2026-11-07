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


def evaluate_holdout(knowledge_root, task_paths, out_dir, runner=None, seeds: int = 1,
                     max_iters: int = 4, timeout: float = 900.0) -> dict:
    """Re-run the fixed holdout tasks against this knowledge base.

    `runner(task_path, knowledge_root, seed, run_dir) -> {"status", "speedup"}`
    defaults to `agent.loop`.  Score per task is `log(speedup)` (baseline is the
    reference at 1.0x), averaged over tasks: a geometric-mean gain, so a slow task
    cannot dominate a fast one and scores are comparable across knowledge bases.
    """
    import math

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tasks = {}
    log_scores = []
    for task_path in task_paths:
        name = os.path.splitext(os.path.basename(task_path))[0]
        per_seed = []
        for seed in range(1, max(1, int(seeds)) + 1):
            run_dir = os.path.join(out_dir, name, "seed%d" % seed)
            try:
                result = (runner or _agent_runner)(
                    task_path, knowledge_root, seed, run_dir,
                    max_iters=max_iters, timeout=timeout)
            except Exception as exc:
                result = {"status": "error", "speedup": 0.0,
                          "error": "%s: %s" % (type(exc).__name__, exc)}
            per_seed.append(result)
        ok = [row for row in per_seed if row.get("status") in ("pass", "pass_unstable")]
        speedups = [float(row.get("speedup") or 0.0) for row in ok]
        speedup = max(speedups) if speedups else 0.0
        log_ratio = math.log(speedup) if speedup > 0 else None
        tasks[name] = {
            "speedup": round(speedup, 4),
            "log_ratio": round(log_ratio, 4) if log_ratio is not None else None,
            "passes": len(ok),
            "runs": len(per_seed),
            "statuses": [row.get("status") for row in per_seed],
        }
        if log_ratio is not None:
            log_scores.append(log_ratio)
    return {
        "score": round(sum(log_scores) / len(log_scores), 4) if log_scores else None,
        "metric": "mean log(speedup) over fixed holdout tasks",
        "n_tasks": len(tasks),
        "tasks": tasks,
        "knowledge_root": os.path.abspath(knowledge_root),
    }


def _agent_runner(task_path, knowledge_root, seed, run_dir, max_iters: int = 4,
                  timeout: float = 900.0) -> dict:
    """Default holdout runner: one agent.loop run, no distillation."""
    import glob
    import subprocess
    import sys

    os.makedirs(run_dir, exist_ok=True)
    provider = os.environ.get("KERNEL_AGENT_PROVIDER", "replay")
    model = os.environ.get("KERNEL_AGENT_MODEL", "deepseek-chat")
    base_url = os.environ.get("KERNEL_AGENT_BASE_URL", "https://api.deepseek.com/v1")
    key_env = os.environ.get("KERNEL_AGENT_API_KEY_ENV", "DEEPSEEK_API_KEY")
    device = os.environ.get("KERNEL_AGENT_DEVICE", "cpu")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path,
        "--provider", provider, "--model", model, "--base-url", base_url,
        "--api-key-env", key_env, "--device", device,
        "--knowledge-root", knowledge_root, "--out-dir", run_dir,
        "--max-iters", str(int(max_iters)), "--timeout", str(timeout),
        "--profiler", "off", "--no-distill", "--quiet",
    ]
    replay = os.environ.get("KERNEL_AGENT_REPLAY", "")
    if provider == "replay" and replay:
        cmd += ["--replay-file", replay]
    if os.environ.get("KERNEL_AGENT_HARDWARE"):
        cmd += ["--hardware", os.environ["KERNEL_AGENT_HARDWARE"]]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=timeout + 240)
    metas = glob.glob(os.path.join(run_dir, "*", "meta.json"))
    if not metas:
        return {"status": "no_meta", "speedup": 0.0}
    with open(sorted(metas)[-1], "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    best = meta.get("best") or {}
    return {"status": best.get("status"), "speedup": float(best.get("speedup") or 0.0),
            "task": meta.get("task"), "seed": seed}
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
