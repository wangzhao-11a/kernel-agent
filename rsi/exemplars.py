"""Code exemplar library: winning solutions recorded with their measured context.

The strategy pool records *what* worked (statements); this records *how it looked*
(the code), so retrieval can inject few-shot examples instead of prose. Entries are
condition-tagged (task, hardware, dtype, op class, tags) and de-duplicated by content
hash; the best few per (task, hardware) are kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
EXEMPLARS_FILE = "exemplars.jsonl"
MAX_SOURCE_CHARS = 8000
MAX_PER_TASK_HW = 3


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path: str) -> list:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=fh)


def record_exemplar(root: str, task: str, source: str, speedup: float, hardware: str = "",
                    dtypes=None, op_class: str = "", tags=None, strategies=None,
                    profile_top=None, run_ref: str = "") -> dict:
    """Store a winning solution (idempotent by content hash)."""
    path = os.path.join(root, EXEMPLARS_FILE)
    rows = _read_jsonl(path)
    digest = hashlib.sha256((source or "").encode("utf-8")).hexdigest()[:16]
    for row in rows:
        if row.get("id") == digest:
            row["speedup"] = round(max(float(row.get("speedup") or 0.0), float(speedup)), 4)
            row["updated"] = _now()
            _write_jsonl(path, rows)
            return row
    row = {
        "id": digest,
        "task": task,
        "speedup": round(float(speedup), 4),
        "hardware": hardware,
        "dtypes": list(dtypes or []),
        "op_class": op_class,
        "tags": list(tags or []),
        "strategies": list(strategies or []),
        "profile_top": list(profile_top or []),
        "run_ref": run_ref,
        "source": (source or "")[:MAX_SOURCE_CHARS],
        "created": _now(),
        "updated": _now(),
    }
    rows.append(row)
    # keep at most MAX_PER_TASK_HW exemplars per (task, hardware), best speedups first
    keep, buckets = [], {}
    for candidate in sorted(rows, key=lambda r: -float(r.get("speedup") or 0.0)):
        key = (candidate.get("task"), candidate.get("hardware"))
        if buckets.get(key, 0) < MAX_PER_TASK_HW:
            buckets[key] = buckets.get(key, 0) + 1
            keep.append(candidate)
    _write_jsonl(path, keep)
    return row


def _score(row: dict, feats: dict, hardware: str) -> float:
    ftags = set(feats.get("tags") or [])
    score = len(set(row.get("tags") or []) & ftags) / max(1, len(ftags))
    if row.get("op_class") and row.get("op_class") == (feats.get("op_class") or ""):
        score += 0.5
    if row.get("hardware") == hardware:
        score += 0.3
    if (feats.get("dtypes") or [""])[0] in (row.get("dtypes") or []):
        score += 0.1
    if row.get("task") == feats.get("name"):
        score += 0.4  # an exemplar from the same task is the strongest hint
    score += 0.2 * min(2.0, float(row.get("speedup") or 1.0) - 1.0)
    return score


def retrieve_exemplars(root: str, feats: dict, hardware: str, top_k: int = 2,
                       min_speedup: float = 1.0) -> list:
    rows = _read_jsonl(os.path.join(root, EXEMPLARS_FILE))
    scored = [(row, _score(row, feats, hardware)) for row in rows
              if float(row.get("speedup") or 0.0) >= min_speedup and row.get("source")]
    scored.sort(key=lambda item: -item[1])
    return [row for row, _score_value in scored[:top_k]]


def render_exemplars(exemplars: list, max_chars: int = 6000) -> str:
    """Render a budgeted few-shot block for the task prompt."""
    if not exemplars:
        return ""
    lines = [
        "## Code exemplars from the pool (already verified on hardware — study, reuse or beat)",
        "These solutions passed correctness and were measured under matching conditions:",
    ]
    budget = max_chars
    for row in exemplars:
        header = "### %s on %s — %.2fx (strategies: %s)" % (
            row.get("task"), row.get("hardware") or "?", float(row.get("speedup") or 0.0),
            ", ".join(row.get("strategies") or []) or "-")
        if row.get("profile_top"):
            header += "\n# profile top ops: %s" % ", ".join(str(o) for o in row["profile_top"][:3])
        code = "```python\n%s\n```" % (row.get("source") or "").strip()
        block = header + "\n" + code
        if len(block) > budget:
            break
        budget -= len(block)
        lines.append(block)
    return "\n".join(lines)