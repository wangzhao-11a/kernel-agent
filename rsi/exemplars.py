"""Code exemplar library: winning solutions recorded with their measured context.

The strategy pool records *what* worked (statements); this records *how it looked*
(the code), so retrieval can inject few-shot examples instead of prose. Entries are
de-duplicated by content hash and kept as a quality-diversity archive: one fastest
example per behavior cell (implementation form x main bottleneck), so a slower
triton variant and a faster torch variant of the same task can coexist.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
EXEMPLARS_FILE = "exemplars.jsonl"
MAX_SOURCE_CHARS = 8000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _form(source: str) -> str:
    text = (source or "").lower()
    if "triton" in text:
        return "triton"
    if "ascendc" in text or "ascend_c" in text or "aclnn" in text or "kernel_launch" in text:
        return "ascendc"
    return "torch_fused"


def _bottleneck(profile_top) -> str:
    top = [str(item) for item in (profile_top or []) if str(item).strip()]
    return top[0][:80] if top else "unknown"


def update_descendant_gain(root: str, exemplar_ids, attempt_speedup: float) -> list:
    """Credit exemplars whose descendants reached a better speedup (P3.5)."""
    ids = {str(x) for x in (exemplar_ids or []) if str(x)}
    if not ids:
        return []
    path = os.path.join(root, EXEMPLARS_FILE)
    with _FileLock(path + ".lock"):
        rows = _read_jsonl(path)
        updated = []
        for row in rows:
            if row.get("id") not in ids:
                continue
            gain = float(attempt_speedup or 0.0) - float(row.get("speedup") or 0.0)
            row["descendant_gain"] = round(max(float(row.get("descendant_gain") or 0.0), gain), 4)
            row["updated"] = _now()
            updated.append(row)
        if updated:
            _write_jsonl(path, rows)
    return updated


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


from rsi._io import _FileLock, write_jsonl as _write_jsonl


def record_exemplar(root: str, task: str, source: str, speedup: float, hardware: str = "",
                    dtypes=None, op_class: str = "", tags=None, strategies=None,
                    profile_top=None, run_ref: str = "", device_class: str = "",
                    evidence_valid: bool = None) -> dict:
    """Store a winning solution (idempotent by content hash, one per behavior cell)."""
    path = os.path.join(root, EXEMPLARS_FILE)
    with _FileLock(path + ".lock"):
        rows = _read_jsonl(path)
        digest = hashlib.sha256((source or "").encode("utf-8")).hexdigest()[:16]
        form = _form(source)
        bottleneck = _bottleneck(profile_top)
        cell = "%s|%s" % (form, bottleneck)
        for row in rows:
            if row.get("id") == digest:
                row["speedup"] = round(max(float(row.get("speedup") or 0.0), float(speedup)), 4)
                row["form"] = form
                row["bottleneck"] = bottleneck
                row["cell"] = cell
                if device_class:
                    row["device_class"] = device_class
                if evidence_valid is not None:
                    row["evidence_valid"] = bool(evidence_valid)
                row["updated"] = _now()
                _write_jsonl(path, rows)
                return row
        row = {
            "id": digest,
            "task": task,
            "speedup": round(float(speedup), 4),
            "hardware": hardware,
            "device_class": device_class,
            "evidence_valid": evidence_valid,
            "dtypes": list(dtypes or []),
            "op_class": op_class,
            "tags": list(tags or []),
            "strategies": list(strategies or []),
            "profile_top": list(profile_top or []),
            "form": form,
            "bottleneck": bottleneck,
            "cell": cell,
            "descendant_gain": 0.0,
            "run_ref": run_ref,
            "source": (source or "")[:MAX_SOURCE_CHARS],
            "created": _now(),
            "updated": _now(),
        }
        rows.append(row)
        # Quality-diversity archive: keep the fastest example per (task, hardware,
        # form, bottleneck) cell instead of a global top-N list.
        keep, cells = [], {}
        for candidate in sorted(rows, key=lambda r: -float(r.get("speedup") or 0.0)):
            key = (candidate.get("task"), candidate.get("hardware"),
                   candidate.get("form") or "torch_fused",
                   candidate.get("bottleneck") or "unknown")
            if key in cells:
                continue
            cells[key] = candidate
            keep.append(candidate)
        _write_jsonl(path, keep)
        return next((candidate for candidate in keep if candidate.get("id") == digest), row)


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


def render_exemplars_index(exemplars: list, max_chars: int = 900) -> str:
    """Index-only exemplar view: one line per entry, source loaded on demand.

    Code exemplars are the last thing that should be inlined in the first-turn
    prompt.  The model fetches a body with {"action":"recall","id":"ex-<id>"}.
    """
    if not exemplars:
        return ""
    lines = ["## Code exemplars — index (load one with {\"action\":\"recall\",\"id\":\"ex-<id>\"})"]
    used = len(lines[0]) + 1
    for row in exemplars:
        header = "- [EX %s] task=%s form=%s speedup=%.2fx bottleneck=%s" % (
            row.get("id"), row.get("task"), row.get("form") or "unknown",
            float(row.get("speedup") or 0.0), row.get("bottleneck") or "unknown")
        if used + len(header) + 1 > max_chars:
            break
        lines.append(header)
        used += len(header) + 1
    return "\n".join(lines)


def render_exemplars(exemplars: list, max_chars: int = 6000) -> str:
    """Render a budgeted few-shot block for the task prompt."""
    if not exemplars:
        return ""
    lines = [
        "## Code exemplars from the pool (study, reuse or beat; each entry states its provenance)",
        "These solutions passed correctness; timings were measured under the conditions shown:",
    ]
    budget = max_chars
    for row in exemplars:
        header = "### %s on %s — %.2fx (form=%s, bottleneck=%s, strategies: %s)" % (
            row.get("task"), row.get("hardware") or "?", float(row.get("speedup") or 0.0),
            row.get("form") or "unknown", row.get("bottleneck") or "unknown",
            ", ".join(row.get("strategies") or []) or "-")
        if row.get("evidence_valid") is False:
            header += " [mechanics-only: measured on %s, not %s performance evidence]" % (
                row.get("device_class") or "unknown", row.get("hardware") or "the claimed hardware")
        elif row.get("device_class"):
            header += " [measured on %s]" % row["device_class"]
        if row.get("profile_top"):
            header += "\n# profile top ops: %s" % ", ".join(str(o) for o in row["profile_top"][:3])
        code = "```python\n%s\n```" % (row.get("source") or "").strip()
        block = header + "\n" + code
        if len(block) > budget:
            break
        budget -= len(block)
        lines.append(block)
    return "\n".join(lines)