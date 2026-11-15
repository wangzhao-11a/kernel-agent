"""Read the CANN operator source-map catalog produced by scripts/scan_cann_ops.py.

The catalog answers a different question from the knowledge cards:

- knowledge card: what does this op/API mean, and what are its constraints?
- source map: where are the official kernel/tiling/host/API files for this op?

Both are useful.  The source map is injected read-only so the agent can inspect
official implementations without guessing paths.
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "cann_ops_catalog.json")


def _norm(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    for prefix in ("aclnn", "inplace"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("950", "910", "v2", "v3", "v4"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def load_catalog(path: str = DEFAULT_CATALOG) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def find_ops(catalog: dict, names, max_ops: int = 1) -> list:
    if not catalog:
        return []
    wanted = [_norm(n) for n in names if str(n or "").strip()]
    rows = catalog.get("ops") or []
    exact = [row for row in rows if row.get("key") in wanted]
    if exact:
        return exact[:max_ops]
    loose = []
    for row in rows:
        key = str(row.get("key") or "")
        if any(w and (w in key or key in w) for w in wanted):
            loose.append(row)
    loose.sort(key=lambda row: (len(str(row.get("key") or "")), str(row.get("name") or "")))
    return loose[:max_ops]


def _file_bullet(label: str, paths, limit: int) -> list:
    if not paths:
        return []
    rows = ["- %s:" % label]
    for path in paths[:limit]:
        rows.append("  - `%s`" % path)
    if len(paths) > limit:
        rows.append("  - … %d more" % (len(paths) - limit))
    return rows


def read_source(catalog: dict, path: str, max_chars: int = 16000) -> str:
    """Read one installed CANN source file, restricted to the catalog's impl root."""
    root = str((catalog or {}).get("impl_root") or "").strip()
    rel = str(path or "").strip().replace("\\", "/")
    if not root or not rel or ".." in rel.split("/"):
        return ""
    root_real = os.path.realpath(root)
    target = rel if os.path.isabs(rel) else os.path.join(root_real, rel)
    target_real = os.path.realpath(target)
    if target_real != root_real and not target_real.startswith(root_real + os.sep):
        return ""
    if not os.path.isfile(target_real):
        return ""
    try:
        with open(target_real, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read(max_chars + 1)
    except OSError:
        return ""
    if len(body) > max_chars:
        body = body[:max_chars] + "\n\n…(truncated)"
    return body


def grep_source(catalog: dict, pattern: str, op_names=None, max_results: int = 20,
                max_files: int = 4000, max_chars: int = 240) -> list:
    """Search installed CANN sources for a literal/regex pattern.

    Restricted to the catalog's impl root and, when op_names match, to those
    operator directories.  Returns file/line snippets for the agent to inspect
    with the `source` action.
    """
    root = str((catalog or {}).get("impl_root") or "").strip()
    text = str(pattern or "").strip()
    if not root or not text:
        return []
    root_real = os.path.realpath(root)
    roots = []
    if op_names:
        for op in find_ops(catalog, op_names, max_ops=5):
            path = str(op.get("path") or "")
            if path and os.path.isdir(path):
                roots.append(path)
    if not roots:
        roots = [root_real]
    try:
        regex = re.compile(text, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(text), re.IGNORECASE)
    rows = []
    visited = 0
    for base in roots:
        base_real = os.path.realpath(base)
        if not (base_real == root_real or base_real.startswith(root_real + os.sep)):
            continue
        for dirpath, dirs, files in os.walk(base_real):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       ("__pycache__", "build", "dist", ".git")]
            for name in files:
                if not name.endswith((".cpp", ".cc", ".cxx", ".h", ".hpp", ".py", ".json", ".md")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > 2_000_000:
                        continue
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if regex.search(line):
                                rows.append({
                                    "path": os.path.relpath(path, root_real).replace("\\", "/"),
                                    "abs_path": path,
                                    "line": lineno,
                                    "text": line.strip()[:max_chars],
                                })
                                if len(rows) >= max_results:
                                    return rows
                except OSError:
                    continue
                visited += 1
                if visited >= max_files:
                    return rows
    return rows


def render_grep(rows: list) -> str:
    if not rows:
        return ""
    lines = ["## Source grep results"]
    for row in rows:
        lines.append("- `%s:%d`  %s" % (row["path"], int(row["line"]), row["text"]))
    return "\n".join(lines)


def _relevant(paths, include_arch35: bool) -> list:
    paths = list(paths or [])
    if include_arch35:
        return paths
    clean = [path for path in paths if "arch35/" not in path.replace("\\", "/")]
    return clean


def _for_prompt(op: dict, paths, include_arch35: bool) -> list:
    base = str(op.get("path") or "")
    out = []
    for path in _relevant(paths, include_arch35):
        text = str(path).replace("\\", "/")
        if not os.path.isabs(text) and base:
            text = (base.rstrip("/") + "/" + text).replace("\\", "/")
        out.append(text)
    return out


def source_map_block(catalog: dict, spec, max_ops: int = 1, max_files: int = 8) -> str:
    if not catalog:
        return ""
    soc_text = ("%s %s" % (getattr(spec, "soc", ""), getattr(spec, "hardware", ""))).lower()
    include_arch35 = "950" in soc_text
    meta = spec.metadata or {}
    names = [meta.get("op"), getattr(spec, "id", ""),
             os.path.splitext(os.path.basename(str(getattr(spec, "path", "") or "")))[0]]
    ops = find_ops(catalog, names, max_ops=max_ops)
    if not ops:
        return ""
    lines = [
        "## CANN official source map (read-only reference)",
        "These are installed CANN implementation paths, not verified semantics. Use them to inspect",
        "the official kernel/tiling layout; confirm API behavior against the knowledge cards and compiler.",
        "Read one with {\"action\": \"source\", \"path\": \"<path>\"} (paths are absolute under the impl root).",
    ]
    for op in ops:
        lines.append("")
        lines.append("### `%s` (%s)" % (op.get("name"), op.get("category")))
        lines.append("- source: `%s`" % op.get("path"))
        lines.append("- files: %d · kernel=%s · tiling=%s · host=%s · api=%s" % (
            int(op.get("files") or 0),
            "yes" if op.get("has_kernel") else "no",
            "yes" if op.get("has_tiling") else "no",
            "yes" if op.get("has_host") else "no",
            "yes" if op.get("has_api") else "no"))
        if op.get("arch35_only"):
            lines.append("- **arch35 only**: this installed source is 950-specific; do not copy it as a 910B reference.")
        kernel_files = _for_prompt(op, op.get("kernel_files"), include_arch35)
        tiling_files = _for_prompt(op, op.get("tiling_files"), include_arch35)
        if not include_arch35 and any("arch35/" in str(p).replace("\\", "/")
                                      for p in (op.get("kernel_files") or []) + (op.get("tiling_files") or [])):
            lines.append("- arch35 files omitted for a 910B target")
        lines.extend(_file_bullet("kernel", kernel_files, max_files))
        lines.extend(_file_bullet("tiling", tiling_files, max_files))
        lines.extend(_file_bullet("api", _for_prompt(op, op.get("api_files"), include_arch35), max_files))
        cards = op.get("kb_cards") or []
        if not include_arch35:
            cards = [card for card in cards
                     if "_950" not in card and "/950/" not in card.replace("\\", "/")]
        if cards:
            lines.append("- knowledge cards:")
            for card in cards[:4]:
                lines.append("  - `%s`" % card)
    return "\n".join(lines)