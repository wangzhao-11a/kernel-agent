#!/usr/bin/env python3
"""Scan the installed CANN operator implementations into a source-map catalog.

The CANN installation ships ~1000 AscendC operator directories under
`opp/built-in/op_impl/ai_core/tbe/impl/ops_*/ascendc/<op>`.  This script records
where each operator's kernel/tiling/host/API files live and whether the
cannbot-knowledge bundle already has an Operator card for it.

Usage (on m18):
    python scripts/scan_cann_ops.py \
        --impl-root /usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl \
        --kb-root /data2/kernel-agent/kb/cannbot-knowledge \
        --out data/cann_ops_catalog.json \
        --markdown data/cann_ops_catalog.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

DEFAULT_IMPL = "/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl"
DEFAULT_KB = "/data2/kernel-agent/kb/cannbot-knowledge"
DEFAULT_OUT = os.path.join("data", "cann_ops_catalog.json")
DEFAULT_MD = os.path.join("data", "cann_ops_catalog.md")

KEY_SUFFIXES = (".cpp", ".h", ".hpp", ".cc", ".py", ".json")
SKIP_DIRS = {"__pycache__", ".git", "build", "dist"}


def _norm(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    for prefix in ("aclnn", "inplace"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("950", "910", "v2", "v3", "v4"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _read_frontmatter(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(3000)
    except OSError:
        return {}
    if not head.startswith("---"):
        return {}
    end = head.find("\n---", 3)
    if end < 0:
        return {}
    meta = {}
    for line in head[3:end].splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            value = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
        meta[key] = value
    return meta


def scan_kb_cards(kb_root: str) -> tuple[dict, int]:
    """Map normalized operator name -> list of Operator card paths."""
    by_name = defaultdict(list)
    total = 0
    if not kb_root or not os.path.isdir(kb_root):
        return by_name, total
    base = os.path.join(kb_root, "knowledge")
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            meta = _read_frontmatter(path)
            if str(meta.get("type") or "").strip().lower() != "operator":
                continue
            total += 1
            rel = os.path.relpath(path, kb_root).replace("\\", "/")
            stem = os.path.splitext(name)[0]
            by_name[_norm(stem)].append(rel)
    return by_name, total


def _files(op_dir: str) -> list:
    rows = []
    for root, dirs, names in os.walk(op_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(KEY_SUFFIXES):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, op_dir).replace("\\", "/")
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
                rows.append((rel, size))
    return rows


def _classify(rel: str, op_name: str) -> set:
    """Classify a flattened installed-source file (no op_kernel/op_host subdirs)."""
    low = rel.lower()
    name = low.rsplit("/", 1)[-1]
    tags = set()
    if "arch35" in low:
        tags.add("arch35")
    if name.endswith((".cpp", ".hpp", ".h", ".cc")):
        if "tiling" in name:
            tags.add("tiling")
        if ("op_kernel/" in "/" + low or "op_host/" in "/" + low
                or any(token in name for token in
                       ("_apt.", "_impl.", "_kernel.", "_single.", "_bf16.", "_f16.", "_f32."))):
            tags.add("kernel")
        if ("op_host/" in "/" + low or name.endswith("_def.cpp")
                or "host" in name):
            tags.add("host")
    if "op_api/" in "/" + low or "op_plugin/" in "/" + low or "aclnn" in name:
        tags.add("api")
    if name.endswith(".json") and ("config/" in "/" + low or "binary" in low):
        tags.add("config")
    return tags


def scan_impl(impl_root: str, kb_by_name: dict) -> list:
    ops = []
    for category in sorted(os.listdir(impl_root)):
        cat_dir = os.path.join(impl_root, category)
        asc_dir = os.path.join(cat_dir, "ascendc")
        if not category.startswith("ops_") or not os.path.isdir(asc_dir):
            continue
        for op_name in sorted(os.listdir(asc_dir)):
            op_dir = os.path.join(asc_dir, op_name)
            if not os.path.isdir(op_dir):
                continue
            files = _files(op_dir)
            classified = [(rel, _classify(rel, op_name)) for rel, _ in files]
            kernel = [rel for rel, tags in classified if "kernel" in tags]
            tiling = [rel for rel, tags in classified if "tiling" in tags]
            host = [rel for rel, tags in classified if "host" in tags]
            api = [rel for rel, tags in classified if "api" in tags]
            configs = [rel for rel, tags in classified if "config" in tags]
            arch35_kernel = [rel for rel, tags in classified if "kernel" in tags and "arch35" in tags]
            arch35_only = bool(kernel) and len(arch35_kernel) == len(kernel)
            cards = kb_by_name.get(_norm(op_name), [])
            ops.append({
                "name": op_name,
                "key": _norm(op_name),
                "category": category,
                "path": op_dir,
                "files": len(files),
                "bytes": sum(size for _, size in files),
                "has_kernel": bool(kernel),
                "has_tiling": bool(tiling),
                "has_host": bool(host),
                "has_api": bool(api),
                "arch35_only": arch35_only,
                "kernel_files": kernel[:20],
                "tiling_files": tiling[:20],
                "host_files": host[:12],
                "api_files": api[:12],
                "config_files": configs[:8],
                "kb_cards": cards,
                "in_kb": bool(cards),
            })
    return ops


def write_markdown(path: str, catalog: dict) -> None:
    ops = catalog["ops"]
    by_cat = defaultdict(list)
    for op in ops:
        by_cat[op["category"]].append(op)
    lines = [
        "# CANN operator source catalog",
        "",
        "Generated: `%s`  " % catalog["generated_at"],
        "CANN root: `%s`  " % catalog["impl_root"],
        "Knowledge root: `%s`" % catalog["kb_root"],
        "",
        "This is a **source map**, not a semantics card.  It tells an agent where the",
        "official kernel, tiling, host and API files are; use the knowledge cards for",
        "verified semantics and constraints.",
        "",
        "## Coverage",
        "",
        "| category | operators | in KB | missing |",
        "|---|---:|---:|---:|",
    ]
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        in_kb = sum(1 for row in rows if row["in_kb"])
        lines.append("| %s | %d | %d | %d |" % (cat, len(rows), in_kb, len(rows) - in_kb))
    lines.append("| **total** | **%d** | **%d** | **%d** |" % (
        len(ops), sum(1 for row in ops if row["in_kb"]),
        sum(1 for row in ops if not row["in_kb"])))
    lines.append("")
    for cat in sorted(by_cat):
        lines.append("## %s" % cat)
        lines.append("")
        lines.append("| operator | files | kernel | tiling | arch35-only | KB card |")
        lines.append("|---|---:|---|---|---|---|")
        for op in sorted(by_cat[cat], key=lambda row: row["name"]):
            card = op["kb_cards"][0] if op["kb_cards"] else ""
            lines.append("| `%s` | %d | %s | %s | %s | %s |" % (
                op["name"], op["files"],
                "yes" if op["has_kernel"] else "-",
                "yes" if op["has_tiling"] else "-",
                "yes" if op.get("arch35_only") else "-",
                ("`%s`" % card) if card else "-"))
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scan CANN operators into a source-map catalog.")
    ap.add_argument("--impl-root", default=DEFAULT_IMPL)
    ap.add_argument("--kb-root", default=DEFAULT_KB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--markdown", default=DEFAULT_MD)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.impl_root):
        raise SystemExit("impl root not found: %s" % args.impl_root)
    kb_by_name, kb_total = scan_kb_cards(args.kb_root)
    ops = scan_impl(args.impl_root, kb_by_name)
    counts = Counter(op["category"] for op in ops)
    catalog = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "impl_root": args.impl_root,
        "kb_root": args.kb_root,
        "counts": {
            "operators": len(ops),
            "kb_operator_cards": kb_total,
            "operators_with_kb_card": sum(1 for op in ops if op["in_kb"]),
            "operators_missing_kb_card": sum(1 for op in ops if not op["in_kb"]),
            "by_category": dict(sorted(counts.items())),
        },
        "ops": ops,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(catalog, fh, ensure_ascii=False, indent=1)
    write_markdown(args.markdown, catalog)
    print("operators: %d" % len(ops))
    print("with KB card: %d" % catalog["counts"]["operators_with_kb_card"])
    print("missing KB card: %d" % catalog["counts"]["operators_missing_kb_card"])
    for cat, n in sorted(counts.items()):
        print("  %-18s %d" % (cat, n))
    print("json: %s" % args.out)
    print("markdown: %s" % args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())