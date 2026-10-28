"""Add a JSONL schema version to old evaluation/attempt records.

Phase-1 adds `schema_version` to evaluation results and agent attempts.  This
script is idempotent: records that already carry the version are untouched,
invalid JSONL lines are reported and preserved, and each changed file gets a
`.bak` copy before the atomic rewrite.

    python scripts/migrate_jsonl_schema.py --path runs/results.jsonl \
        --path "runs/agent/*/attempts.jsonl" --version 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile


def migrate_file(path: str, version: int, dry_run: bool = False) -> dict:
    report = {"path": path, "rows": 0, "changed": 0, "invalid": 0, "backup": None}
    if not os.path.isfile(path):
        report["error"] = "missing"
        return report
    rows, changed = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            report["rows"] += 1
            try:
                row = json.loads(line)
            except ValueError:
                report["invalid"] += 1
                rows.append(line.rstrip("\n"))
                continue
            if isinstance(row, dict) and "schema_version" not in row:
                row["schema_version"] = int(version)
                changed += 1
            rows.append(json.dumps(row, ensure_ascii=False))
    report["changed"] = changed
    if changed and not dry_run:
        backup = path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            report["backup"] = backup
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".schema-", dir=d, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(row + "\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="migrate_jsonl_schema")
    ap.add_argument("--path", action="append", required=True,
                    help="JSONL path or glob (repeatable)")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    paths = []
    for pattern in args.path:
        matched = glob.glob(pattern, recursive=True)
        paths.extend(matched or [pattern])
    total = {"files": 0, "rows": 0, "changed": 0, "invalid": 0}
    for path in sorted(set(paths)):
        report = migrate_file(path, args.version, dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False))
        total["files"] += 1
        for key in ("rows", "changed", "invalid"):
            total[key] += int(report.get(key) or 0)
    print("total " + json.dumps(total, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())