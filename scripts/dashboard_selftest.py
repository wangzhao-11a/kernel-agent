"""Self-test for the operator-memory dashboard build.

    python scripts/dashboard_selftest.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.build_memory_dashboard import load_data, main as build_main  # noqa: E402

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
        RESULTS["pass"].append(name)
        print("PASS  %s" % name)
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))


def main():
    data = load_data()

    def data_shape():
        assert data["summary"]["facts"] >= 1, data["summary"]
        assert data["facts"] and data["failures"] and data["skills"], data["summary"]
    check("bundle contains facts/failures/skills", data_shape)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "memory.html")

        def build():
            rc = build_main(["--out", out])
            assert rc == 0, rc
            html = open(out, encoding="utf-8").read()
            assert "Operator Memory" in html, html[:200]
            assert "__DATA__" not in html and "__GENERATED__" not in html
            match = re.search(r"const D = (\{.*?\});\s*\nconst S", html, re.S)
            assert match, "embedded JSON not found"
            payload = json.loads(match.group(1))
            assert payload["summary"]["facts"] == data["summary"]["facts"], payload["summary"]
            assert payload["summary"]["failures_open"] == data["summary"]["failures_open"]
        check("build emits self-contained HTML with embedded JSON", build)
    print("\npassed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())