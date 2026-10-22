"""Offline self-test for the skill compiler.

    python scripts/skills_selftest.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi.memory import MemoryStore, scope_of  # noqa: E402
from rsi.skills import (  # noqa: E402
    compile_skill, compile_template, render_skill, render_template, validate_skill,
)

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
        RESULTS["pass"].append(name)
        print("PASS  %s" % name)
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))


PY_SOURCE = """BLOCK = 4096

def run(x):
    return x + BLOCK
"""

CPP_SOURCE = """constexpr static int64_t TILE_ELEMS = 2048;
#define BLOCK 4096
// kernel uses BLOCK and TILE_ELEMS below
"""


def main():
    tmp = tempfile.mkdtemp(prefix="skills-selftest-")
    try:
        def python_template():
            template, params = compile_template(PY_SOURCE, [{"name": "BLOCK", "range": [1024, 8192]}])
            assert "BLOCK = {{BLOCK}}" in template, template
            assert params[0]["default"] == 4096, params
            rendered = render_template(template, {"BLOCK": 2048})
            assert "BLOCK = 2048" in rendered, rendered
        check("compile/render: python numeric assignment", python_template)

        def cpp_template():
            template, params = compile_template(
                CPP_SOURCE, [{"name": "TILE_ELEMS"}, {"name": "BLOCK", "range": [1024, 8192]}])
            assert "TILE_ELEMS = {{TILE_ELEMS}}" in template, template
            assert "#define BLOCK {{BLOCK}}" in template, template
            assert {p["name"] for p in params} == {"TILE_ELEMS", "BLOCK"}, params
        check("compile: C++ constexpr and #define", cpp_template)

        def compile_and_validate():
            store = MemoryStore(os.path.join(tmp, "kb"))
            py_file = os.path.join(tmp, "sol.py")
            with open(py_file, "w", encoding="utf-8") as fh:
                fh.write(PY_SOURCE)
            scope = scope_of({"op": "vector_add", "op_family": "elementwise",
                              "soc": "ascend-910b", "dtype": ["float32"], "form": ["python"]})
            scope.pop("task", None)
            skill = compile_skill(store, py_file, "vector_add_block", [{"name": "BLOCK"}], scope)
            assert skill["parameters"][0]["default"] == 4096, skill
            out = os.path.join(tmp, "rendered.py")
            render_skill(store, "vector_add_block", scope, {"BLOCK": 2048}, out)
            assert "BLOCK = 2048" in open(out, encoding="utf-8").read()

            checker = os.path.join(tmp, "check.py")
            with open(checker, "w", encoding="utf-8") as fh:
                fh.write(
                    "import sys\n"
                    "text = open(sys.argv[1], encoding='utf-8').read()\n"
                    "assert 'BLOCK = 2048' in text, text\n"
                )
            command = '"%s" "%s" "{file}"' % (sys.executable, checker)
            first = validate_skill(store, "vector_add_block", scope, command,
                                   {"BLOCK": 2048}, workdir=tmp)
            assert first["status"] == "pass" and first["confidence"] == "candidate", first
            second = validate_skill(store, "vector_add_block", scope, command,
                                    {"BLOCK": 2048}, workdir=tmp)
            assert second["status"] == "pass" and second["confidence"] == "verified", second
            # recompiling the same skill must not downgrade its verified status
            again = compile_skill(store, py_file, "vector_add_block", [{"name": "BLOCK"}], scope)
            assert again["confidence"] == "verified", again
        check("compile -> render -> validate twice -> verified", compile_and_validate)

        def invalid_param():
            try:
                compile_template(PY_SOURCE, [{"name": "MISSING"}])
            except Exception:
                return
            raise AssertionError("missing parameter was accepted")
        check("compile rejects a parameter absent from the source", invalid_param)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\npassed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())