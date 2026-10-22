"""Skill compiler: turn a working solution into a parameterized, renderable template.

A strategy says *what* worked; an exemplar is a frozen code snapshot.  A skill is
between them: the same code with named knobs (``BLOCK``, ``TILE_ELEMS``, ``R``)
replaced by ``{{NAME}}`` placeholders, plus preconditions and validation evidence.
That makes procedural memory reusable across shapes and tasks.

    # compile a solution/example into a skill
    python -m rsi.skills compile --file solutions/examples/swi_glu_triton.py \
        --name swiglu_triton_pair_tile --param BLOCK=4096 \
        --op swi_glu --op-family elementwise --soc ascend-910b \
        --dtype float16,float32,bfloat16 --form triton --language triton

    # render it with a different knob
    python -m rsi.skills render --name swiglu_triton_pair_tile \
        --op swi_glu --soc ascend-910b --form triton --value BLOCK=2048 --out /tmp/sol.py

    # validate the rendered file (exit 0 = pass); two passes -> verified
    python -m rsi.skills validate --name swiglu_triton_pair_tile \
        --op swi_glu --soc ascend-910b --form triton --value BLOCK=2048 \
        --command "python harness/runner.py --task tasks/level2/grouped_gemm_swiglu.py --solution {file}"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile

from .memory import (
    DEFAULT_ROOT, MemoryStore, normalize_features, scope_of,
)

PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


class SkillCompileError(ValueError):
    pass


def _find_assignment(source: str, name: str):
    """Find the first numeric assignment/definition of `name`.

    Supports Python (`BLOCK = 4096`), C/C++ (`constexpr int64_t BLOCK = 4096;`)
    and `#define BLOCK 4096`.  Returns (start, end, value) of the number or None.
    """
    escaped = re.escape(name)
    patterns = [
        r"(?m)^[ \t]*(?:constexpr[ \t]+)?(?:static[ \t]+)?(?:const[ \t]+)?"
        r"(?:int64_t|int32_t|uint32_t|size_t|int|long|auto)?[ \t]*"
        + escaped + r"[ \t]*=[ \t]*(?P<value>-?\d+)",
        r"(?m)^[ \t]*#[ \t]*define[ \t]+" + escaped + r"[ \t]+(?P<value>-?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.start("value"), match.end("value"), int(match.group("value"))
    return None


def compile_template(source: str, params) -> tuple:
    """Replace numeric definitions with {{NAME}} placeholders.

    `params` is a list of dicts: {"name": "BLOCK", "value": 4096, "range": [..]}.
    If `value` is omitted, the value found in the source is used.
    """
    template = source
    parameters = []
    for spec in params:
        if isinstance(spec, str):
            name, _, value = spec.partition("=")
            spec = {"name": name.strip(), "value": int(value) if value else None}
        name = str(spec.get("name") or "").strip()
        if not name:
            raise SkillCompileError("parameter name is required")
        found = _find_assignment(template, name)
        if not found:
            raise SkillCompileError("parameter %r has no numeric assignment in the source" % name)
        start, end, source_value = found
        value = spec.get("value")
        if value is None:
            value = source_value
        value = int(value)
        lo, hi = (spec.get("range") or [None, None])[:2]
        if lo is not None and value < int(lo):
            raise SkillCompileError("%s=%s is below range %s" % (name, value, lo))
        if hi is not None and value > int(hi):
            raise SkillCompileError("%s=%s is above range %s" % (name, value, hi))
        template = template[:start] + "{{%s}}" % name + template[end:]
        parameters.append({
            "name": name,
            "type": "int",
            "default": value,
            "range": [int(lo) if lo is not None else None,
                      int(hi) if hi is not None else None],
        })
    return template, parameters


def render_template(template: str, values: dict | None = None) -> str:
    values = dict(values or {})
    out = template
    for name in PLACEHOLDER_RE.findall(template):
        if name not in values:
            raise SkillCompileError("missing value for placeholder %r" % name)
        out = out.replace("{{%s}}" % name, str(values[name]))
    return out


def _scope_from_args(args) -> dict:
    features = {
        "op": args.op, "op_family": args.op_family, "soc": args.soc,
        "cann": args.cann, "bisheng": args.bisheng, "triton": args.triton,
        "dtype": [x.strip() for x in args.dtype.split(",") if x.strip()],
        "form": [x.strip() for x in args.form.split(",") if x.strip()],
        "task": args.task,
    }
    scope = scope_of(features)
    scope.pop("task", None)  # skills transfer across tasks
    return scope


def find_skill(store: MemoryStore, name: str, scope: dict):
    exact = [r for r in store.skills if r.get("name") == name and r.get("scope") == scope]
    if exact:
        return exact[0]
    same_name = [r for r in store.skills if r.get("name") == name]
    return same_name[0] if len(same_name) == 1 else None


def _parse_values(pairs) -> dict:
    values = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SkillCompileError("--value expects NAME=VALUE, got %r" % pair)
        name, _, value = pair.partition("=")
        values[name.strip()] = int(value)
    return values


def compile_skill(store: MemoryStore, file_path: str, name: str, params,
                  scope: dict, language: str = "", preconditions=None,
                  validation=None, measured=None) -> dict:
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    template, parameters = compile_template(source, params)
    existing = find_skill(store, name, scope)
    record = {
        "name": name,
        "language": language or ("ascendc" if file_path.endswith((".cpp", ".h")) else "python"),
        "scope": scope,
        "preconditions": list(preconditions or (existing or {}).get("preconditions") or []),
        "parameters": parameters,
        "template": template,
        "template_ref": file_path,
        "validation": dict(validation or (existing or {}).get("validation") or {}),
        "measured": dict(measured or (existing or {}).get("measured") or {}),
        "confidence": (existing or {}).get("confidence") or "candidate",
        "invalidators": list((existing or {}).get("invalidators") or []),
    }
    return store.add_skill(record)


def render_skill(store: MemoryStore, name: str, scope: dict, values: dict | None,
                 out_path: str) -> dict:
    skill = find_skill(store, name, scope)
    if not skill:
        raise SkillCompileError("skill %r not found for this scope" % name)
    template = skill.get("template") or ""
    if not template and skill.get("template_ref"):
        with open(skill["template_ref"], "r", encoding="utf-8", errors="replace") as fh:
            template = fh.read()
    rendered = render_template(template, values)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    defaults = {p["name"]: p.get("default") for p in (skill.get("parameters") or [])}
    return {"skill": skill, "rendered": out_path, "values": {**defaults, **(values or {})}}


def validate_skill(store: MemoryStore, name: str, scope: dict, command: str,
                   values: dict | None = None, workdir: str = "") -> dict:
    skill = find_skill(store, name, scope)
    if not skill:
        return {"status": "not_found", "skill": name}
    tmp = tempfile.mkdtemp(prefix="skill-validate-")
    out_path = os.path.join(tmp, os.path.basename(skill.get("template_ref") or "solution.py"))
    render_skill(store, name, scope, values, out_path)
    args = command.replace("{file}", out_path)
    proc = subprocess.run(args, cwd=workdir or None, shell=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=900)
    passed = proc.returncode == 0
    validation = dict(skill.get("validation") or {})
    runs = list(validation.get("runs") or [])
    run_id = "validate-%d" % (len(runs) + 1)
    runs.append({"run": run_id, "passed": passed, "command": args,
                 "returncode": proc.returncode})
    validation["runs"] = runs
    validation["passes"] = sum(1 for r in runs if r.get("passed"))
    validation["last"] = runs[-1]
    confidence = skill.get("confidence") or "candidate"
    if passed and validation["passes"] >= 2:
        confidence = "verified"
    elif not passed and validation["passes"] == 0 and len(runs) >= 2:
        confidence = "refuted"
    updated = dict(skill)
    updated["validation"] = validation
    updated["confidence"] = confidence
    store.add_skill(updated)
    return {"status": "pass" if passed else "fail", "skill": name,
            "confidence": confidence, "passes": validation["passes"],
            "runs": len(runs), "tail": (proc.stdout + proc.stderr)[-1200:]}


def _add_scope_args(p):
    p.add_argument("--op", default="")
    p.add_argument("--op-family", default="")
    p.add_argument("--soc", default="")
    p.add_argument("--cann", default="")
    p.add_argument("--bisheng", default="")
    p.add_argument("--triton", default="")
    p.add_argument("--dtype", default="")
    p.add_argument("--form", default="")
    p.add_argument("--task", default="")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.skills", description="Compile/render/validate operator skills.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile")
    c.add_argument("--file", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    c.add_argument("--language", default="")
    c.add_argument("--preconditions-json", default="")
    c.add_argument("--measured-json", default="")
    _add_scope_args(c)

    r = sub.add_parser("render")
    r.add_argument("--name", required=True)
    r.add_argument("--value", action="append", default=[], metavar="NAME=VALUE")
    r.add_argument("--out", required=True)
    _add_scope_args(r)

    v = sub.add_parser("validate")
    v.add_argument("--name", required=True)
    v.add_argument("--value", action="append", default=[], metavar="NAME=VALUE")
    v.add_argument("--command", required=True, help="use {file} for the rendered path")
    v.add_argument("--workdir", default="")
    _add_scope_args(v)

    l = sub.add_parser("list")

    args = ap.parse_args(argv)
    store = MemoryStore(args.root)

    if args.cmd == "compile":
        params = []
        for pair in args.param:
            name, _, value = pair.partition("=")
            params.append({"name": name.strip(), "value": int(value) if value else None})
        skill = compile_skill(
            store, args.file, args.name, params, _scope_from_args(args),
            language=args.language,
            preconditions=json.loads(args.preconditions_json) if args.preconditions_json else [],
            measured=json.loads(args.measured_json) if args.measured_json else {},
        )
        print(json.dumps({"status": "compiled", "skill": skill}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "render":
        result = render_skill(store, args.name, _scope_from_args(args),
                              _parse_values(args.value), args.out)
        print("rendered: %s" % result["rendered"])
        for name, value in result["values"].items():
            print("  %s=%s" % (name, value))
        return 0

    if args.cmd == "validate":
        result = validate_skill(store, args.name, _scope_from_args(args), args.command,
                                _parse_values(args.value), args.workdir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "pass" else 1

    if args.cmd == "list":
        for row in sorted(store.skills, key=lambda r: r.get("id", "")):
            params = ",".join(p.get("name", "") for p in (row.get("parameters") or []))
            print("%-9s %-34s %-10s params=%-20s %s" % (
                row.get("id"), row.get("name"), row.get("confidence"), params,
                json.dumps(row.get("scope") or {}, ensure_ascii=False)))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())