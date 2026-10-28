"""Deterministic pre-build review for AscendC edits.

This is intentionally small and explainable: it catches the failure modes we have
actually seen (in-place Gather, GatherMask misuse, header-only no-ops, plan/API
mismatch, element-unit Gather offsets, arch35 leakage).  It is not a compiler.
"""
from __future__ import annotations

import difflib
import json
import re

from harness.policy import cpp_policy_findings


def _add(findings, severity, code, message):
    findings.append({"severity": severity, "code": code, "message": message})


def review_edit(files: dict, plan: dict | None, spec=None) -> dict:
    files = {str(k): str(v or "") for k, v in (files or {}).items()}
    plan = plan or {}
    findings = []
    code_text = "\n".join(files.values())
    plan_text = json.dumps(plan, ensure_ascii=False).lower()

    for name, body in files.items():
        # Gather(dst, src, ...) with dst == src corrupts the source.
        for match in re.finditer(r"\bGather\s*\(\s*([A-Za-z_]\w*)\s*,\s*\1\s*,", body):
            _add(findings, "error", "inplace_gather",
                 "%s: Gather dst and src are the same tensor `%s`; use distinct buffers."
                 % (name, match.group(1)))

    if "gathermask" in code_text.lower() and "gathermask" not in plan_text:
        _add(findings, "warn", "gathermask",
             "GatherMask is a masked-reduction API with a different signature; "
             "use Gather Level 2 unless the plan explicitly justifies GatherMask.")

    for api in (plan.get("apis") or []):
        token = str(api).strip().split()[0] if str(api).strip() else ""
        if len(token) >= 4 and token.lower() not in code_text.lower():
            _add(findings, "warn", "plan_api_missing",
                 "plan lists API `%s` but the changed code does not use it." % token)

    has_cpp = any(name.endswith((".cpp", ".cc", ".cxx")) for name in files)
    if files and not has_cpp and any(
            key in plan_text for key in ("kernel", "tiling", "gather", "pipeline", "queue")):
        _add(findings, "warn", "header_only",
             "Only headers changed, but the plan describes kernel/tiling/queue work; "
             "the implementation change is not in a .cpp file.")

    if "tiny" in plan_text and not re.search(r"tiny|gather", code_text, re.I):
        _add(findings, "warn", "tiny_missing",
             "Plan mentions a tiny-row path but the code has no tiny/Gather branch.")

    for name, body in files.items():
        if "Gather(" in body and "SetValue" in body and "sizeof" not in body:
            _add(findings, "info", "gather_offset_unit",
                 "%s: Gather offsets look like element indices; Gather expects byte offsets "
                 "(multiply by sizeof(T))." % name)

    if "arch35" in code_text.lower():
        _add(findings, "warn", "arch35",
             "Changed code references arch35 (950-specific) paths; do not use them for 910B.")

    spec_policy = getattr(spec, "policy", {}) or {}
    if spec_policy.get("forbid_native_compute"):
        for finding in cpp_policy_findings(files):
            _add(findings, finding.get("severity", "error"), finding.get("code", "policy"),
                 finding.get("message", "policy violation"))

    errors = [f for f in findings if f["severity"] == "error"]
    warnings = [f for f in findings if f["severity"] == "warn"]
    return {
        "ok": not errors,
        "findings": findings,
        "errors": errors,
        "warnings": warnings,
    }


def render_review(review: dict) -> str:
    if not review or not review.get("findings"):
        return ""
    lines = ["## Static review"]
    for item in review["findings"]:
        lines.append("- [%s] `%s`: %s" % (
            item["severity"].upper(), item["code"], item["message"]))
    if review.get("errors"):
        lines.append("BLOCKED: fix the errors above before write_files is evaluated.")
    return "\n".join(lines)


def destructive_rewrite(changed: dict, baseline: dict) -> list:
    """Reject the failure mode where a model cannot reproduce a large file and
    submits a placeholder / a tiny stub instead."""
    findings = []
    for rel, content in (changed or {}).items():
        old = str((baseline or {}).get(rel) or "")
        old_lines = old.count("\n") + (1 if old else 0)
        new_lines = str(content or "").count("\n") + (1 if content else 0)
        if old_lines <= 300:
            continue
        if "PLACEHOLDER" in str(content).upper() or new_lines < old_lines // 2:
            findings.append({
                "severity": "error",
                "code": "massive_rewrite",
                "message": ("%s has %d lines but the submitted content has %d; refusing a destructive "
                            "rewrite. Use read_file + patch_files instead." % (
                                rel, old_lines, new_lines)),
            })
    return findings


def change_set(previous: dict, current: dict, baseline: dict) -> dict:
    """Summarize effective edits and classify them into change dimensions."""
    previous = {str(k): str(v or "") for k, v in (previous or {}).items()}
    current = {str(k): str(v or "") for k, v in (current or {}).items()}
    baseline = {str(k): str(v or "") for k, v in (baseline or {}).items()}
    added = removed = 0
    dims = set()
    files = []
    for rel in sorted(set(previous) | set(current)):
        old = previous.get(rel, baseline.get(rel, ""))
        new = current.get(rel, baseline.get(rel, ""))
        if old == new:
            continue
        files.append(rel)
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
        added += sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed += sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        touched = "\n".join(diff).lower()
        path = rel.lower()
        if "tiling" in path or any(token in touched for token in
                                   ("tiling", "tile", "rows_per", "rowstile", "blocklength", "numblocks")):
            dims.add("tiling")
        if any(token in touched for token in
               ("tque", "initbuffer", "setflag", "waitflag", "pipe_", "hardevent")):
            dims.add("queue")
        if any(token in touched for token in
               ("datacopy", "gather", "cast", "exp(", "muls", "adds", "div(", "mul(", "reduce")):
            dims.add("vector")
        if "launch" in path or "plugin" in path or "runopapi" in touched:
            dims.add("launch")
        if "op_kernel" in path and "tiling" not in path:
            dims.add("kernel")
    return {
        "files": files,
        "added": added,
        "removed": removed,
        "dimensions": sorted(dims),
    }


def render_change_set(change: dict) -> str:
    if not change or not change.get("files"):
        return ""
    lines = [
        "## Change set",
        "- files: %s" % ", ".join("`%s`" % name for name in change["files"]),
        "- lines: +%d / -%d" % (int(change.get("added") or 0), int(change.get("removed") or 0)),
        "- dimensions: %s" % (", ".join(change.get("dimensions") or []) or "none"),
    ]
    if len(change.get("dimensions") or []) > 1:
        lines.append("- warning: multiple dimensions changed; prefer one dimension per experiment.")
    return "\n".join(lines)