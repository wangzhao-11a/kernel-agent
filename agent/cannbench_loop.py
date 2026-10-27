"""AscendC/CannBench agent loop: edit a scaffold, build, run official cases, iterate.

This is the AscendC sibling of `agent.loop`.  It keeps the same contract:

    write files -> build/test -> structured feedback -> write again -> finish

but the "solution" is a set of edits to a CannBench package (`csrc/...`,
`cann_bench/...`) instead of one Python file.  `harness.cannbench` owns the
build/test/bench execution and parsing; this module only talks to the model and
records the trajectory.

Offline demo (no key, no NPU, uses a mock spec from scripts/cannbench_selftest.py):

    python -m agent.cannbench_loop --spec <mock_spec.json> --provider replay \
        --replay-file <replay.json> --device-label cpu
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import llm as llm_mod  # noqa: E402
from agent.review import (  # noqa: E402
    change_set, destructive_rewrite, render_change_set, render_review, review_edit,
)
from harness.cannbench import (  # noqa: E402
    CannBenchError, format_profile, format_summary, load_spec, run_cannbench, run_profile,
)
from rsi.cards import DEFAULT_KB_ROOT as DEFAULT_CANN_KB_ROOT  # noqa: E402
from rsi.cards import read_card, render_candidates, retrieve_cards, search_cards  # noqa: E402
from rsi.cann_ops import DEFAULT_CATALOG as DEFAULT_OPS_CATALOG  # noqa: E402
from rsi.cann_ops import (  # noqa: E402
    find_ops, grep_source, load_catalog, read_source, render_grep, source_map_block,
)
from rsi.distill import distill_run  # noqa: E402
from rsi.memory import MemoryStore  # noqa: E402
from rsi.provenance import provenance  # noqa: E402
from rsi.retrieve import combos_for, priors_block, retrieve  # noqa: E402
from rsi.store import DEFAULT_ROOT as DEFAULT_KNOWLEDGE_ROOT  # noqa: E402
from rsi.store import StrategyStore  # noqa: E402

SYSTEM = """You are a senior AscendC kernel engineer working as an autonomous agent.
You edit a fixed CannBench scaffold, build a wheel, and run the official cases.
Your goal is to make every official case pass and, when a benchmark is available,
beat the vendor baseline without changing semantics.

## Hard AscendC constraints (measured on 910B/950; violating them gives errcode 264/95 or a device fault)
- Host launch: `at_npu::native::OpCommand::RunOpApi("Name", [=]() -> int {...})`; the lambda must
  capture by value. Do not call the launch function outside RunOpApi.
- UB -> GM stores must go through a VECOUT queue (`TQue<QuePosition::VECOUT, N>`); do not write
  `GlobalTensor` from a VECCALC tensor directly.
- Never read or write GM scalars from the kernel (errcode 264/95). Row/stride arithmetic must be
  scalar per program or host-side; per-element integer division is also very slow.
- 16bit -> float cast uses `CAST_NONE`; float -> 16bit uses `CAST_RINT`. Match the golden's fp32
  accumulation and cast-on-store semantics.
- Vector base addresses must be 32B aligned. `DataCopyPad` with blockCount > 1 requires the
  per-block byte length to be a multiple of 32B; otherwise copy one row at a time.
- int8 data cannot use the UB block-copy path; `Compare` (tensor-tensor) is not verified.
- On 950PR, short non-aligned rows are usually faster with the SIMT path; long rows use the
  tile/DMA path. Do not assume a 910B kernel is correct or fast on 950.
- The submission package must not use torch/torch_npu compute APIs, CPU fallbacks, or non-contiguous
  outputs. The canonical compute must be in the AscendC kernel.

## Response protocol (STRICT)
Reply with exactly ONE JSON object, nothing outside it:
  {"action": "read_file", "path": "<package file>", "start": 1, "end": 200}
      Read a line range of a package file (read-only; use it when Current contents is truncated).
  {"action": "patch_files",
   "patches": [{"path": "<allowed file>", "find": "<exact text>", "replace": "<new text>"}]}
      Apply exact single-occurrence replacements. Prefer this over rewriting a large file.
  {"action": "plan",
   "layout": "<outer/half/inner, contiguity, alignment, dtype path>",
   "tiling": {"rows_per_tile": 0, "cols_per_tile": 0, "ub_budget": "...", "cores": 0},
   "queues": ["VECIN x2", "VECCALC x2", "VECOUT x2"],
   "apis": ["DataCopyPad", "Gather", "Cast"],
   "sync": ["V_MTE2", "MTE3_V"],
   "risks": ["Gather offsets are bytes", "arch35 source is 950-only"],
   "expected_bottleneck": "MTE2 | V | MTE3 | launch | scalar"}
      Required before the first write_files unless --no-require-plan is set.
  {"action": "write_files",
   "files": {"<relative path>": "<complete file content>"},
   "strategies": ["<names of practices you applied>"],
   "expected_speedup": 1.15}
      Only paths listed in ALLOWED FILES may be edited. Each value is the FULL new file.
  {"action": "run_eval"}
      Rebuild and rerun the current files unchanged.
  {"action": "kb_search", "query": "<free-text query>", "top_k": 4}
      Search the CANN knowledge base yourself when the preloaded cards are not enough.
  {"action": "knowledge", "id": "<card path from the candidates block>"}
      Read one CANN knowledge card before guessing an API signature or hardware constraint.
  {"action": "source", "path": "<kernel/tiling/api path from the source map>"}
      Read one installed CANN source file (read-only, restricted to the CANN impl root).
  {"action": "source_grep", "pattern": "<regex or literal>", "op": "<optional op>"}
      Search official CANN sources for a pattern and get file:line snippets.
  {"action": "reflect", "lesson": "<what you learned>", "next": "<the one next change>"}
      Record a lesson after a result; use it to choose the next experiment.
  {"action": "profile", "shape": "1000003,2", "dtype": "bfloat16", "dim": 1}
      Profile one shape and return VEC/scalar/MTE2/MTE3 ratios plus the bottleneck.
  {"action": "finish", "summary": "<final approach, case pass count, speedup, caveats>"}

## Strategy
1. Read the current kernel and the case table. Identify whether the failure is build, correctness,
   or performance before changing code.
2. Before writing code, produce a plan: layout, tiling, queues, APIs, sync, risks and the expected
   bottleneck. Do not skip this design step.
3. If "Current contents" is truncated, use `read_file` with line ranges before editing.
4. Prefer `patch_files` (exact find/replace) for small changes in large files; `write_files` is
   only safe when you can reproduce the complete file.
5. Use the CANN knowledge cards: read the candidate card before relying on an API signature,
   byte-offset convention, tiling rule, or architecture restriction. Do not invent API semantics.
   If the preloaded cards are insufficient, call `kb_search` with a concrete query, then
   `knowledge` to read a hit.
6. Inspect official implementations with `source`/`source_grep` when the KB does not answer how a
   tiling/queue/API pattern is actually used.
7. Change one thing at a time: addressing/tiling, dtype path, queue/buffer usage, or launch/tiling
   parameters. Keep a correct fallback path for cases your fast path cannot handle.
8. After a pass/fail, use `reflect` to record the lesson and the single next change.
9. When speedup plateaus, call `profile` for the worst shape. Use the bottleneck (MTE2/MTE3/V/scalar)
   to choose the next single change instead of guessing.
10. Correctness covers all official cases, not just the first shape. Watch prime/odd sizes, arbitrary
    `dim`, fp16/bf16/fp32, and inf/NaN inputs.
11. If the build fails, read the first compiler error, fix it, and do not reformat unrelated code.
12. Never edit the harness, tests, build scripts, or files outside ALLOWED FILES.
"""

TASK = """# CannBench task: {id}
{description}

- package: `{package}`
- soc: `{soc}` | hardware label: `{hardware}` | baseline: `{baseline}`
- build: `{build}`
- test: `{test}`
- bench: `{bench}`

## ALLOWED FILES (edit only these)
{allowed}

## Current contents
{contents}

{priors}

{memory}

{skills}

{knowledge}

{source_map}

## Current state
{state}
{plan_rule}

Evaluation budget: up to {max_iters} evaluations (build+test+bench). Plan / knowledge / source /
reflect turns do not consume it; total model turns are capped at {max_turns}.
Reply with exactly one JSON action.
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(path: str, cap: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[:cap]
    except OSError:
        return ""


def _resolve_template(ref: str, spec) -> str:
    """Resolve a skill template path against the project root, spec dir, or package."""
    ref = str(ref or "").strip()
    if not ref:
        return ""
    candidates = []
    if os.path.isabs(ref):
        candidates.append(ref)
    else:
        candidates.extend([
            os.path.join(ROOT, ref),
            os.path.join(os.path.dirname(spec.path), ref),
            os.path.join(spec.package, ref),
        ])
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return ""


def _skill_templates_block(mem, feats: dict, spec, max_chars: int = 16000) -> str:
    """Read matched skill templates into the prompt (the context only shows template_ref)."""
    if not mem or max_chars <= 0:
        return ""
    blocks = []
    used = 0
    for hit in mem.retrieve(feats, kinds=("skill",)):
        record = hit["record"]
        path = _resolve_template(record.get("template_ref") or "", spec)
        if not path:
            continue
        try:
            rel = os.path.relpath(path, spec.package).replace("\\", "/")
        except ValueError:
            rel = ""
        if rel in spec.edit_files:
            continue   # already present in "Current contents"
        body = _read(path, cap=min(12000, max_chars))
        if not body.strip():
            continue
        head = "## Skill template: %s (%s, %s)" % (
            record.get("name") or os.path.basename(path), hit["id"],
            record.get("confidence") or "candidate")
        if record.get("measured"):
            head += "\nmeasured: " + json.dumps(record["measured"], ensure_ascii=False)
        if record.get("preconditions"):
            head += "\npreconditions: " + "; ".join(str(x) for x in record["preconditions"])
        lang = "python" if path.lower().endswith(".py") else "cpp"
        block = head + "\n```%s\n" % lang + body.rstrip() + "\n```"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _detect_device(explicit: str = "") -> str:
    if explicit:
        return explicit
    try:
        import torch_npu  # noqa: F401
        return "npu"
    except Exception:
        return "cpu"


def _parse_action(text: str):
    if not text:
        return None, "empty response"
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    try:
        obj = json.loads(s)
    except ValueError:
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return None, "no JSON object found"
        try:
            obj = json.loads(s[start:end + 1])
        except ValueError as exc:
            return None, "invalid JSON: %s" % exc
    if not isinstance(obj, dict) or "action" not in obj:
        return None, "reply is not a JSON action object"
    return obj, None


def _error_signature(result: dict) -> str:
    """Stable signature for the first compiler/runtime error of a failed evaluation."""
    status = str(result.get("status") or "")
    if not status or status == "pass":
        return ""
    for stage in ("build", "test", "bench"):
        record = result.get(stage) or {}
        if record.get("returncode") in (0, None):
            continue
        tail = record.get("tail") or ""
        for line in tail.splitlines():
            low = line.lower()
            if "error:" in low or "fatal error" in low:
                return status + "|" + line.strip()[:220]
        if tail.strip():
            return status + "|" + tail.strip().splitlines()[-1][:220]
    return status + "|" + str(result.get("error") or "")


def _kb_platforms(spec) -> tuple:
    text = ("%s %s" % (spec.hardware or "", spec.soc or "")).lower()
    if "950" in text:
        return ("a5",)
    if "310" in text:
        return ("a2",)
    return ("a2", "a3")


def _features(spec) -> dict:
    meta = spec.metadata or {}
    tags = ["cannbench", "ascendc"] + [str(t) for t in (meta.get("tags") or [])]
    op = str(meta.get("op") or "")
    if not op and "swi_glu" in spec.id.lower():
        op = "swi_glu"
    return {
        "name": spec.id,
        "task": spec.id,
        "op": op,
        "op_class": str(meta.get("op_class") or ""),
        "op_family": str(meta.get("op_class") or ""),
        "soc": spec.hardware or spec.soc,
        "hardware": spec.hardware,
        "form": ["ascendc"],
        "dtypes": [str(d).lower() for d in (meta.get("dtypes") or [])],
        "dtype": [str(d).lower() for d in (meta.get("dtypes") or [])],
        "tags": sorted(set(tags)),
    }


def _content_block(spec, files: dict) -> str:
    blocks = []
    for rel in spec.edit_files:
        source = files.get(rel)
        if source is None:
            source = _read(os.path.join(spec.package, rel))
        blocks.append("### %s\n```cpp\n%s\n```" % (rel, source.rstrip()))
    return "\n\n".join(blocks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.cannbench_loop",
                                 description="AscendC/CannBench agent loop: edit -> build -> official cases.")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-iters", type=int, default=8,
                    help="maximum evaluations (build+test+bench); plan/research/reflect turns are free")
    ap.add_argument("--max-turns", type=int, default=0,
                    help="maximum model turns (0 = max-iters + 8)")
    ap.add_argument("--max-repeat-errors", type=int, default=3,
                    help="stop after N attempts with the same first error (0 disables)")
    ap.add_argument("--target-speedup", type=float, default=1.0)
    ap.add_argument("--replay-file", default="")
    ap.add_argument("--device-label", default="")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "agent"))
    ap.add_argument("--knowledge-root", default=DEFAULT_KNOWLEDGE_ROOT)
    ap.add_argument("--memory-root", default="",
                    help="facts/failures/skills root (default: same as --knowledge-root)")
    ap.add_argument("--kb-root", default=DEFAULT_CANN_KB_ROOT,
                    help="CANN knowledge-card root (cannbot-knowledge)")
    ap.add_argument("--kb-cards", type=int, default=4,
                    help="max CANN knowledge-card candidates in the first prompt (0 disables)")
    ap.add_argument("--kb-body-chars", type=int, default=2500,
                    help="max chars when auto-reading the top card or an agent-requested card")
    ap.add_argument("--no-kb-auto", action="store_true",
                    help="show candidates but do not auto-read the top card body")
    ap.add_argument("--ops-catalog", default=DEFAULT_OPS_CATALOG,
                    help="CANN operator source-map catalog (scripts/scan_cann_ops.py)")
    ap.add_argument("--no-source-map", action="store_true",
                    help="do not inject the installed CANN source map for this operator")
    ap.add_argument("--source-chars", type=int, default=12000,
                    help="max chars returned by the source action")
    ap.add_argument("--no-rsi", action="store_true")
    ap.add_argument("--no-require-plan", action="store_true",
                    help="allow write_files without a preceding plan action")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the deterministic pre-build static review")
    ap.add_argument("--no-rollback", action="store_true",
                    help="keep a failed/lower-speedup edit instead of restoring the best passing files")
    ap.add_argument("--no-skill-templates", action="store_true",
                    help="do not inline matched skill template files into the task prompt")
    ap.add_argument("--max-skill-chars", type=int, default=16000,
                    help="max total characters of inlined skill templates (0 disables)")
    ap.add_argument("--distill", action="store_true",
                    help="update the strategy pool from this run (strict provenance: CPU mechanics "
                         "runs will be skipped)")
    ap.add_argument("--promote-skill", default="",
                    help="promote a passing run into a skill with this name (requires real-device provenance)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    baseline_files = {}
    for rel in spec.edit_files:
        try:
            with open(os.path.join(spec.package, rel), "r", encoding="utf-8",
                      errors="replace") as fh:
                baseline_files[rel] = fh.read()
        except OSError:
            baseline_files[rel] = ""
    device = _detect_device(args.device_label)
    feats = _features(spec)
    prov = provenance({"device": device, "hardware": spec.hardware})
    mem_feats = dict(feats)
    if not prov.get("valid"):
        # mechanics-only run: do not file its failures under the claimed 950/910B scope
        mem_feats["soc"] = "cpu"
        mem_feats["hardware"] = "cpu"
    store = StrategyStore(args.knowledge_root) if not args.no_rsi else None
    mem = MemoryStore(args.memory_root or args.knowledge_root) if not args.no_rsi else None
    matched = retrieve(store, feats, spec.hardware or device, top_k=5) if store else []
    combos = combos_for(store, [s["id"] for s, _score, _match in matched]) if store else []
    priors = priors_block(feats, spec.hardware or device, matched, combos) if store else ""
    memory_block = mem.context(feats, kinds=("fact", "failure", "skill")) if mem else ""
    skill_block = ""
    if mem and not args.no_skill_templates:
        skill_block = _skill_templates_block(mem, feats, spec, args.max_skill_chars)
    kb_cards = []
    kb_block = ""
    if args.kb_cards > 0:
        kb_cards = retrieve_cards(args.kb_root, feats, top_k=args.kb_cards,
                                  platforms=_kb_platforms(spec), required_form="ascendc")
        kb_block = render_candidates(kb_cards)
        if kb_cards and not args.no_kb_auto:
            body = read_card(args.kb_root, kb_cards[0]["id"], max_chars=args.kb_body_chars)
            if body:
                kb_block += "\n\n### Auto-loaded top card: %s\n%s" % (kb_cards[0]["id"], body)
    if not kb_block:
        kb_block = "## CANN knowledge base\n(no cards matched; rely on the scaffold and compiler feedback)"
    ops_catalog = {} if args.no_source_map else load_catalog(args.ops_catalog)
    source_block = source_map_block(ops_catalog, spec) if ops_catalog else ""

    api_key = llm_mod.resolve_api_key(args.provider, explicit=args.api_key,
                                      env_name=args.api_key_env)
    if args.provider in ("openai", "anthropic") and not api_key:
        print("ERROR: no API key found. Set KERNEL_AGENT_API_KEY or pass --api-key.")
        return 2
    replayer = None
    if args.provider == "replay":
        if not args.replay_file:
            print("ERROR: --provider replay needs --replay-file")
            return 2
        with open(args.replay_file, "r", encoding="utf-8") as fh:
            replayer = llm_mod.ReplayModel(json.load(fh).get("responses", []))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, "%s_%s" % (stamp, re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.id)))
    os.makedirs(run_dir, exist_ok=True)
    transcript = open(os.path.join(run_dir, "transcript.jsonl"), "a", encoding="utf-8")
    attempts = open(os.path.join(run_dir, "attempts.jsonl"), "a", encoding="utf-8")

    def log(role: str, content: str, meta: dict = None) -> None:
        row = {"ts": _now(), "role": role, "content": content}
        if meta:
            row.update(meta)
        transcript.write(json.dumps(row, ensure_ascii=False) + "\n")
        transcript.flush()

    sys_prompt = SYSTEM
    max_turns = args.max_turns if args.max_turns > 0 else args.max_iters + 8
    first = TASK.format(
        id=spec.id, description=spec.description or "(no description)",
        package=spec.package, soc=spec.soc or "?", hardware=spec.hardware or "?",
        baseline=spec.baseline or "?", build=" ".join(_as_list(spec.build_command)),
        test=" ".join(_as_list(spec.test_command)),
        bench=" ".join(_as_list(spec.bench_command)) if spec.bench_command else "(none)",
        allowed="\n".join("- `%s`" % rel for rel in spec.edit_files),
        contents=_content_block(spec, {}),
        priors=priors or "## Tuning priors\n(none retrieved for this task)",
        memory=memory_block or "## Operator memory\n(disabled)",
        skills=skill_block or "## Skill templates\n(none matched)",
        knowledge=kb_block,
        source_map=source_block or "## CANN official source map\n(no source-map entry for this operator)",
        state=("No edits applied yet; the scaffold is the baseline."
               + (" First reply with a plan action; write_files is rejected until the plan is accepted."
                  if not args.no_require_plan else "")),
        plan_rule=("PLAN REQUIRED: reply with a plan action before write_files."
                   if not args.no_require_plan else "PLAN OPTIONAL."),
        max_iters=args.max_iters,
        max_turns=max_turns,
    )
    log("system", sys_prompt)
    log("user", first)
    messages = [{"role": "user", "content": first}]

    files = {}
    best = {"status": None, "speedup": 0.0, "files": None}
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    iteration = 0
    eval_count = 0
    require_plan = not args.no_require_plan
    plan_data = {}
    plan_ready = False
    reviews = []
    rollbacks = []
    reflections = []
    profiles = []
    last_result = {}
    rollback_note = ""
    repeat_sig = ""
    repeat_count = 0
    repeat_stop = False

    def note_repeat(result: dict, message: str) -> str:
        nonlocal repeat_sig, repeat_count, repeat_stop
        sig = _error_signature(result)
        if not sig:
            repeat_sig, repeat_count = "", 0
            return message
        if sig == repeat_sig:
            repeat_count += 1
        else:
            repeat_sig, repeat_count = sig, 1
        if args.max_repeat_errors and repeat_count >= args.max_repeat_errors:
            repeat_stop = True
            message += ("\n\nREPEATED FAILURE (%d/%d): the same error recurred; stopping instead of "
                        "resending it. Use a different API/type/approach." % (
                            repeat_count, args.max_repeat_errors))
        elif repeat_count >= 2:
            message += ("\n\nREPEATED FAILURE (%d/%d): the same error recurred. Do not resend the same "
                        "code; change the API call or element types." % (
                            repeat_count, args.max_repeat_errors or "?"))
        return message

    def evaluate(it: int, declared, expected, change=None):
        nonlocal files, rollback_note, rollbacks, last_result, eval_count
        rollback_note = ""
        last_result = {}
        result = run_cannbench(spec, files=files, work_root=os.path.join(run_dir, "eval-%02d" % it),
                               run_id=os.path.basename(run_dir))
        eval_count += 1
        speedup = (result.get("timing") or {}).get("speedup")
        status = result.get("status")
        last_result = {"status": status, "speedup": speedup, "error": result.get("error")}
        row = {
            "iter": it, "task": spec.id, "spec": spec.path,
            "status": status, "speedup": speedup,
            "error": result.get("error"),
            "expected_speedup": expected,
            "strategies_declared": list(declared or []),
            "strategies_used": list(declared or []),
            "change_set": change or {},
            "files": sorted(files.keys()),
            "workspace": result.get("workspace"),
            "case_summary": result.get("case_summary"),
            "build": {k: (result.get("build") or {}).get(k) for k in ("returncode", "duration_s", "log_path")},
            "test": {k: (result.get("test") or {}).get(k) for k in ("returncode", "duration_s", "log_path")},
            "timestamp": _now(),
            "distilled": False,
        }
        attempts.write(json.dumps(row, ensure_ascii=False) + "\n")
        attempts.flush()
        log("harness", format_summary(result), meta={"iteration": it, "status": status})
        if mem:
            if status == "pass":
                mem.resolve_failures(mem_feats, os.path.basename(run_dir), "official cases pass")
            else:
                mem.record_failure_from_result(result, mem_feats, os.path.basename(run_dir), spec.id)
        had_pass = best.get("status") == "pass"
        if status == "pass":
            if speedup is None or (speedup or 0) >= (best.get("speedup") or 0):
                best.update(status=status, speedup=speedup or 0.0, files=dict(files),
                            case_summary=result.get("case_summary"))
            elif best["status"] is None:
                best.update(status=status, speedup=speedup or 0.0, files=dict(files),
                            case_summary=result.get("case_summary"))
            elif had_pass and not args.no_rollback:
                files = dict(best.get("files") or {})
                rollback_note = ("ROLLBACK: candidate passed but regressed (%.3fx < best %.3fx); "
                                 "restored the best passing files." % (
                                     speedup or 0.0, best.get("speedup") or 0.0))
        else:
            if best["status"] is None:
                best.update(status=status, speedup=speedup or 0.0, files=dict(files),
                            case_summary=result.get("case_summary"))
            if not args.no_rollback:
                if had_pass:
                    files = dict(best.get("files") or {})
                    rollback_note = ("ROLLBACK: candidate failed (%s); restored the best passing files."
                                     % status)
                else:
                    files = {}
                    rollback_note = ("ROLLBACK: first attempt failed (%s); restored the clean scaffold."
                                     % status)
        if rollback_note:
            rollbacks.append({"iter": it, "note": rollback_note, "files": sorted(files.keys())})
            log("rollback", rollback_note, meta={"iteration": it, "files": sorted(files.keys())})
        return result

    for iteration in range(1, max_turns + 1):
        try:
            reply = llm_mod.chat(
                messages, system=sys_prompt, provider=args.provider, model=args.model,
                base_url=args.base_url, api_key=api_key, temperature=args.temperature,
                max_tokens=args.max_tokens, replay=replayer,
            )
        except llm_mod.LLMError as exc:
            log("system", "LLM error: %s" % exc)
            print("LLM error: %s" % exc)
            break
        text = reply.get("text", "")
        usage = reply.get("usage") or {}
        for key in usage_total:
            usage_total[key] += int(usage.get(key, 0) or 0)
        messages.append({"role": "assistant", "content": text})
        log("assistant", text, meta={"iteration": iteration})

        action, err = _parse_action(text)
        if err:
            message = "Reply with exactly one JSON action object (%s)." % err
            messages.append({"role": "user", "content": message})
            log("user", message)
            continue
        kind = str(action.get("action", "")).lower()

        def current_content(rel: str) -> str:
            return files.get(rel, baseline_files.get(rel, ""))

        if kind == "read_file":
            rel = str(action.get("path") or "").replace("\\", "/").strip("/")
            if (not rel or rel.startswith("../") or "/../" in rel or os.path.isabs(rel)
                    or not os.path.isfile(os.path.join(spec.package, rel))):
                message = "read_file path must be a package-relative file."
            else:
                if rel in spec.edit_files:
                    content = current_content(rel)
                else:
                    try:
                        with open(os.path.join(spec.package, rel), "r",
                                  encoding="utf-8", errors="replace") as fh:
                            content = fh.read()
                    except OSError:
                        content = ""
                lines = content.splitlines()
                try:
                    start = max(1, int(action.get("start") or 1))
                    end = int(action.get("end") or (start + 199))
                except (TypeError, ValueError):
                    start, end = 1, 200
                if start > len(lines):
                    start = max(1, len(lines))
                end = min(len(lines), max(start, end), start + 399)
                if not lines:
                    message = "FILE %s is empty." % rel
                else:
                    body = "\n".join("%6d\t%s" % (i, lines[i - 1])
                                     for i in range(start, end + 1))
                    log("read_file", "%s lines %d-%d/%d" % (rel, start, end, len(lines)),
                        meta={"iteration": iteration, "path": rel})
                    message = ("FILE %s lines %d-%d/%d\n%s\n\nUse patch_files or write_files next."
                               % (rel, start, end, len(lines), body))
        elif kind == "patch_files":
            patches = action.get("patches")
            if not isinstance(patches, list) or not patches:
                message = "patch_files needs a non-empty 'patches' list."
            else:
                candidate = dict(files)
                touched = set()
                patch_error = ""
                for patch in patches:
                    if not isinstance(patch, dict):
                        patch_error = "each patch must be an object"
                        break
                    rel = str(patch.get("path") or "").replace("\\", "/").strip("/")
                    find = patch.get("find") if "find" in patch else patch.get("old")
                    replace = patch.get("replace") if "replace" in patch else patch.get("new", "")
                    if rel not in spec.edit_files:
                        patch_error = "path not allowed: %s" % rel
                        break
                    if not isinstance(find, str) or not find:
                        patch_error = "patch.find must be a non-empty string"
                        break
                    current = candidate.get(rel, baseline_files.get(rel, ""))
                    count = current.count(find)
                    if count != 1:
                        patch_error = ("find text occurs %d times in %s (must be exactly 1)"
                                       % (count, rel))
                        break
                    candidate[rel] = current.replace(find, str(replace), 1)
                    touched.add(rel)
                if patch_error:
                    message = "PATCH REJECTED: %s" % patch_error
                else:
                    new_map = {rel: candidate.get(rel, baseline_files.get(rel, ""))
                               for rel in touched}
                    action = dict(action)
                    action["files"] = new_map
                    kind = "write_files"

        if kind == "plan":
            required = ("layout", "tiling", "apis", "risks")
            plan_data = {key: action.get(key) for key in
                         ("layout", "tiling", "queues", "apis", "sync", "risks",
                          "expected_bottleneck")}
            missing = [key for key in required if not plan_data.get(key)]
            if missing:
                message = ("plan is missing %s. Include layout, tiling {rows_per_tile, cols_per_tile, "
                           "ub_budget, cores}, queues, apis, sync, risks, expected_bottleneck."
                           % ", ".join(missing))
            else:
                plan_ready = True
                log("plan", json.dumps(plan_data, ensure_ascii=False),
                    meta={"iteration": iteration})
                message = ("PLAN ACCEPTED. Now reply with write_files using this design; keep the "
                           "first change small and preserve a correct fallback.\n"
                           + json.dumps(plan_data, ensure_ascii=False, indent=2))
        elif kind == "kb_search":
            query = str(action.get("query") or "").strip()
            try:
                top_k = max(1, min(8, int(action.get("top_k") or 4)))
            except (TypeError, ValueError):
                top_k = 4
            if not query:
                message = "kb_search needs a non-empty \"query\"."
            else:
                hits = search_cards(args.kb_root, query, top_k=top_k,
                                    platforms=_kb_platforms(spec))
                log("kb_search", "query=%s hits=%d" % (query, len(hits)),
                    meta={"iteration": iteration, "query": query,
                          "hits": [c["id"] for c in hits]})
                if hits:
                    message = ("KB SEARCH %r\n%s\n\nUse knowledge with an id above if you need the body."
                               % (query, render_candidates(hits)))
                else:
                    message = ("KB SEARCH %r returned no cards. Try a shorter query or use source_grep."
                               % query)
        elif kind == "source_grep":
            pattern = str(action.get("pattern") or "").strip()
            op_name = str(action.get("op") or (spec.metadata or {}).get("op") or spec.id)
            if not ops_catalog:
                message = "source_grep is unavailable: no CANN ops catalog is loaded."
            elif not pattern:
                message = "source_grep needs a non-empty \"pattern\"."
            else:
                rows = grep_source(ops_catalog, pattern, op_names=[op_name],
                                   max_results=20)
                log("source_grep", "pattern=%s hits=%d" % (pattern, len(rows)),
                    meta={"iteration": iteration, "pattern": pattern,
                          "hits": [r["path"] for r in rows[:10]]})
                message = (render_grep(rows) or
                           ("SOURCE GREP %r returned no hits under %s."
                            % (pattern, op_name)))
                if rows:
                    message += "\n\nRead a file with source if you need more context."
        elif kind == "reflect":
            lesson = str(action.get("lesson") or "").strip()
            if not lesson:
                message = "reflect needs a non-empty \"lesson\"."
            else:
                record = {"iteration": iteration, "lesson": lesson,
                          "next": str(action.get("next") or ""),
                          "last_result": last_result}
                reflections.append(record)
                with open(os.path.join(run_dir, "reflections.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                log("reflect", lesson, meta={"iteration": iteration,
                                             "next": record["next"]})
                message = ("REFLECTION RECORDED. Use this lesson to pick the next single change:\n"
                           + json.dumps(record, ensure_ascii=False, indent=2))
        elif kind == "profile":
            if not spec.profile_command:
                message = "profile is unavailable: this spec has no profile_command."
            else:
                env = {}
                for key, env_name in (("shape", "PROFILE_SHAPE"), ("dtype", "PROFILE_DTYPE"),
                                      ("dim", "PROFILE_DIM"), ("iters", "PROFILE_ITERS")):
                    if action.get(key) is not None:
                        env[env_name] = str(action[key])
                result = run_profile(
                    spec, files=files,
                    work_root=os.path.join(run_dir, "profile-%02d" % iteration),
                    run_id=os.path.basename(run_dir), env=env)
                profiles.append({"iteration": iteration, "env": env,
                                 "status": result.get("status"),
                                 "metrics": result.get("metrics") or {},
                                 "error": result.get("error")})
                log("profile", format_profile(result), meta={
                    "iteration": iteration, "status": result.get("status"),
                    "metrics": result.get("metrics") or {}})
                message = format_profile(result) + "\n\nUse one bottleneck to choose the next change."
        elif kind == "write_files":
            if require_plan and not plan_ready:
                message = ("A plan is required before write_files. First reply with "
                           "{\"action\":\"plan\", ...} covering layout, tiling, queues, apis, "
                           "sync, risks and expected_bottleneck.")
            else:
                new_files = action.get("files")
                if not isinstance(new_files, dict) or not new_files:
                    message = "write_files needs a non-empty 'files' object."
                else:
                    bad = [rel for rel in new_files if str(rel).replace("\\", "/").strip("/") not in spec.edit_files]
                    if bad:
                        message = ("Only these files may be edited: %s. Rejected: %s"
                                   % (spec.edit_files, bad))
                    else:
                        new_map = {str(rel).replace("\\", "/").strip("/"): str(content)
                                   for rel, content in new_files.items()}
                        candidate = dict(files)
                        touched = set()
                        for rel, content in new_map.items():
                            current = candidate.get(rel, baseline_files.get(rel, ""))
                            if content == current:
                                continue
                            touched.add(rel)
                            if content == baseline_files.get(rel, ""):
                                candidate.pop(rel, None)   # explicit revert to baseline
                            else:
                                candidate[rel] = content
                        if not touched:
                            message = ("write_files produced no content changes: every submitted "
                                       "file matches the current workspace. Make one substantive "
                                       "edit that matches the plan, or use run_eval/finish.")
                        else:
                            review_map = {rel: candidate.get(rel, baseline_files.get(rel, ""))
                                          for rel in touched}
                            cs = change_set(files, candidate, baseline_files)
                            review = (review_edit(review_map, plan_data, spec)
                                      if not args.no_review else
                                      {"ok": True, "findings": [], "errors": [], "warnings": []})
                            for item in destructive_rewrite(review_map, baseline_files):
                                review["findings"].append(item)
                                if item["severity"] == "error":
                                    review["errors"].append(item)
                                else:
                                    review["warnings"].append(item)
                            review["ok"] = not review.get("errors")
                            if len(cs.get("dimensions") or []) > 1:
                                item = {"severity": "warn", "code": "multi_dimension",
                                        "message": ("changed %s in one step; prefer one dimension "
                                                    "per experiment." % ", ".join(cs["dimensions"]))}
                                review["findings"].append(item)
                                review["warnings"].append(item)
                            reviews.append(review)
                            review_text = render_review(review)
                            change_text = render_change_set(cs)
                            if review_text or change_text:
                                log("review", "\n\n".join(x for x in (change_text, review_text) if x),
                                    meta={"iteration": iteration,
                                          "errors": len(review.get("errors") or []),
                                          "warnings": len(review.get("warnings") or []),
                                          "dimensions": cs.get("dimensions") or []})
                            if review.get("errors"):
                                message = (review_text + "\n\nFix the errors above and resend "
                                           "write_files; the build was not started.")
                            else:
                                files = candidate
                                declared = action.get("strategies") or []
                                if isinstance(declared, str):
                                    declared = [declared]
                                expected = action.get("expected_speedup")
                                result = evaluate(iteration, declared, expected, change=cs)
                                message = format_summary(result)
                                if change_text:
                                    message += "\n\n" + change_text
                                if review_text:
                                    message += "\n\n" + review_text
                                if rollback_note:
                                    message += "\n\n" + rollback_note
                                message += "\n\nReply with the next JSON action."
                                message = note_repeat(result, message)
        elif kind == "run_eval":
            result = evaluate(iteration, [], None, change=change_set(files, files, baseline_files))
            message = format_summary(result)
            if rollback_note:
                message += "\n\n" + rollback_note
            message += "\n\nReply with the next JSON action."
            message = note_repeat(result, message)
        elif kind == "knowledge":
            card_id = str(action.get("id") or "").strip()
            if not card_id:
                names = ", ".join(c["id"] for c in kb_cards) or "(no candidates)"
                message = "knowledge needs an \"id\" — use one of: %s" % names
            else:
                body = read_card(args.kb_root, card_id, max_chars=args.kb_body_chars)
                if not body:
                    message = ("No knowledge card at id %r. Candidates: %s"
                               % (card_id, ", ".join(c["id"] for c in kb_cards) or "(none)"))
                else:
                    log("knowledge", "read %s (%d chars)" % (card_id, len(body)),
                        meta={"iteration": iteration, "card": card_id})
                    message = ("KNOWLEDGE CARD %s\n%s\n\nUse it, then reply with the next JSON action."
                               % (card_id, body))
        elif kind == "source":
            src_path = str(action.get("path") or "").strip()
            if not ops_catalog:
                message = "source is unavailable: no CANN ops catalog is loaded."
            elif not src_path:
                message = "source needs a \"path\" from the CANN official source map."
            else:
                body = read_source(ops_catalog, src_path, max_chars=args.source_chars)
                if not body:
                    message = ("No readable source at %r (must be under the CANN impl root)."
                               % src_path)
                else:
                    log("source", "read %s (%d chars)" % (src_path, len(body)),
                        meta={"iteration": iteration, "path": src_path})
                    message = ("SOURCE FILE %s\n```cpp\n%s\n```\n\nUse it, then reply with the next JSON action."
                               % (src_path, body))
        elif kind == "finish":
            log("user", "finish requested: %s" % action.get("summary", ""))
            break
        else:
            message = ("Unknown action %r. Use read_file | patch_files | plan | write_files | run_eval | "
                       "kb_search | knowledge | source | source_grep | reflect | profile | finish."
                       % kind)

        messages.append({"role": "user", "content": message})
        log("user", message)

        if repeat_stop:
            log("system", "repeated error %d times -> stopping" % repeat_count)
            break

        if eval_count >= args.max_iters:
            log("system", "evaluation budget exhausted (%d) -> stopping" % eval_count)
            break

        if best["status"] == "pass" and best["speedup"] >= args.target_speedup:
            log("system", "target %.2fx reached (best %.3fx)" % (args.target_speedup, best["speedup"]))
            break

    final_dir = os.path.join(run_dir, "final_files")
    if best.get("files"):
        for rel, content in best["files"].items():
            target = os.path.join(final_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(content)

    meta = {
        "task": spec.id, "spec": spec.path, "package": spec.package,
        "device": device, "hardware": spec.hardware, "soc": spec.soc,
        "features": feats, "iterations": iteration, "best": best,
        "plan": plan_data,
        "reviews": reviews,
        "rollbacks": rollbacks,
        "reflections": reflections,
        "profiles": profiles,
        "kb_root": args.kb_root,
        "kb_candidates": [c["id"] for c in kb_cards],
        "ops_catalog": args.ops_catalog if ops_catalog else "",
        "source_map_ops": [op.get("name") for op in
                           (find_ops(ops_catalog, [(spec.metadata or {}).get("op"), spec.id], 1)
                            if ops_catalog else [])],
        "final_files": final_dir if best.get("files") else None,
        "rsi": {"enabled": bool(store),
                "priors_matched": [{"id": s["id"], "score": round(score, 3)}
                                   for s, score, _m in matched],
                "memory": ({
                    "facts": len(mem.facts),
                    "failures": len(mem.failures),
                    "open_failures": sum(1 for f in mem.failures if f.get("status") == "open"),
                    "skills": len(mem.skills),
                } if mem else None)},
        "usage": usage_total, "timestamp": _now(),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if args.promote_skill and mem and best.get("status") == "pass" and best.get("files"):
        candidates = [os.path.join(final_dir, rel) for rel in spec.edit_files
                      if os.path.isfile(os.path.join(final_dir, rel))]
        template_ref = candidates[0] if candidates else os.path.join(final_dir, spec.edit_files[0])
        promo = mem.promote_skill_from_run(
            run_dir, args.promote_skill, template_ref, feats,
            validation={"task": spec.id, "cases": best.get("case_summary") or {}},
        )
        meta["rsi"]["promoted_skill"] = promo
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print("promoted skill: %s (%s)" % (args.promote_skill, promo.get("status")))
        if promo.get("reason"):
            print("  reason: %s" % promo["reason"])

    transcript.close()
    attempts.close()

    if args.distill and store:
        report = distill_run(run_dir, store=store, strict_provenance=True)
        meta["rsi"]["distill"] = {
            "evidence": len(report.get("evidence") or []),
            "created": len(report.get("created") or []),
            "refined": len(report.get("refined") or []),
            "skipped": len(report.get("skipped") or []),
            "provenance_valid": (report.get("provenance") or {}).get("valid"),
        }
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print("strategy pool: evidence=%d created=%d skipped=%d provenance_valid=%s" % (
            meta["rsi"]["distill"]["evidence"], meta["rsi"]["distill"]["created"],
            meta["rsi"]["distill"]["skipped"], meta["rsi"]["distill"]["provenance_valid"]))

    print("run dir: %s" % run_dir)
    print("iterations: %d | best: %s at %s" % (
        iteration, best.get("status"),
        ("%.3fx" % best["speedup"]) if best.get("speedup") else "-"))
    return 0 if best.get("status") == "pass" else 1


def _as_list(command) -> list:
    if isinstance(command, (list, tuple)):
        return [str(x) for x in command]
    return [str(command or "")]


if __name__ == "__main__":
    raise SystemExit(main())