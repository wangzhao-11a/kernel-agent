"""Minimal kernel-optimization agent loop with RSI (recursive self-improvement).

Pipeline per run:
    retrieve strategy priors (condition match) -> prompt the model -> evaluate each attempt
    -> record attempts.jsonl + trajectory.json -> distill into the strategy pool
    -> (next run, same or other task) updated priors.

Any other agent (OpenHands, Claude Agent SDK, ...) can drive the same evaluation through
`python -m harness.runner`; this loop exists so the pipeline works end-to-end out of the box,
including fully offline (replay) runs for testing.

    # real model on the 910B server
    export DEEPSEEK_API_KEY=sk-...
    python -m agent.loop --task tasks/level2/mla_attention.py --device npu \\
        --provider openai --model deepseek-chat --base-url https://api.deepseek.com/v1

    # offline mechanics test (no API key, no NPU)
    python -m agent.loop --task tasks/level1/vector_add.py --provider replay \\
        --replay-file configs/replay_vector_add.example.json

    # what priors would this task get? (no agent run)
    python -m rsi.retrieve --task tasks/level2/grouped_matmul.py --hardware ascend-910b
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
from agent.prompts import build_feedback, build_task_prompt, system_prompt  # noqa: E402
from harness import device as dev  # noqa: E402
from harness import stats as stats_mod  # noqa: E402
from harness.runner import run_evaluation  # noqa: E402
from harness.tasks import load_task  # noqa: E402
from harness.worker import format_summary  # noqa: E402
from rsi import bottlenecks  # noqa: E402
from rsi.cards import DEFAULT_KB_ROOT as DEFAULT_CANN_KB_ROOT  # noqa: E402
from rsi.cards import read_card, render_candidates, retrieve_cards  # noqa: E402
from rsi.distill import attribute_from_source, distill_run, expand_declared, resolve_declared  # noqa: E402
from rsi.exemplars import render_exemplars, retrieve_exemplars  # noqa: E402
from rsi.features import hardware_from_device, task_features  # noqa: E402
from rsi.forms import form_for, form_hint  # noqa: E402
from rsi.memory import MemoryStore  # noqa: E402
from rsi import params as params_mod  # noqa: E402
from rsi.provenance import provenance  # noqa: E402
from rsi.retrieve import combos_for, failure_hints, priors_block, retrieve  # noqa: E402
from rsi.store import DEFAULT_ROOT as DEFAULT_KNOWLEDGE_ROOT  # noqa: E402
from rsi.store import StrategyStore  # noqa: E402
from rsi.trajectory import build_trajectory  # noqa: E402


def parse_action(text: str):
    """Extract one JSON action object from a model reply (tolerates fences and prose)."""
    if not text:
        return None, "empty response"
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    try:
        obj = json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return None, "no JSON object found in reply"
        try:
            obj = json.loads(s[start:end + 1])
        except Exception as exc:
            return None, "invalid JSON: %s" % exc
    if not isinstance(obj, dict) or "action" not in obj:
        return None, "reply is not a JSON action object (missing 'action')"
    return obj, None


def _load_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_meta(run_dir: str, meta: dict) -> None:
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def _memory_pseudo(result: dict) -> dict:
    """Adapt a harness result to the shape expected by MemoryStore failure extraction."""
    status = result.get("status")
    trials = (result.get("correctness") or {}).get("trials") or []
    failed = [t for t in trials if not t.get("ok")]
    return {
        "status": status,
        "error": result.get("error"),
        "case_summary": {"failed": len(failed), "total": len(trials)},
        "cases": [{"passed": bool(t.get("ok"))} for t in trials],
        "test": {"tail": result.get("error") or ""},
        "build": {"tail": result.get("error") or ""},
    }


def _error_signature(result: dict) -> str:
    """Stable signature for repeated-failure detection (strip PIDs/timestamps)."""
    raw = str(result.get("error") or "").strip()
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if ln.strip()][:2]
    text = " | ".join(lines)
    text = re.sub(r"\[PID:\d+\]|\[\d{4}-\d\d-\d\d[^\]]*\]", "", text)
    return " ".join(text.split())[:300]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.loop", description="Kernel optimization agent loop with RSI.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default=None, help="JSON file with provider/model/base_url/... defaults")
    ap.add_argument("--provider", default=None, choices=["openai", "anthropic", "replay"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--target-speedup", type=float, default=None)
    ap.add_argument("--keep-going", action="store_true", help="keep iterating after the target is reached")
    ap.add_argument("--tune-max-variants", type=int, default=6,
                    help="maximum rendered variants for one tune action")
    ap.add_argument("--replay-file", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hardware", default="", help="hardware label for strategy matching (default: task meta / device)")
    ap.add_argument("--require", default="", help="run-level requirements forwarded to the harness, e.g. 'triton'")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--profiler", default="torch", choices=["off", "torch", "msprof", "ncu"],
                    help="profiling backend fed back into the loop (default torch = cross-platform)")
    ap.add_argument("--profile-every", type=int, default=2,
                    help="auto-profile after every Nth passing iteration (0 = only via the profile action)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "agent"))
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--knowledge-root", default=DEFAULT_KNOWLEDGE_ROOT)
    ap.add_argument("--kb-root", default=DEFAULT_CANN_KB_ROOT,
                    help="cannbot-knowledge checkout used as a read-on-demand knowledge source")
    ap.add_argument("--kb-cards", type=int, default=4,
                    help="max knowledge-card candidates in the prompt (0 disables the block)")
    ap.add_argument("--no-kb-auto", action="store_true",
                    help="do not auto-load the best matching card body for kernel-form tasks")
    ap.add_argument("--no-rsi", action="store_true", help="disable priors / attribution / distillation")
    ap.add_argument("--no-memory", action="store_true",
                    help="disable operator memory (facts/failures/skills) for this run")
    ap.add_argument("--no-distill", action="store_true", help="do not update the strategy pool at the end")
    ap.add_argument("--verify-distill", action="store_true", help="replay attempts before distilling")
    ap.add_argument("--strict-provenance", action="store_true",
                    help="refuse to update performance evidence when device != claimed hardware "
                         "(e.g. CPU mechanics run labelled ascend-910b)")
    ap.add_argument("--policy-exempt", action="store_true",
                    help="legacy/reference/replay run: skip strict task policy checks")
    ap.add_argument("--max-repeat-errors", type=int, default=3,
                    help="stop after N consecutive attempts with the same error signature (0 disables)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)

    def pick(name, default=None):
        value = getattr(args, name, None)
        return value if value is not None else cfg.get(name, default)

    provider = pick("provider", "openai")
    model = pick("model", "")
    base_url = pick("base_url", "")
    temperature = float(pick("temperature", 0.2))
    max_iters = int(pick("max_iters", 12))
    target = float(pick("target_speedup", 1.0))
    api_key_env = pick("api_key_env", "") or ""
    use_rsi = not args.no_rsi

    task = load_task(args.task)
    # run-level requirements become part of the task's constraints (prompt + harness)
    cli_requires = [part.strip().lower() for part in str(args.require or "").split(",") if part.strip()]
    if cli_requires:
        task.meta["require"] = sorted(set([str(r).lower() for r in (task.meta.get("require") or [])] + cli_requires))
    resolved_device = dev.resolve_device(args.device)
    feats = task_features(task)
    hardware = args.hardware or feats.get("hardware") or hardware_from_device(resolved_device)

    store = StrategyStore(args.knowledge_root) if use_rsi else None
    matched = retrieve(store, feats, hardware, top_k=5) if use_rsi else []
    combos = combos_for(store, [s["id"] for s, _score, _match in matched]) if use_rsi else []
    priors = priors_block(feats, hardware, matched, combos) if use_rsi else ""
    exemplars = retrieve_exemplars(args.knowledge_root, feats, hardware, top_k=2) if use_rsi else []
    exemplar_block = render_exemplars(exemplars)
    form = form_for(feats)
    form_block = form_hint(feats)
    mem = MemoryStore(args.knowledge_root) if (use_rsi and not args.no_memory) else None
    mem_features = {
        "op": task.name,
        "op_family": feats.get("op_class") or "",
        "soc": hardware,
        "dtype": feats.get("dtypes") or [],
        "form": (["triton"] if form == "triton_kernel"
                 else (["torch_fused"] if form == "torch_fused_first" else [])),
        "task": task.name,
    }
    mem_prov = provenance({"device": resolved_device, "hardware": hardware})
    mem_write_features = dict(mem_features)
    if not mem_prov.get("valid"):
        # mechanics-only run: file failures under cpu, not the claimed NPU scope
        mem_write_features["soc"] = "cpu"
    memory_block = mem.context(mem_features, kinds=("fact", "failure", "skill")) if mem else ""
    kb_cards = retrieve_cards(args.kb_root, feats, top_k=args.kb_cards, required_form=form) if use_rsi else []
    kb_block = render_candidates(kb_cards)
    kb_auto = None
    if kb_cards and form == "triton_kernel" and not args.no_kb_auto:
        top_card = kb_cards[0]
        body = read_card(args.kb_root, top_card["id"], max_chars=2500)
        if body:
            kb_auto = {"id": top_card["id"], "title": top_card.get("title")}
            kb_block = (kb_block + "\n\n### Auto-loaded top card (best match for a kernel-form task)\n"
                        "```markdown\n" + body + "\n```")
    kb_reads = []
    attempted_ids = set()

    api_key = llm_mod.resolve_api_key(provider, explicit=args.api_key or "", env_name=api_key_env)
    if provider in ("openai", "anthropic") and not api_key:
        print("ERROR: no API key found. Set one of the *_API_KEY env vars or pass --api-key.")
        return 2

    replayer = None
    replay_exempt = False
    if provider == "replay":
        if not args.replay_file:
            print("ERROR: --provider replay needs --replay-file")
            return 2
        with open(args.replay_file, "r", encoding="utf-8") as fh:
            replay_payload = json.load(fh)
        replay_exempt = bool(replay_payload.get("policy_exempt", False))
        replayer = llm_mod.ReplayModel(replay_payload.get("responses", []))
    policy_exempt = bool(args.policy_exempt or replay_exempt)

    run_name = "%s_%s" % (datetime.now().strftime("%Y%m%d-%H%M%S"), task.name)
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    transcript_jsonl = open(os.path.join(run_dir, "transcript.jsonl"), "a", encoding="utf-8")
    transcript_md = open(os.path.join(run_dir, "transcript.md"), "a", encoding="utf-8")
    attempts_file = open(os.path.join(run_dir, "attempts.jsonl"), "a", encoding="utf-8")

    def log(role: str, content: str, meta: dict = None) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "role": role, "content": content}
        if meta:
            record.update(meta)
        transcript_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        transcript_jsonl.flush()
        transcript_md.write("\n### %s\n\n%s\n" % (role, content))
        transcript_md.flush()

    sys_prompt = system_prompt(
        device=resolved_device,
        atol=task.tolerance.get("atol", 1e-5),
        rtol=task.tolerance.get("rtol", 1e-5),
        warmup=task.warmup if args.warmup is None else args.warmup,
        rep=task.rep if args.rep is None else args.rep,
        baseline=task.baseline,
    )
    first_prompt = build_task_prompt(task, resolved_device, max_iters, priors=priors,
                                     exemplars=exemplar_block, knowledge=kb_block, form=form_block,
                                     memory=memory_block)
    log("system", sys_prompt)
    log("user", first_prompt)
    messages = [{"role": "user", "content": first_prompt}]

    best = {"status": None, "speedup": 0.0, "path": None}
    current_path = None
    strikes = 0
    passed_count = 0
    nudge_count = 0
    finisher = None
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    iteration = 0

    def do_profile(path: str, it: int):
        """Profile the current solution in a subprocess; returns (text, meta)."""
        from harness.profile_runner import profile as run_profile

        res = run_profile(args.task, path, device=args.device, backend=args.profiler,
                          out_dir=os.path.join(run_dir, "profiling"), timeout=args.timeout, iters=3)
        if res.get("status") == "ok":
            text = res["text"]
        else:
            text = "profiling unavailable (%s): %s" % (res.get("status"), res.get("error"))
        log("profiling", text, meta={"iteration": it, "backend": args.profiler, "status": res.get("status")})
        if not args.quiet:
            print("profiling (%s): %s" % (args.profiler, res.get("status")))
        summary = res.get("summary") or {}
        advice = bottlenecks.lookup({"top_ops": summary.get("top_ops") or []})
        if advice:
            text = text + "\n\n" + bottlenecks.render(advice)
        meta = {"backend": args.profiler, "status": res.get("status"),
                "top_ops": summary.get("top_ops") or None, "advice": advice or None,
                "out_dir": res.get("out_dir")}
        return text, meta

    repeat_sig = ""
    repeat_count = 0
    repeat_stop = False
    max_repeat_errors = int(args.max_repeat_errors or 0)

    def evaluate(path: str, it: int, declared, expected) -> str:
        nonlocal passed_count, repeat_sig, repeat_count, repeat_stop
        result = run_evaluation(
            task=args.task, solution=path, device=args.device, timeout=args.timeout,
            warmup=args.warmup, rep=args.rep, trials=args.trials,
            require=args.require or None,
            policy_exempt=policy_exempt,
            run_id=run_name, iteration=it, out_dir=run_dir,
            results_file=args.results_file, quiet=True,
        )
        summary = format_summary(result)
        log("harness", summary, meta={"iteration": it, "status": result.get("status")})
        status = result.get("status") or "unknown"
        speedup = float((result.get("timing") or {}).get("speedup") or 0.0)
        repeat_note = ""
        if status != "pass" and max_repeat_errors > 0:
            sig = _error_signature(result)
            if sig:
                if sig == repeat_sig:
                    repeat_count += 1
                else:
                    repeat_sig, repeat_count = sig, 1
                if repeat_count >= 2:
                    repeat_note = (
                        "REPEATED FAILURE: the same error signature has occurred %d consecutive "
                        "attempts:\n%s\nDo NOT submit another version with the same shape/axis "
                        "logic. Re-derive the tensor shapes on paper, switch strategy, or finish."
                        % (repeat_count, sig)
                    )
                if repeat_count >= max_repeat_errors:
                    repeat_stop = True
                    repeat_note += "\nStopping: the same failure repeated %d times." % repeat_count
        else:
            repeat_sig, repeat_count = "", 0

        resolved = []
        if use_rsi:
            resolved = [r for r in (resolve_declared(d, store) for d in expand_declared(declared, store)) if r]
            if not resolved:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        source = fh.read()
                except OSError:
                    source = ""
                resolved = attribute_from_source(source, store, feats)
            attempted_ids.update(resolved)

        profile_text, profile_meta = "", None
        if status == "pass":
            passed_count += 1
        # guarantee at least one profile per run: first passing attempt always, then every N-th
        if (status == "pass" and args.profiler != "off" and args.profile_every > 0
                and (passed_count == 1 or passed_count % args.profile_every == 0)):
            profile_text, profile_meta = do_profile(path, it)

        measured = speedup if status == "pass" else 0.0
        # Counterfactual control: before a candidate can replace the best and
        # feed strategy confidence, re-measure the previous best (revert the
        # candidate's change set) and require a CI-significant gain.
        ablation = None
        evidence_eligible = True
        promote = False
        better = bool(status == "pass" and (
            best.get("status") != "pass" or speedup >= float(best.get("speedup") or 0.0)))
        if (status == "pass" and best.get("status") == "pass" and best.get("path")
                and best.get("path") != path and (declared or resolved)
                and speedup >= float(best.get("speedup") or 0.0)):
            control_res = run_evaluation(
                task=args.task, solution=best["path"], device=args.device,
                timeout=args.timeout, warmup=args.warmup, rep=args.rep, trials=args.trials,
                require=args.require or None, policy_exempt=policy_exempt,
                run_id=run_name, iteration=it,
                out_dir=os.path.join(run_dir, "ablation-%02d" % it),
                results_file=args.results_file, quiet=True,
            )
            ctiming = control_res.get("timing") or {}
            cspeed = float(ctiming.get("speedup") or 0.0)
            ci = stats_mod.bootstrap_ratio_ci(
                (ctiming.get("solution") or {}).get("samples_ms"),
                (result.get("timing") or {}).get("solution", {}).get("samples_ms"))
            significant = stats_mod.ci_excludes_one(ci)
            ablation = {
                "control_speedup": round(cspeed, 4),
                "control_status": control_res.get("status"),
                "ci95": ci,
                "significant": bool(significant),
            }
            evidence_eligible = bool(significant)
            promote = bool(significant)
            if not significant:
                better = False
        attempt = {
            "schema_version": 1,
            "iter": it,
            "task": task.name,
            "task_file": os.path.abspath(args.task),
            "solution": os.path.abspath(path),
            "status": status,
            "speedup": speedup,
            "error": result.get("error"),
            "expected_speedup": expected,
            "strategies_declared": list(declared),
            "strategies_used": resolved,
            "uses_triton": bool((result.get("solution_features") or {}).get("uses_triton")),
            "kb_reads": list(dict.fromkeys(kb_reads)),
            "calibration_error": (round(abs(expected - measured), 4) if expected is not None else None),
            "ablation": ablation,
            "evidence_eligible": evidence_eligible,
            "promote": promote,
            "profile": profile_meta,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "distilled": False,
        }
        attempts_file.write(json.dumps(attempt, ensure_ascii=False) + "\n")
        attempts_file.flush()
        if mem:
            if status == "pass":
                mem.resolve_failures(mem_write_features, run_name, "eval pass")
            else:
                mem.record_failure_from_result(_memory_pseudo(result), mem_write_features,
                                               run_name, task.name)

        uses_kernel = bool((result.get("solution_features") or {}).get("uses_triton"))
        if better:
            best.update(status=status, speedup=speedup, path=path, uses_triton=uses_kernel)

        hints = failure_hints(matched, attempted_ids) if (use_rsi and status != "pass") else ""
        if repeat_note:
            hints = (hints + "\n\n" + repeat_note).strip()
        return build_feedback(iteration=it, status=status, summary=summary, speedup=speedup,
                              best=best, hints=hints, profiling=profile_text)

    for iteration in range(1, max_iters + 1):
        try:
            reply = llm_mod.chat(
                messages, system=sys_prompt, provider=provider, model=model, base_url=base_url,
                api_key=api_key, temperature=temperature, max_tokens=args.max_tokens, replay=replayer,
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

        action, err = parse_action(text)
        if err:
            strikes += 1
            if str(reply.get("finish_reason") or "") == "length":
                message = ("Your reply was CUT OFF by the output token limit (finish_reason=length), so the "
                           "JSON action never closed. Reply again with ONLY the JSON object: keep the "
                           "solution compact, drop long commentary, and write it as one line per statement "
                           "with no extra comments. Nothing outside the JSON object.")
            else:
                message = ("Your reply was not a single JSON action object (%s). "
                           "Reply with exactly one JSON object and nothing else." % err)
            messages.append({"role": "user", "content": message})
            log("user", message)
            if strikes >= 3:
                log("system", "aborting: too many unparsable replies")
                break
            continue
        strikes = 0
        kind = str(action.get("action", "")).lower()

        if kind == "write_solution":
            content = action.get("content")
            declared = action.get("strategies") or []
            if isinstance(declared, str):
                declared = [declared]
            declared = [str(d).strip() for d in declared if str(d).strip()]
            expected = _as_float(action.get("expected_speedup"))
            if not isinstance(content, str) or not content.strip():
                message = "write_solution needs a non-empty 'content' string containing the full source."
            else:
                current_path = os.path.join(run_dir, "solution_iter%02d.py" % iteration)
                with open(current_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                message = evaluate(current_path, iteration, declared, expected)
        elif kind == "tune":
            content = action.get("content")
            raw_space = action.get("params")
            if not isinstance(content, str) or not content.strip():
                message = ("tune needs a non-empty 'content' template with {{PARAM}} placeholders "
                           "and a 'params' object.")
            else:
                try:
                    space = params_mod.parse_space(raw_space)
                    check = params_mod.validate(content, space)
                    if not check["ok"]:
                        raise params_mod.ParamError(
                            "template/params mismatch: placeholders=%s declared=%s missing=%s unused=%s"
                            % (check["placeholders"], check["declared"],
                               check["missing_declaration"], check["unused"]))
                    variants = params_mod.grid(space, args.tune_max_variants, seed=iteration)
                except params_mod.ParamError as exc:
                    message = "tune rejected: %s" % exc
                else:
                    rows = []
                    best_tune = None
                    for idx, values in enumerate(variants):
                        try:
                            rendered = params_mod.render(content, values)
                        except params_mod.ParamError as exc:
                            message = "tune render failed: %s" % exc
                            break
                        path = os.path.join(run_dir, "tune_iter%02d_var%02d.py" % (iteration, idx))
                        with open(path, "w", encoding="utf-8") as fh:
                            fh.write(rendered)
                        res = run_evaluation(
                            task=args.task, solution=path, device=args.device,
                            timeout=args.timeout, warmup=args.warmup, rep=args.rep,
                            trials=args.trials, require=args.require or None,
                            policy_exempt=policy_exempt, run_id=run_name, iteration=iteration,
                            out_dir=os.path.join(run_dir, "tune-%02d" % iteration),
                            results_file=args.results_file, quiet=True,
                        )
                        status = res.get("status") or "unknown"
                        speed = float((res.get("timing") or {}).get("speedup") or 0.0)
                        log("harness", format_summary(res),
                            meta={"iteration": iteration, "tune_variant": idx, "params": values})
                        rows.append({"idx": idx, "params": values, "status": status,
                                     "speedup": speed, "path": path,
                                     "uses_triton": bool((res.get("solution_features") or {}).get("uses_triton"))})
                        if status == "pass" and (best_tune is None or speed > best_tune["speedup"]):
                            best_tune = rows[-1]
                    if rows:
                        chosen = best_tune or max(rows, key=lambda r: r["speedup"])
                        current_path = chosen["path"]
                        if best_tune and (best.get("status") != "pass"
                                          or best_tune["speedup"] > float(best.get("speedup") or 0.0)):
                            best.update(status="pass", speedup=best_tune["speedup"],
                                        path=best_tune["path"], uses_triton=best_tune["uses_triton"])
                        table = "\n".join(
                            "  %s -> %s %.3fx" % (r["params"], r["status"], r["speedup"])
                            for r in rows)
                        message = ("Tuned %d/%d variants (structure kept, numeric axes searched):\n%s\n"
                                   "Best variant: %s (%s, %.3fx). The current solution is the best variant."
                                   % (len(rows), len(variants), table, chosen["params"],
                                      chosen["status"], chosen["speedup"]))
                    else:
                        message = "tune produced no variants."
        elif kind == "run_eval":
            if current_path is None:
                message = "No solution submitted yet. Use write_solution first."
            else:
                message = evaluate(current_path, iteration, [], None)
        elif kind == "knowledge":
            card_id = str(action.get("id") or "").strip()
            if not card_id:
                message = ("knowledge needs an \"id\" — use a card path from the CANN knowledge cards "
                           "candidates list.")
            else:
                body = read_card(args.kb_root, card_id)
                if body:
                    if card_id not in kb_reads:
                        kb_reads.append(card_id)
                    log("knowledge", "read %s (%d chars)" % (card_id, len(body)),
                        meta={"iteration": iteration})
                    message = ("Knowledge card %s:\n```markdown\n%s\n```\n"
                               "Apply what fits this task, then reply with exactly one JSON action."
                               % (card_id, body))
                else:
                    message = ("No knowledge card at id %r (use a path from the candidates list)."
                               % card_id)
        elif kind == "profile":
            if args.profiler == "off":
                message = "Profiling is disabled (--profiler off). Use write_solution / run_eval / finish."
            elif current_path is None:
                message = "No solution submitted yet. Use write_solution first."
            else:
                text, _meta = do_profile(current_path, iteration)
                message = "Profiling result:\n```\n%s\n```\nReply with exactly one JSON action." % text
        elif kind == "finish":
            finisher = str(action.get("summary", ""))
            log("user", "finish requested: %s" % finisher)
            if best.get("status") != "pass" and iteration < max_iters and nudge_count < 2:
                nudge_count += 1
                message = ("No passing solution yet and you still have %d iteration(s) of budget — do not "
                           "stop. Switch implementation form (torch fused ops vs triton kernel), load a "
                           "CANN knowledge card if you have not, and use the profiling feedback. "
                           "Reply with exactly one JSON action." % (max_iters - iteration))
            else:
                break
        else:
            message = "Unknown action %r. Use one of: write_solution | run_eval | finish." % kind

        messages.append({"role": "user", "content": message})
        log("user", message)

        if repeat_stop:
            log("system", "stopping: repeated failure %d times" % repeat_count)
            break

        if best["status"] == "pass" and best["speedup"] >= target and not args.keep_going:
            log("system", "target speedup %.2fx reached (best %.3fx) -> stopping" % (target, best["speedup"]))
            break

    attempts_file.close()

    final_path = None
    if best.get("path"):
        final_path = os.path.join(run_dir, "final_solution.py")
        shutil.copyfile(best["path"], final_path)

    # meta.json is written first so trajectory building / distillation can read it
    meta = {
        "task": task.name,
        "task_file": os.path.abspath(args.task),
        "iterations": iteration,
        "provider": provider,
        "model": model,
        "device": resolved_device,
        "hardware": hardware,
        "features": feats,
        "profiler": args.profiler,
        "profile_every": args.profile_every,
        "rsi": {
            "enabled": use_rsi,
            "knowledge_root": args.knowledge_root if use_rsi else "",
            "priors_matched": [{"id": s["id"], "key": s.get("key"), "score": round(score, 3)}
                               for s, score, _match in matched],
            "attempted": sorted(attempted_ids),
            "exemplars": [row["id"] for row in exemplars],
            "kb_root": args.kb_root if use_rsi else "",
            "kb_candidates": [card["id"] for card in kb_cards],
            "kb_reads": list(dict.fromkeys(kb_reads)),
            "kb_auto": kb_auto,
            "form": form,
            "strict_provenance": bool(args.strict_provenance),
            "memory": ({
                "facts": len(mem.facts),
                "failures": len(mem.failures),
                "open_failures": sum(1 for f in mem.failures if f.get("status") == "open"),
                "skills": len(mem.skills),
            } if mem else None),
            "nudges": nudge_count,
            "trajectory_attempts": None,
            "distill": None,
        },
        "best": best,
        "final_solution": final_path,
        "target_speedup": target,
        "finish_summary": finisher,
        "usage": usage_total,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _write_meta(run_dir, meta)

    if use_rsi:
        traj = build_trajectory(run_dir, hardware=hardware)
        distill_summary = None
        if not args.no_distill:
            report = distill_run(run_dir, store=store, verify=args.verify_distill,
                                 device=args.device, timeout=args.timeout, results_file=args.results_file,
                                 strict_provenance=args.strict_provenance)
            distill_summary = {
                "evidence": len(report["evidence"]),
                "created": len(report["created"]),
                "combos": len(report["combos"]),
                "refined": len(report["refined"]),
                "skipped": len(report["skipped"]),
                "replay_ok": report["replay_ok"],
                "provenance_valid": (report.get("provenance") or {}).get("valid"),
            }
            log("system", "strategy pool updated: %s" % json.dumps(distill_summary))
            print("strategy pool updated: evidence=%d created=%d combos=%d refined=%d"
                  % (distill_summary["evidence"], distill_summary["created"],
                     distill_summary["combos"], distill_summary["refined"]))
        meta["rsi"]["trajectory_attempts"] = len(traj.get("attempts", []))
        meta["rsi"]["distill"] = distill_summary
        _write_meta(run_dir, meta)

    transcript_jsonl.close()
    transcript_md.close()

    print("run dir: %s" % run_dir)
    print("iterations used: %d | best: %s at %.3fx" % (iteration, best.get("status"), best.get("speedup", 0.0)))
    if final_path:
        print("final solution: %s" % final_path)
    if finisher:
        print("agent summary: %s" % finisher)
    return 0 if best.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())