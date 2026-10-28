"""Worker: evaluate ONE (task, solution) pair and write a JSON result.

Always run through harness.runner (subprocess + timeout) so that crashes, hangs and
CUDA/NPU errors in agent-written code stay isolated from the harness.

    python -m harness.worker --task tasks/level1/vector_add.py --solution sol.py --out out.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import time
import traceback
from datetime import datetime, timezone

from . import benchmark as bench_mod
from . import correctness as corr_mod
from . import device as dev
from . import policy as pol
from . import roofline as roofline_mod
from . import stats as stats_mod
from .tasks import TaskLoadError, load_task


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail(text, n: int = 1500) -> str:
    text = text or ""
    return text[-n:] if len(text) > n else text


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def import_solution(path: str):
    spec = importlib.util.spec_from_file_location("kernel_agent_solution", os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load solution module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suspicion_report(task, solution_path: str, solution_module) -> list:
    """Cheap anti-cheat signals: the solution must not reach into task/harness code."""
    reasons = []
    try:
        with open(solution_path, "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError:
        return reasons
    stem = os.path.splitext(os.path.basename(task.path))[0]
    for line in src.splitlines():
        s = line.strip()
        if re.match(r"^(from|import)\s+", s) and ("harness" in s or stem in s):
            reasons.append("imports task/harness module: %s" % s[:120])
    task_file = os.path.abspath(task.path)
    for name, val in vars(solution_module).items():
        vf = getattr(val, "__file__", None)
        if vf and os.path.abspath(str(vf)) == task_file:
            reasons.append("solution holds a reference to the task module (%s)" % name)
    return reasons


def policy_report(task, solution_path: str, solution_module, policy: dict) -> dict:
    """Auxiliary AST/static checks plus kernel evidence for a strict task."""
    try:
        with open(solution_path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError:
        source = ""
    findings = pol.audit_source(source, os.path.splitext(os.path.basename(task.path))[0])
    evidence = pol.kernel_evidence(source, solution_module)
    errors = []
    for f in findings:
        if f["code"] in ("reference_import", "dynamic_import", "dynamic_exec"):
            errors.append(f)
        elif policy.get("forbid_native_compute") and f["code"] == "native_compute":
            errors.append(f)
        elif policy.get("forbid_numpy_compute") and f["code"] == "numpy_compute":
            errors.append(f)
        elif policy.get("forbid_reference") and f["code"] in ("reference_call", "reference_ref"):
            errors.append(f)
    if policy.get("require_kernel") and not evidence.get("any"):
        errors.append({"severity": "error", "code": "kernel_evidence_missing",
                       "message": "task requires a real kernel; no triton @jit/cpp_extension/CANN marker found"})
    return {"findings": findings, "errors": errors, "kernel_evidence": evidence}


def _clone_inputs(inputs):
    out = []
    for x in inputs:
        if hasattr(x, "clone"):
            out.append(x.clone())
        elif hasattr(x, "copy"):
            out.append(x.copy())
        else:
            out.append(x)
    return out


def _run_case(task, run, device, inputs, policy, policy_exempt, allow_nonfinite):
    """Reference + candidate for one input set, with the runtime policy guard."""
    ref = task.reference(*inputs)
    try:
        with pol.policy_guard(policy, policy_exempt):
            out = run(*_clone_inputs(inputs))
    except pol.PolicyViolation as exc:
        return {"ok": False, "policy_violation": {"code": exc.code, "detail": exc.detail},
                "reason": "policy violation: %s%s" % (exc.code, (": " + exc.detail) if exc.detail else "")}
    except Exception as exc:
        return {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    res = corr_mod.compare_outputs(ref, out, float(task.tolerance.get("atol", 1e-5)),
                                   float(task.tolerance.get("rtol", 1e-5)),
                                   allow_nonfinite=allow_nonfinite)
    return res


def heldout_report(task, run, device, policy, policy_exempt, allow_nonfinite):
    """Evaluate held-out shapes (or a fresh trial seed for legacy single-shape tasks)."""
    cases = task.cases("heldout")
    checked = []
    if cases:
        for case in cases:
            inputs = task.case_inputs(case, device, 0)
            res = _run_case(task, run, device, inputs, policy, policy_exempt, allow_nonfinite)
            checked.append({"name": case["name"], "shape": case.get("shape"),
                            "dtype": case.get("dtype"), **res})
            if not res.get("ok"):
                return {"ok": False, "cases": checked,
                        "reason": "held-out case %s: %s" % (case["name"], res.get("reason"))}
        return {"ok": True, "cases": checked, "reason": None}
    if not policy_exempt and (policy.get("forbid_native_compute") or policy.get("require_kernel")):
        inputs = task.make_inputs(device, int(task.num_trials) + 7)
        res = _run_case(task, run, device, inputs, policy, policy_exempt, allow_nonfinite)
        checked.append({"name": "fresh-seed", "shape": None, "dtype": None, **res})
        if not res.get("ok"):
            return {"ok": False, "cases": checked,
                    "reason": "fresh-seed check: %s" % res.get("reason")}
        return {"ok": True, "cases": checked, "reason": None}
    return None


def _wrap_inference(fn):
    t = dev.torch_module()
    if t is None:
        return fn

    def wrapper(*args, **kwargs):
        with t.inference_mode():
            return fn(*args, **kwargs)

    return wrapper


def _baseline_fn(task, kind: str, device: str):
    if kind in ("eager", ""):
        return task.reference
    if kind == "op":
        fn = getattr(task.module, "baseline", None)
        if not callable(fn):
            raise RuntimeError("task declares baseline='op' but the module defines no baseline()")
        return fn
    if kind == "compile":
        t = dev.torch_module()
        if t is None or not hasattr(t, "compile"):
            raise RuntimeError("torch.compile is not available")
        return t.compile(task.reference)
    raise RuntimeError("unknown baseline kind: %r" % kind)


def _baseline_sane(task, baseline_fn, device: str):
    """A non-eager baseline must produce (loosely) correct values, else it is a fake bar."""
    try:
        inputs = task.make_inputs(device, 0)
        ref = _wrap_inference(task.reference)(*inputs)
        out = _wrap_inference(baseline_fn)(*inputs)
    except Exception as exc:
        return False, "baseline raised %s: %s" % (type(exc).__name__, str(exc)[:160])
    try:
        ref_leaves = corr_mod.leaves(ref)
        out_leaves = corr_mod.leaves(out)
        if len(ref_leaves) != len(out_leaves):
            return False, "baseline output arity differs from the reference"
        for r, o in zip(ref_leaves, out_leaves):
            if tuple(getattr(o, "shape", ())) != tuple(getattr(r, "shape", ())):
                return False, "baseline output shape differs from the reference"
            if hasattr(o, "to") and hasattr(r, "dtype"):
                o = o.to(r.dtype)
            diff = abs(o - r)
            limit = 5e-2 + 5e-2 * abs(r)
            if int((diff > limit).sum()):
                return False, "baseline values look wrong (max_abs_err=%.3e)" % float(diff.max())
    except Exception as exc:
        return False, "baseline sanity check failed: %s" % exc
    return True, None


def format_summary(result: dict) -> str:
    lines = ["status: %s" % result.get("status")]
    if result.get("task"):
        lines.append("task: %s (level %s) | device: %s" % (result.get("task"), result.get("level"), result.get("device")))
    info = result.get("device_info") or {}
    if info.get("name"):
        lines.append("hardware: %s" % info["name"])
    c = result.get("correctness")
    if c is not None:
        lines.append(
            "correctness: %s (trials=%d, max_abs_err=%.3e)"
            % ("ok" if c.get("ok") else "FAILED", len(c.get("trials", [])), c.get("max_abs_err", 0.0))
        )
    t = result.get("timing")
    if t:
        lines.append(
            "timing: solution=%.4f ms | baseline(%s)=%.4f ms | speedup=%.3fx | stability=%.1f%%"
            % (
                t["solution"]["median_ms"],
                t.get("baseline_kind", "eager"),
                t["baseline"]["median_ms"],
                t.get("speedup", 0.0),
                100.0 * t.get("stability_delta", 0.0),
            )
        )
        if t.get("speedup_ci95"):
            lines.append("speedup CI95: [%.3f, %.3f]" % tuple(t["speedup_ci95"]))
        cases = t.get("shape_cases") or []
        if len(cases) > 1:
            worst = t.get("worst_case") or {}
            gate = t.get("shape_gate") or {}
            lines.append("shapes: %d cases | weighted=%.3fx | worst=%s %.3fx%s" % (
                len(cases), t.get("weighted_speedup", 0.0), worst.get("name"),
                worst.get("speedup", 0.0),
                (" | gate %s (>= %.3fx)" % ("ok" if gate.get("passed") else "FAILED",
                                            gate.get("min_required", 0.0))) if gate else ""))
        roof = t.get("roofline") or {}
        if roof:
            parts = ["bottleneck=%s" % roof.get("bottleneck"),
                     "bw=%.1f GB/s" % roof.get("achieved_bw_gbps", 0.0)]
            if roof.get("bw_utilization") is not None:
                parts.append("bw_util=%.1f%%" % (100 * roof["bw_utilization"]))
            if roof.get("flops_utilization") is not None:
                parts.append("flops_util=%.1f%%" % (100 * roof["flops_utilization"]))
            lines.append("roofline: " + ", ".join(parts))
    if result.get("policy_strict"):
        lines.append("policy: strict (native compute forbidden%s)" % (
            ", kernel required" if (result.get("policy") or {}).get("require_kernel") else ""))
    held = result.get("heldout") or {}
    if held.get("cases"):
        lines.append("held-out: %s (%d case(s))" % (
            "ok" if held.get("ok") else "FAILED", len(held["cases"])))
    audit = result.get("policy_audit") or {}
    if audit.get("cache_suspected") or audit.get("side_stream_suspected"):
        lines.append("policy audit: cache=%s side_stream=%s" % (
            audit.get("cache_suspected"), audit.get("side_stream_suspected")))
    if result.get("suspicious"):
        lines.append("suspicious: %s" % "; ".join(result.get("suspicious_reasons") or []))
    if result.get("status") != "pass" and result.get("error"):
        lines.append("error: %s" % result["error"])
    return "\n".join(lines)


def _finish(result: dict, out_path: str, t_start: float) -> dict:
    result["wall_time_s"] = round(time.time() - t_start, 2)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print("WARNING: could not write result to %s: %s" % (out_path, exc))
    print(format_summary(result))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.worker", description="Evaluate one solution against one task.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--baseline", default=None, help="override the task baseline: eager | compile")
    ap.add_argument("--flush-l2", dest="flush_l2", action="store_true", default=None)
    ap.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    ap.add_argument("--fail-on-suspicious", action="store_true")
    ap.add_argument("--policy-exempt", action="store_true",
                    help="legacy/reference/replay run: skip strict task policy checks")
    ap.add_argument("--ignore-require", action="store_true",
                    help="skip harness-enforced task requirements (require: [...])")
    ap.add_argument("--require", default="",
                    help="comma-separated run-level requirements, merged with the task's own")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--iter", type=int, default=0)
    args = ap.parse_args(argv)

    t_start = time.time()
    device = dev.resolve_device(args.device)
    result = {
        "schema_version": 1,
        "task": None,
        "task_file": os.path.abspath(args.task),
        "level": None,
        "solution": os.path.abspath(args.solution),
        "solution_sha256": None,
        "device": device,
        "device_info": dev.device_info(device),
        "status": "unknown",
        "error": None,
        "suspicious": False,
        "suspicious_reasons": [],
        "policy": None,
        "policy_exempt": bool(args.policy_exempt),
        "policy_findings": [],
        "heldout": None,
        "correctness": None,
        "timing": None,
        "run_id": args.run_id,
        "iter": args.iter,
        "timestamp": _now(),
    }

    try:
        result["solution_sha256"] = _sha256_file(args.solution)
    except OSError as exc:
        result["status"] = "import_error"
        result["error"] = "solution file not readable: %s" % exc
        return _finish(result, args.out, t_start)

    try:
        task = load_task(args.task)
    except Exception as exc:
        result["status"] = "task_error"
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        return _finish(result, args.out, t_start)

    result["task"] = task.name
    result["level"] = task.level
    result["policy"] = task.policy
    result["task_schema_version"] = task.schema_version
    strict = bool(not args.policy_exempt and (
        task.policy.get("forbid_native_compute") or task.policy.get("require_kernel")
        or task.policy.get("forbid_numpy_compute")))
    result["policy_strict"] = strict
    result["task_tags"] = [str(t) for t in (task.meta.get("tags") or [])]
    result["task_op_class"] = str(task.meta.get("op_class") or "")
    result["task_hardware"] = str(task.meta.get("hardware") or "")
    result["task_dtypes"] = sorted({str(s.get("dtype", "float32")).lower() for s in (task.input_specs or [])})

    try:
        solution = import_solution(args.solution)
    except Exception:
        result["status"] = "import_error"
        result["error"] = _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)

    entry = getattr(solution, task.entrypoint, None)
    if not callable(entry):
        result["status"] = "import_error"
        result["error"] = "solution defines no callable %r (entrypoint of task %s)" % (task.entrypoint, task.name)
        return _finish(result, args.out, t_start)

    reasons = suspicion_report(task, args.solution, solution)
    p_report = policy_report(task, args.solution, solution, task.policy)
    result["policy_findings"] = p_report["findings"]
    result["kernel_evidence"] = p_report["kernel_evidence"]
    # Legacy tasks and explicitly exempt runs record findings but are not blocked.
    blocking = [] if (args.policy_exempt or task.policy.get("torch_fused_exempt")) else list(p_report["errors"])
    if reasons:
        result["suspicious"] = True
        result["suspicious_reasons"] = reasons
        if strict or args.fail_on_suspicious:
            blocking.extend({"severity": "error", "code": "suspicious",
                             "message": r} for r in reasons)
    if blocking:
        result["status"] = "fail_policy"
        result["error"] = "; ".join(f.get("message", "") for f in blocking)
        result["policy_violations"] = blocking
        return _finish(result, args.out, t_start)

    # harness-enforced task requirements, e.g. TASK["require"] = ["triton"] (+ run-level --require)
    requires = [str(r).lower() for r in (task.meta.get("require") or [])]
    requires += [part.strip().lower() for part in str(args.require or "").split(",") if part.strip()]
    requires = sorted(set(requires))
    result["require"] = requires
    if requires:
        try:
            with open(args.solution, "r", encoding="utf-8", errors="replace") as fh:
                source_text = fh.read()
        except OSError:
            source_text = ""
        uses_triton = bool(re.search(r"import\s+triton|from\s+triton|@triton\.jit|\btl\.", source_text))
        result["solution_features"] = {"uses_triton": uses_triton}
        if "triton" in requires and not uses_triton and not args.ignore_require:
            result["status"] = "fail_constraint"
            result["error"] = ("task requires a triton (triton-ascend) implementation; the solution does "
                               "not import or use triton")
            return _finish(result, args.out, t_start)

    run = _wrap_inference(entry)
    warmup = args.warmup if args.warmup is not None else task.warmup
    rep = args.rep if args.rep is not None else task.rep
    trials = args.trials if args.trials is not None else task.num_trials
    flush = task.flush_l2 if args.flush_l2 is None else args.flush_l2

    allow_nonfinite = bool(task.policy.get("allow_nonfinite"))
    try:
        c_res = corr_mod.run_correctness(
            task, run, device, trials, policy=task.policy,
            policy_exempt=args.policy_exempt, allow_nonfinite=allow_nonfinite)
    except pol.PolicyViolation as exc:
        result["status"] = "fail_policy"
        result["error"] = "policy violation: %s%s" % (exc.code, (": " + exc.detail) if exc.detail else "")
        return _finish(result, args.out, t_start)
    except Exception:
        result["status"] = "crash"
        result["error"] = _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)
    result["correctness"] = c_res
    if not c_res.get("ok"):
        result["status"] = "fail_policy" if c_res.get("policy_violation") else "fail_correctness"
        result["error"] = c_res.get("reason")
        return _finish(result, args.out, t_start)

    held = heldout_report(task, run, device, task.policy, args.policy_exempt, allow_nonfinite)
    result["heldout"] = held
    if held and not held.get("ok"):
        result["status"] = "fail_holdout"
        result["error"] = held.get("reason")
        return _finish(result, args.out, t_start)

    try:
        train_cases = task.cases("train")
        inputs = task.case_inputs(train_cases[0], device, 0) if train_cases else task.make_inputs(device, 0)
        sol_t = bench_mod.benchmark(run, inputs, device, warmup, rep, flush,
                                    policy=task.policy, policy_exempt=args.policy_exempt)
        sol_t2 = bench_mod.benchmark(run, inputs, device, max(1, warmup // 2), max(20, rep // 2), flush,
                                     policy=task.policy, policy_exempt=args.policy_exempt)
    except pol.PolicyViolation as exc:
        result["status"] = "fail_policy"
        result["error"] = "policy violation during benchmark: %s%s" % (
            exc.code, (": " + exc.detail) if exc.detail else "")
        return _finish(result, args.out, t_start)
    except Exception:
        result["status"] = "bench_error"
        result["error"] = _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)

    b_kind = args.baseline or task.baseline
    baseline_fallback = None
    baseline_fn = None
    if b_kind != "eager":
        try:
            baseline_fn = _baseline_fn(task, b_kind, device)
        except Exception as exc:
            baseline_fallback = "%s baseline unavailable (%s: %s); using eager" % (
                b_kind, type(exc).__name__, str(exc)[:160])
        if baseline_fn is not None:
            sane, why = _baseline_sane(task, baseline_fn, device)
            if not sane:
                baseline_fallback = "%s baseline rejected (%s); using eager" % (b_kind, why)
                baseline_fn = None
    if baseline_fn is None:
        b_kind = "eager"
        baseline_fn = task.reference

    try:
        base_t = bench_mod.benchmark(_wrap_inference(baseline_fn), inputs, device, warmup, rep, flush)
    except Exception:
        if baseline_fallback is None and b_kind != "eager":
            baseline_fallback = "%s baseline failed at timing; using eager" % b_kind
            b_kind = "eager"
            try:
                base_t = bench_mod.benchmark(_wrap_inference(task.reference), inputs, device, warmup, rep, flush)
            except Exception:
                result["status"] = "bench_error"
                result["error"] = _tail(traceback.format_exc())
                return _finish(result, args.out, t_start)
        else:
            result["status"] = "bench_error"
            result["error"] = _tail(traceback.format_exc())
            return _finish(result, args.out, t_start)
    result["baseline_check"] = {"requested": str(args.baseline or task.baseline),
                                "used": b_kind, "fallback": baseline_fallback}

    speedup = base_t["median_ms"] / sol_t["median_ms"] if sol_t["median_ms"] > 0 else 0.0
    stability = abs(sol_t2["median_ms"] - sol_t["median_ms"]) / max(sol_t["median_ms"], 1e-9)

    # One timing row per declared training case.  With a shape_set the reported
    # speedup is the weight-normalized mean and every case is also kept so a
    # single-regressing shape cannot hide behind a good average.
    shape_cases = train_cases or [None]
    rows = []
    for idx, case in enumerate(shape_cases):
        if idx == 0:
            c_sol, c_base = sol_t, base_t
        else:
            try:
                c_inputs = task.case_inputs(case, device, idx)
                c_sol = bench_mod.benchmark(run, c_inputs, device, warmup, rep, flush,
                                            policy=task.policy, policy_exempt=args.policy_exempt)
                c_base = bench_mod.benchmark(_wrap_inference(baseline_fn), c_inputs, device,
                                             warmup, rep, flush)
            except pol.PolicyViolation as exc:
                result["status"] = "fail_policy"
                result["error"] = "policy violation during benchmark: %s%s" % (
                    exc.code, (": " + exc.detail) if exc.detail else "")
                return _finish(result, args.out, t_start)
            except Exception:
                result["status"] = "bench_error"
                result["error"] = _tail(traceback.format_exc())
                return _finish(result, args.out, t_start)
        c_speed = c_base["median_ms"] / c_sol["median_ms"] if c_sol["median_ms"] > 0 else 0.0
        rows.append({
            "name": (case or {}).get("name") or "default",
            "shape": (case or {}).get("shape"),
            "dtype": (case or {}).get("dtype"),
            "weight": float((case or {}).get("weight", 1.0)),
            "solution_ms": c_sol["median_ms"],
            "baseline_ms": c_base["median_ms"],
            "speedup": round(c_speed, 4),
            "samples_ms": c_sol.get("samples_ms"),
            "baseline_samples_ms": c_base.get("samples_ms"),
        })
    if sum(r["weight"] for r in rows) <= 0:
        for row in rows:
            row["weight"] = 1.0
    total_weight = sum(r["weight"] for r in rows) or float(len(rows))
    weighted_speedup = (sum(r["weight"] * r["speedup"] for r in rows) / total_weight
                        if len(rows) > 1 else speedup)
    worst = min(rows, key=lambda r: r["speedup"])
    if len(rows) > 1:
        speedup_ci = stats_mod.bootstrap_weighted_ratio([
            {"weight": r["weight"], "numerator": r["baseline_samples_ms"],
             "denominator": r["samples_ms"]} for r in rows
        ])
    else:
        speedup_ci = stats_mod.bootstrap_ratio_ci(base_t.get("samples_ms"), sol_t.get("samples_ms"))

    gate_min = task.meta.get("shape_gate_min_speedup")
    if gate_min is None and len(rows) > 1:
        gate_min = 0.9
    gate = None
    if gate_min is not None:
        gate = {"min_required": float(gate_min), "worst_case": worst["name"],
                "worst_speedup": worst["speedup"],
                "passed": bool(worst["speedup"] >= float(gate_min))}
    result["timing"] = {
        "solution": sol_t,
        "solution_second_pass": sol_t2,
        "baseline": base_t,
        "baseline_kind": b_kind,
        "baseline_fallback": baseline_fallback,
        "speedup": round(weighted_speedup, 4),
        "weighted_speedup": round(weighted_speedup, 4),
        "speedup_ci95": speedup_ci,
        "shape_cases": [{k: v for k, v in r.items()
                         if k not in ("samples_ms", "baseline_samples_ms")} for r in rows],
        "worst_case": {k: v for k, v in worst.items()
                       if k not in ("samples_ms", "baseline_samples_ms")},
        "shape_gate": gate,
        "stability_delta": round(stability, 4),
        "unit": "ms",
    }
    if gate and not gate["passed"]:
        result["status"] = "fail_shape_gate"
        result["error"] = ("worst shape %s regressed to %.3fx (gate %.3fx)"
                           % (gate["worst_case"], gate["worst_speedup"], gate["min_required"]))
        return _finish(result, args.out, t_start)
    roof = roofline_mod.estimate(getattr(task, "workload", {}), sol_t["median_ms"], task.meta)
    if roof:
        result["timing"]["roofline"] = roof

    # Strict tasks are audited for two timing escapes: cached outputs on the
    # repeated benchmark input, and work launched on a side stream.  The audit
    # itself is not used as the reported timing.
    audit = {"cache_suspected": False, "side_stream_suspected": False,
             "cache_checked": False, "side_stream_checked": False}
    if strict:
        try:
            train_cases = task.cases("train")
            if train_cases:
                fresh_inputs = task.case_inputs(train_cases[0], device, 11)
            else:
                fresh_inputs = task.make_inputs(device, int(trials) + 11)
            fresh_t = bench_mod.benchmark(
                run, fresh_inputs, device, max(1, warmup // 2),
                max(5, min(int(rep), 20)), flush,
                policy=task.policy, policy_exempt=args.policy_exempt)
            audit["cache_checked"] = True
            audit["fresh_median_ms"] = fresh_t["median_ms"]
            audit["cache_suspected"] = pol.detect_cache(
                sol_t["median_ms"], fresh_t["median_ms"])
        except pol.PolicyViolation as exc:
            result["status"] = "fail_policy"
            result["error"] = "policy violation during cache audit: %s%s" % (
                exc.code, (": " + exc.detail) if exc.detail else "")
            return _finish(result, args.out, t_start)
        except Exception as exc:
            audit["cache_error"] = "%s: %s" % (type(exc).__name__, exc)
        if dev.is_accelerator(device):
            wall = []
            try:
                for _ in range(5):
                    dev.sync(device)
                    t0 = time.perf_counter()
                    with pol.policy_guard(task.policy, args.policy_exempt):
                        run(*inputs)
                    dev.sync(device)
                    wall.append((time.perf_counter() - t0) * 1000.0)
                audit["side_stream_checked"] = True
                audit["wall_median_ms"] = round(statistics.median(wall), 6)
                audit["side_stream_suspected"] = pol.detect_side_stream(
                    sol_t["median_ms"], statistics.median(wall))
            except Exception as exc:
                audit["side_stream_error"] = "%s: %s" % (type(exc).__name__, exc)
    result["policy_audit"] = audit
    if audit["cache_suspected"]:
        result["status"] = "fail_policy"
        result["error"] = ("benchmark cache suspected: fixed-input median %.4fms vs fresh-input "
                           "median %.4fms" % (sol_t["median_ms"], audit.get("fresh_median_ms") or 0.0))
        return _finish(result, args.out, t_start)
    if audit["side_stream_suspected"]:
        result["status"] = "fail_policy"
        result["error"] = ("side stream suspected: event median %.4fms vs synchronized wall "
                           "median %.4fms" % (sol_t["median_ms"], audit.get("wall_median_ms") or 0.0))
        return _finish(result, args.out, t_start)
    result["status"] = "pass"
    return _finish(result, args.out, t_start)


if __name__ == "__main__":
    raise SystemExit(main())