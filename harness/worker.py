"""Worker: evaluate ONE (task, solution) pair and write a JSON result.

Always run through harness.runner (subprocess + timeout) so that crashes, hangs and
CUDA/NPU errors in agent-written code stay isolated from the harness.

Evaluation order matters for anti-cheat reasons:

1. build K rotating benchmark input sets;
2. time the baseline *before* the solution is imported (import-time monkeypatching
   of torch cannot slow the reference down);
3. import the solution and diff torch attribute ids before/after;
4. run correctness, then time the solution on the rotating, perturbed inputs;
5. probe: mutate the inputs once more and compare a fresh output with the
   reference (a memoized output is now stale -> fail_correctness);
6. on accelerators, audit event timing against globally synchronized wall timing
   (side-stream escapes).

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
from . import guard as guard_mod
from . import measure as measure_mod
from . import noise as noise_mod
from .tasks import TaskLoadError, load_task

# Bound at import: the report must not depend on the module attribute `time`.
_PERF_COUNTER = time.perf_counter
_WALL_CLOCK = time.time


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
        if vf is None and hasattr(val, "__globals__"):
            vf = getattr(val, "__globals__", {}).get("__file__")
        if vf and os.path.abspath(str(vf)) == task_file:
            reasons.append("solution holds a reference to the task module (%s)" % name)
    return reasons


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


def make_input_sets(task, device: str, k: int = bench_mod.DEFAULT_INPUT_SETS) -> list:
    """K independently generated input sets, rotated by the benchmark.

    Total size is capped so a large task does not multiply its memory footprint
    by K; the cap can only reduce the set count, never below one.
    """
    sets = []
    total = 0
    for trial in range(max(1, int(k))):
        args = task.make_inputs(device, trial)
        sets.append(list(args))
        total += bench_mod.input_bytes(args)
        if total >= bench_mod.MAX_INPUT_BYTES:
            break
    return sets


def time_baseline(task, input_sets, device: str, warmup: int, rep: int, flush: bool,
                  requested: str):
    """Resolve + sanity-check + time the baseline.

    Returns (timing, kind, fallback, error, fn).  The timing is the pre-import
    guard measurement; the ABAB rounds later re-measure the same function.
    """
    b_kind = requested or task.baseline
    fallback = None
    baseline_fn = None
    if b_kind != "eager":
        try:
            baseline_fn = _baseline_fn(task, b_kind, device)
        except Exception as exc:
            fallback = "%s baseline unavailable (%s: %s); using eager" % (
                b_kind, type(exc).__name__, str(exc)[:160])
        if baseline_fn is not None:
            sane, why = _baseline_sane(task, baseline_fn, device)
            if not sane:
                fallback = "%s baseline rejected (%s); using eager" % (b_kind, why)
                baseline_fn = None
    if baseline_fn is None:
        b_kind = "eager"
        baseline_fn = task.reference

    try:
        base_t = bench_mod.benchmark(_wrap_inference(baseline_fn), input_sets=input_sets,
                                     device=device, warmup=warmup, rep=rep, flush_l2=flush)
        return base_t, b_kind, fallback, None, baseline_fn
    except Exception:
        error = traceback.format_exc()
    if b_kind != "eager":
        fallback = fallback or "%s baseline failed at timing; using eager" % b_kind
        try:
            base_t = bench_mod.benchmark(_wrap_inference(task.reference), input_sets=input_sets,
                                         device=device, warmup=warmup, rep=rep, flush_l2=flush)
            return base_t, "eager", fallback, None, task.reference
        except Exception:
            error = traceback.format_exc()
    return None, b_kind, fallback, error, baseline_fn


def measure_holdout(task, run, baseline_fn, device: str, warmup: int, rep: int,
                    flush: bool, epsilon) -> dict:
    """Correctness + ABAB speedup on the first holdout case (promotion gate)."""
    case = task.holdout_cases[0]
    out = {"case": case.get("name") or "holdout", "ok": False, "speedup": None,
           "speedup_ci": None, "n_rounds": 0, "epsilon": epsilon, "success": False}
    try:
        with task.holdout_constants(case):
            sets = [list(task.holdout_inputs(case, device, trial)) for trial in range(2)]
            inputs = sets[0]
            ref = _wrap_inference(task.reference)(*inputs)
            dev.sync(device)
            got = run(*inputs)
            dev.sync(device)
        cmp = corr_mod.compare_outputs(
            ref, got, float(task.tolerance.get("atol", 1e-5)),
            float(task.tolerance.get("rtol", 1e-5)))
        out["ok"] = bool(cmp.get("ok"))
        if not out["ok"]:
            out["reason"] = cmp.get("reason") or "holdout correctness failed"
            return out
        with task.holdout_constants(case):
            measured = measure_mod.ab_benchmark(
                run, _wrap_inference(baseline_fn), sets, device, rounds=6,
                warmup=max(3, int(warmup) // 2), rep=max(5, int(rep) // 3), flush_l2=flush)
        ci = measured["speedup_ci"]
        unstable = bool(epsilon is not None and measured["stability_delta"] > epsilon)
        if unstable and ci and len(ci) == 2 and epsilon:
            factor = min(3.0, 1.0 + float(measured["stability_delta"]) / float(epsilon))
            mid = float(measured["speedup"])
            ci = [round(mid - (mid - float(ci[0])) * factor, 4),
                  round(mid + (float(ci[1]) - mid) * factor, 4)]
            out["speedup_ci_raw"] = list(measured["speedup_ci"])
            out["ci_widened"] = round(factor, 3)
        out.update({
            "speedup": measured["speedup"],
            "speedup_ci": ci,
            "n_rounds": measured["n_rounds"],
            "stability_delta": measured["stability_delta"],
            "unstable": unstable,
            # Same rule as the training measurement: instability widens the CI,
            # it does not veto a lower bound that still clears 1 + epsilon.
            "success": noise_mod.classify(ci, epsilon) == "success",
        })
    except Exception as exc:
        out["reason"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def cache_probe(task, run, device: str, input_sets) -> dict:
    """Change the inputs once more and check the output was recomputed."""
    probe = {"applicable": False, "ok": None, "reason": None}
    changed = bench_mod.perturb_inputs(input_sets[0], device, step=10 ** 6)
    probe["applicable"] = bool(changed)
    if not changed:
        probe["reason"] = "no floating-point inputs to perturb"
        return probe
    try:
        inputs = input_sets[0]
        ref = _wrap_inference(task.reference)(*inputs)
        dev.sync(device)
        out = run(*inputs)
        dev.sync(device)
        cmp = corr_mod.compare_outputs(
            ref, out, float(task.tolerance.get("atol", 1e-5)),
            float(task.tolerance.get("rtol", 1e-5)))
    except Exception as exc:
        probe["ok"] = False
        probe["reason"] = "%s: %s" % (type(exc).__name__, exc)
        return probe
    probe["ok"] = bool(cmp.get("ok"))
    probe["reason"] = cmp.get("reason")
    return probe


def side_stream_audit(fn, inputs, device: str, reps: int = 5) -> dict:
    """Current-stream event timing vs globally synchronized wall timing."""
    out = {"checked": False, "suspected": False}
    mod = dev.backend_module(device)
    if mod is None or not hasattr(mod, "Event"):
        out["reason"] = "backend provides no timing events"
        return out
    try:
        events = []
        for _ in range(int(reps)):
            start = mod.Event(enable_timing=True)
            end = mod.Event(enable_timing=True)
            start.record()
            fn(*inputs)
            end.record()
            events.append((start, end))
        mod.synchronize()
        event_ms = [float(s.elapsed_time(e)) for s, e in events]
        wall = []
        for _ in range(int(reps)):
            dev.sync(device)
            t0 = _PERF_COUNTER()
            fn(*inputs)
            dev.sync(device)
            wall.append((_PERF_COUNTER() - t0) * 1000.0)
        event_med = float(statistics.median(event_ms))
        wall_med = float(statistics.median(wall))
        out.update({
            "checked": True,
            "event_median_ms": round(event_med, 6),
            "wall_median_ms": round(wall_med, 6),
            "suspected": guard_mod.detect_side_stream(event_med, wall_med),
        })
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


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
        ci = t.get("speedup_ci")
        if ci:
            lines.append("speedup CI95: [%.3f, %.3f] | rounds=%s | rep/round=%s" % (
                ci[0], ci[1], t.get("n_rounds"), t.get("rep_per_round")))
        eps = t.get("epsilon")
        lines.append("stability gate: epsilon=%s unstable=%s" % (
            "uncalibrated" if eps is None else "%.4f" % eps, t.get("unstable", False)))
        lines.append("timing inputs: %d rotating set(s), perturbed=%s, device_sync=%s" % (
            t.get("n_input_sets", 1), t.get("perturbed", False), t.get("device_sync", False)))
        hold = t.get("holdout") or {}
        if hold:
            lines.append("hold-out: %s | correctness=%s | speedup=%s | CI=%s | success=%s" % (
                hold.get("case"), "ok" if hold.get("ok") else "FAILED",
                "%.3fx" % hold["speedup"] if hold.get("speedup") is not None else "-",
                hold.get("speedup_ci"), hold.get("success")))
        pre = result.get("baseline_pre_import") or {}
        if pre.get("median_ms"):
            lines.append("baseline(pre-import guard)=%.4f ms" % pre["median_ms"])
        probe = t.get("cache_probe") or {}
        if probe.get("applicable"):
            lines.append("cache probe: %s%s" % (
                "ok" if probe.get("ok") else "FAILED",
                "" if probe.get("ok") else " (%s)" % (probe.get("reason") or "output mismatch")))
    side = result.get("side_stream_audit") or {}
    if side.get("checked"):
        lines.append("side-stream audit: event=%.4f ms vs wall=%.4f ms -> %s" % (
            side.get("event_median_ms", 0.0), side.get("wall_median_ms", 0.0),
            "SUSPECTED" if side.get("suspected") else "ok"))
    if result.get("torch_attr_changes"):
        lines.append("torch attributes changed during import: %s" % ", ".join(result["torch_attr_changes"]))
    if result.get("runtime_attr_changes"):
        lines.append("evaluator attributes changed: %s" % ", ".join(result["runtime_attr_changes"]))
    if result.get("harness_attr_changes"):
        lines.append("harness module attributes changed: %s" % ", ".join(result["harness_attr_changes"]))
    clock = result.get("clock_audit") or {}
    if clock.get("checked"):
        lines.append("clock audit: %d anomaly iteration(s)%s" % (
            clock.get("anomalies", 0),
            " -> SUSPICIOUS" if clock.get("suspicious") else ""))
    if result.get("suspicious"):
        lines.append("suspicious: %s" % "; ".join(result.get("suspicious_reasons") or []))
    if result.get("status") != "pass" and result.get("error"):
        lines.append("error: %s" % result["error"])
    return "\n".join(lines)


def _finish(result: dict, out_path: str, t_start: float) -> dict:
    result["wall_time_s"] = round(_WALL_CLOCK() - t_start, 2)
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
    ap.add_argument("--rounds", type=int, default=measure_mod.DEFAULT_ROUNDS,
                    help="ABAB measurement rounds (default %d)" % measure_mod.DEFAULT_ROUNDS)
    ap.add_argument("--holdout-config", default=None,
                    help="holdout shape config (default configs/holdout.json)")
    ap.add_argument("--no-holdout", action="store_true", help="skip the holdout promotion gate")
    ap.add_argument("--baseline", default=None, help="override the task baseline: eager | compile")
    ap.add_argument("--flush-l2", dest="flush_l2", action="store_true", default=None)
    ap.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    ap.add_argument("--fail-on-suspicious", action="store_true",
                    help="kept for compatibility; evaluator tampering is always blocking now")
    ap.add_argument("--ignore-require", action="store_true",
                    help="skip harness-enforced task requirements (require: [...])")
    ap.add_argument("--require", default="",
                    help="comma-separated run-level requirements, merged with the task's own")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--iter", type=int, default=0)
    args = ap.parse_args(argv)

    t_start = _WALL_CLOCK()
    device = dev.resolve_device(args.device)
    # Read the calibrated noise amplitude before the solution is imported: a
    # solution cannot loosen this run's gate by rewriting configs/noise.json.
    epsilon = noise_mod.epsilon_for(device=device)
    result = {
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
        "torch_attr_changes": [],
        "runtime_attr_changes": [],
        "harness_attr_changes": [],
        "clock_audit": None,
        "source_audit": [],
        "correctness": None,
        "timing": None,
        "timing_inputs": None,
        "side_stream_audit": None,
        "baseline_error": None,
        "baseline_pre_import": None,
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
        task = load_task(args.task, holdout_config=args.holdout_config)
    except Exception as exc:
        result["status"] = "task_error"
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        return _finish(result, args.out, t_start)

    result["task"] = task.name
    result["level"] = task.level
    result["task_tags"] = [str(t) for t in (task.meta.get("tags") or [])]
    result["task_op_class"] = str(task.meta.get("op_class") or "")
    result["task_hardware"] = str(task.meta.get("hardware") or "")
    result["task_dtypes"] = sorted({str(s.get("dtype", "float32")).lower() for s in (task.input_specs or [])})

    warmup = args.warmup if args.warmup is not None else task.warmup
    rep = args.rep if args.rep is not None else task.rep
    trials = args.trials if args.trials is not None else task.num_trials
    flush = task.flush_l2 if args.flush_l2 is None else args.flush_l2

    # (1) Benchmark inputs are generated once and shared by baseline and solution.
    try:
        input_sets = make_input_sets(task, device)
    except Exception:
        result["status"] = "task_error"
        result["error"] = "could not build benchmark inputs: %s" % _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)
    result["timing_inputs"] = {
        "n_sets": len(input_sets),
        "bytes": sum(bench_mod.input_bytes(s) for s in input_sets),
    }

    # (2) Baseline is timed before the solution import: import-time monkeypatching
    #     of torch cannot slow the reference down.  This is a guard measurement;
    #     the ABAB rounds below produce the reported speedup.
    base_pre, b_kind, baseline_fallback, baseline_error, baseline_fn = time_baseline(
        task, input_sets, device, warmup, rep, flush, args.baseline)
    result["baseline_check"] = {"requested": str(args.baseline or task.baseline),
                                "used": b_kind, "fallback": baseline_fallback}
    result["baseline_pre_import"] = base_pre
    if baseline_error:
        result["baseline_error"] = _tail(baseline_error)

    # (3) Evaluator-attribute snapshot around the import.
    torch_before = guard_mod.snapshot_torch()
    runtime_before = guard_mod.snapshot_runtime()
    harness_before = guard_mod.snapshot_harness()
    try:
        solution = import_solution(args.solution)
    except Exception:
        result["status"] = "import_error"
        result["error"] = _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)
    torch_changes = guard_mod.diff_snapshots(torch_before, guard_mod.snapshot_torch())
    runtime_changes = guard_mod.diff_snapshots(runtime_before, guard_mod.snapshot_runtime())
    harness_changes = guard_mod.diff_snapshots(harness_before, guard_mod.snapshot_harness())
    result["torch_attr_changes"] = torch_changes
    result["runtime_attr_changes"] = runtime_changes
    result["harness_attr_changes"] = harness_changes

    entry = getattr(solution, task.entrypoint, None)
    if not callable(entry):
        result["status"] = "import_error"
        result["error"] = "solution defines no callable %r (entrypoint of task %s)" % (task.entrypoint, task.name)
        return _finish(result, args.out, t_start)

    try:
        with open(args.solution, "r", encoding="utf-8", errors="replace") as fh:
            source_text = fh.read()
    except OSError:
        source_text = ""

    reasons = suspicion_report(task, args.solution, solution)
    findings = guard_mod.audit_source(source_text, os.path.splitext(os.path.basename(task.path))[0])
    result["source_audit"] = findings
    reasons.extend(guard_mod.blocking_reasons(findings))
    if torch_changes:
        reasons.append("monkeypatched torch attributes during import: %s" % ", ".join(torch_changes))
    if runtime_changes:
        reasons.append("monkeypatched evaluator attributes during import: %s" % ", ".join(runtime_changes))
    if harness_changes:
        reasons.append("monkeypatched harness module attributes during import: %s"
                       % ", ".join(harness_changes))
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        result["suspicious"] = True
        result["suspicious_reasons"] = reasons
        result["status"] = "fail_suspicious"
        result["error"] = "; ".join(reasons)
        return _finish(result, args.out, t_start)

    # harness-enforced task requirements, e.g. TASK["require"] = ["triton"] (+ run-level --require)
    requires = [str(r).lower() for r in (task.meta.get("require") or [])]
    requires += [part.strip().lower() for part in str(args.require or "").split(",") if part.strip()]
    requires = sorted(set(requires))
    result["require"] = requires
    if requires:
        uses_triton = bool(re.search(r"import\s+triton|from\s+triton|@triton\.jit|\btl\.", source_text))
        result["solution_features"] = {"uses_triton": uses_triton}
        if "triton" in requires and not uses_triton and not args.ignore_require:
            result["status"] = "fail_constraint"
            result["error"] = ("task requires a triton (triton-ascend) implementation; the solution does "
                               "not import or use triton")
            return _finish(result, args.out, t_start)

    run = _wrap_inference(entry)

    try:
        c_res = corr_mod.run_correctness(task, run, device, trials)
    except Exception:
        result["status"] = "crash"
        result["error"] = _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)
    result["correctness"] = c_res
    if not c_res.get("ok"):
        result["status"] = "fail_correctness"
        result["error"] = c_res.get("reason")
        return _finish(result, args.out, t_start)

    if baseline_error:
        result["status"] = "bench_error"
        result["error"] = "baseline timing failed before the solution import: %s" % result["baseline_error"]
        return _finish(result, args.out, t_start)

    # (4) ABAB interleaved measurement: baseline and candidate alternate for N
    #     rounds, each round contributes one median per side, and the per-round
    #     speedups are bootstrapped to a 95% CI.
    try:
        measurement = measure_mod.ab_benchmark(
            run, _wrap_inference(baseline_fn), input_sets, device,
            rounds=args.rounds, warmup=warmup, rep=rep, flush_l2=flush)
    except Exception:
        result["status"] = "bench_error"
        result["error"] = _tail(traceback.format_exc())
        return _finish(result, args.out, t_start)

    # A late monkeypatch (inside run()) must not be able to touch the baseline.
    torch_after_run = guard_mod.snapshot_torch()
    runtime_after_run = guard_mod.snapshot_runtime()
    harness_after_run = guard_mod.snapshot_harness()
    torch_late = guard_mod.diff_snapshots(torch_before, torch_after_run)
    runtime_late = guard_mod.diff_snapshots(runtime_before, runtime_after_run)
    harness_late = guard_mod.diff_snapshots(harness_before, harness_after_run)
    if torch_late or runtime_late or harness_late:
        result["torch_attr_changes"] = sorted(set(torch_changes + torch_late))
        result["runtime_attr_changes"] = sorted(set(runtime_changes + runtime_late))
        result["harness_attr_changes"] = sorted(set(harness_changes + harness_late))
        late = torch_late + runtime_late + harness_late
        result["status"] = "fail_suspicious"
        result["error"] = "monkeypatched evaluator attributes during the run: %s" % ", ".join(late)
        return _finish(result, args.out, t_start)

    anomalies = int(measurement.get("clock_anomalies") or 0)
    result["clock_audit"] = {"checked": True, "anomalies": anomalies,
                             "suspicious": anomalies > 0}
    if anomalies:
        result["status"] = "fail_suspicious"
        result["error"] = ("clock audit: the module-level clock disagreed with the bound clock "
                           "on %d timed iteration(s)" % anomalies)
        return _finish(result, args.out, t_start)

    unstable = bool(epsilon is not None and measurement["stability_delta"] > epsilon)
    # Instability must not throw away a real gain: it widens the interval and
    # marks the row.  A widened lower bound that still clears 1 + epsilon is
    # still evidence (the success rule itself lives in harness.noise.classify).
    raw_ci = measurement.get("speedup_ci")
    if unstable and raw_ci and len(raw_ci) == 2 and epsilon:
        factor = min(3.0, 1.0 + float(measurement["stability_delta"]) / float(epsilon))
        mid = float(measurement["speedup"])
        lo = mid - (mid - float(raw_ci[0])) * factor
        hi = mid + (float(raw_ci[1]) - mid) * factor
        measurement["speedup_ci_raw"] = list(raw_ci)
        measurement["speedup_ci"] = [round(lo, 4), round(hi, 4)]
        measurement["ci_widened"] = round(factor, 3)
    holdout = None
    train_ci = measurement.get("speedup_ci")
    if (not args.no_holdout and task.holdout_cases and train_ci
            and (epsilon is None or train_ci[0] > 1.0 + epsilon)):
        holdout = measure_holdout(task, run, baseline_fn, device, warmup, rep, flush, epsilon)
    result["timing"] = {
        **measurement,
        "baseline_kind": b_kind,
        "baseline_fallback": baseline_fallback,
        "epsilon": epsilon,
        "unstable": unstable,
        "holdout": holdout,
        "n_input_sets": len(input_sets),
        "perturbed": True,
        "device_sync": dev.is_accelerator(device),
        "unit": "ms",
    }

    # (5) Post-timing probe: inputs changed after the last timed call, so a
    #     memoized output no longer matches the reference of the current input.
    probe = cache_probe(task, run, device, input_sets)
    result["timing"]["cache_probe"] = probe
    if probe.get("applicable") and not probe.get("ok"):
        result["status"] = "fail_correctness"
        result["error"] = ("post-timing probe: solution returned a stale result after its "
                           "inputs changed (%s)" % (probe.get("reason") or "output mismatch"))
        return _finish(result, args.out, t_start)

    # (6) Side-stream audit on accelerators (CPU has no streams to escape to).
    if dev.is_accelerator(device):
        audit = side_stream_audit(run, input_sets[0], device)
        result["side_stream_audit"] = audit
        if audit.get("suspected"):
            result["status"] = "fail_suspicious"
            result["error"] = ("side stream suspected: current-stream event median %.4f ms vs "
                               "global-sync wall median %.4f ms"
                               % (audit.get("event_median_ms", 0.0), audit.get("wall_median_ms", 0.0)))
            return _finish(result, args.out, t_start)

    result["status"] = "pass_unstable" if unstable else "pass"
    return _finish(result, args.out, t_start)


if __name__ == "__main__":
    raise SystemExit(main())
