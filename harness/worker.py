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
import time
import traceback
from datetime import datetime, timezone

from . import benchmark as bench_mod
from . import correctness as corr_mod
from . import device as dev
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
    if reasons:
        result["suspicious"] = True
        result["suspicious_reasons"] = reasons
        if args.fail_on_suspicious:
            result["status"] = "fail_suspicious"
            result["error"] = "; ".join(reasons)
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

    try:
        inputs = task.make_inputs(device, 0)
        sol_t = bench_mod.benchmark(run, inputs, device, warmup, rep, flush)
        sol_t2 = bench_mod.benchmark(run, inputs, device, max(1, warmup // 2), max(20, rep // 2), flush)
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
    result["timing"] = {
        "solution": sol_t,
        "solution_second_pass": sol_t2,
        "baseline": base_t,
        "baseline_kind": b_kind,
        "baseline_fallback": baseline_fallback,
        "speedup": round(speedup, 4),
        "stability_delta": round(stability, 4),
        "unit": "ms",
    }
    result["status"] = "pass"
    return _finish(result, args.out, t_start)


if __name__ == "__main__":
    raise SystemExit(main())