"""Correctness checking, framework-agnostic (torch tensors or numpy arrays).

Anti-cheat by construction: inputs are re-generated per trial, the solution receives
fresh clones, and mutation of its inputs is reported as a failure.
"""
from __future__ import annotations


def _module_root(x) -> str:
    return type(x).__module__.split(".")[0]


def leaves(x) -> list:
    """Flatten nested tuples/lists/dicts of tensors into a flat list."""
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(leaves(item))
        return out
    if isinstance(x, dict):
        out = []
        for key in sorted(x.keys(), key=str):
            out.extend(leaves(x[key]))
        return out
    return [x]


def _clone(x):
    if hasattr(x, "clone"):  # torch tensor
        return x.clone()
    if hasattr(x, "copy"):  # numpy array
        return x.copy()
    return x


def _equal(a, b) -> bool:
    try:
        return bool((a == b).all())
    except Exception:
        return a == b


def _all_finite(x) -> bool:
    root = _module_root(x)
    try:
        if root == "torch":
            import torch

            return bool(torch.isfinite(x).all())
        if root == "numpy":
            import numpy as np

            return bool(np.isfinite(x).all())
    except Exception:
        return True
    return True


def _numel(x) -> int:
    n = getattr(x, "numel", None)
    if callable(n):
        return int(n())
    try:
        return int(x.size)
    except Exception:
        return 0


def compare_outputs(ref, out, atol: float, rtol: float) -> dict:
    rl = leaves(ref)
    ol = leaves(out)
    if len(rl) != len(ol):
        return {"ok": False, "reason": "output arity mismatch: reference has %d value(s), solution returned %d" % (len(rl), len(ol))}

    max_abs = 0.0
    max_rel = 0.0
    for idx, (r, o) in enumerate(zip(rl, ol)):
        r_shape = tuple(getattr(r, "shape", ()))
        o_shape = tuple(getattr(o, "shape", ()))
        if r_shape != o_shape:
            return {"ok": False, "reason": "output %d shape mismatch: reference %s vs solution %s" % (idx, r_shape, o_shape)}
        r_dtype = getattr(r, "dtype", None)
        o_dtype = getattr(o, "dtype", None)
        if r_dtype != o_dtype:
            return {"ok": False, "reason": "output %d dtype mismatch: reference %s vs solution %s" % (idx, r_dtype, o_dtype)}
        if not _all_finite(r):
            return {"ok": False, "reason": "reference contains non-finite values (task misconfigured)"}
        if not _all_finite(o):
            return {"ok": False, "reason": "output %d contains NaN/Inf" % idx}
        diff = abs(o - r)
        limit = atol + rtol * abs(r)
        bad = int((diff > limit).sum())
        abs_err = float(diff.max()) if _numel(diff) else 0.0
        with_err = abs(diff)
        rel_err = float((with_err / (abs(r) + 1e-12)).max()) if _numel(diff) else 0.0
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        if bad:
            return {
                "ok": False,
                "reason": "output %d: %d/%d elements exceed tolerance (max_abs_err=%.3e, atol=%.1e, rtol=%.1e)"
                % (idx, bad, _numel(diff), abs_err, atol, rtol),
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
            }
    return {"ok": True, "reason": None, "max_abs_err": max_abs, "max_rel_err": max_rel}


def run_correctness(task, solution_fn, device: str, num_trials=None) -> dict:
    n = int(num_trials if num_trials is not None else task.num_trials)
    atol = float(task.tolerance.get("atol", 1e-5))
    rtol = float(task.tolerance.get("rtol", 1e-5))
    trials = []
    overall = {"ok": True, "reason": None, "trials": trials, "max_abs_err": 0.0, "max_rel_err": 0.0}

    for trial in range(n):
        inputs = task.make_inputs(device, trial)
        sol_inputs = [_clone(x) for x in inputs]
        snapshot = [_clone(x) for x in inputs]
        ref = task.reference(*inputs)
        try:
            out = solution_fn(*sol_inputs)
        except Exception as exc:
            overall["ok"] = False
            overall["reason"] = "trial %d: solution raised %s: %s" % (trial, type(exc).__name__, exc)
            trials.append({"trial": trial, "ok": False, "reason": overall["reason"], "mutated_inputs": False})
            break
        res = compare_outputs(ref, out, atol, rtol)
        mutated = any(not _equal(a, b) for a, b in zip(snapshot, sol_inputs))
        overall["max_abs_err"] = max(overall["max_abs_err"], float(res.get("max_abs_err") or 0.0))
        overall["max_rel_err"] = max(overall["max_rel_err"], float(res.get("max_rel_err") or 0.0))
        trials.append({
            "trial": trial,
            "ok": bool(res["ok"]),
            "reason": res.get("reason"),
            "mutated_inputs": bool(mutated),
            "max_abs_err": float(res.get("max_abs_err", 0.0)),
        })
        if not res["ok"]:
            overall["ok"] = False
            overall["reason"] = "trial %d: %s" % (trial, res["reason"])
            break
        if mutated:
            overall["ok"] = False
            overall["reason"] = "trial %d: solution mutated its inputs (not allowed)" % trial
            break

    return overall