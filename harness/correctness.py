"""Correctness checking, framework-agnostic (torch tensors or numpy arrays).

Anti-cheat by construction: inputs are re-generated per trial, the solution receives
fresh clones, and mutation of its inputs is reported as a failure.
"""
from __future__ import annotations

from .policy import PolicyViolation, normalize_policy, policy_guard


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


def _is_floating(x) -> bool:
    root = _module_root(x)
    try:
        if root == "torch":
            return bool(x.is_floating_point())
        if root == "numpy":
            return bool(getattr(x.dtype, "kind", "") == "f")
    except Exception:
        return False
    return False


def _bool_all(x) -> bool:
    try:
        return bool(x.all())
    except Exception:
        return bool(x)


def _nonfinite_masks(x):
    root = _module_root(x)
    try:
        if root == "torch":
            import torch
            return torch.isnan(x), torch.isinf(x)
        if root == "numpy":
            import numpy as np
            return np.isnan(x), np.isinf(x)
    except Exception:
        pass
    return None, None


def compare_outputs(ref, out, atol: float, rtol: float, allow_nonfinite: bool = False) -> dict:
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

        r_nan, r_inf = _nonfinite_masks(r) if _is_floating(r) else (None, None)
        o_nan, o_inf = _nonfinite_masks(o) if _is_floating(o) else (None, None)
        ref_nonfinite = bool(r_nan is not None and ((int(r_nan.sum()) > 0) or (int(r_inf.sum()) > 0)))
        if ref_nonfinite and not allow_nonfinite:
            return {"ok": False, "reason": "reference contains non-finite values (task misconfigured)"}
        if ref_nonfinite:
            # NaN/Inf pattern and sign must match exactly; finite lanes use atol/rtol.
            if not _bool_all(r_nan == o_nan):
                return {"ok": False, "reason": "output %d: NaN pattern differs from reference" % idx}
            if not _bool_all(r_inf == o_inf):
                return {"ok": False, "reason": "output %d: Inf pattern differs from reference" % idx}
            if int(r_inf.sum()) if r_inf is not None else 0:
                if not _bool_all((r[r_inf] == o[o_inf])):
                    return {"ok": False, "reason": "output %d: Inf sign differs from reference" % idx}
            finite = ~(r_nan | r_inf)
            if int(finite.sum()) == 0:
                continue
            diff = abs(o - r)
            diff[~finite] = 0
            limit = atol + rtol * abs(r)
            bad = int(((diff > limit) & finite).sum())
            abs_err = float(diff.max()) if _numel(diff) else 0.0
            rel_err = float((diff / (abs(r) + 1e-12)).max()) if _numel(diff) else 0.0
        else:
            if not _all_finite(o):
                return {"ok": False, "reason": "output %d contains NaN/Inf" % idx}
            diff = abs(o - r)
            limit = atol + rtol * abs(r)
            bad = int((diff > limit).sum())
            abs_err = float(diff.max()) if _numel(diff) else 0.0
            rel_err = float((diff / (abs(r) + 1e-12)).max()) if _numel(diff) else 0.0
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


def run_correctness(task, solution_fn, device: str, num_trials=None,
                    policy=None, policy_exempt: bool = False,
                    allow_nonfinite: bool = False) -> dict:
    n = int(num_trials if num_trials is not None else task.num_trials)
    atol = float(task.tolerance.get("atol", 1e-5))
    rtol = float(task.tolerance.get("rtol", 1e-5))
    trials = []
    overall = {"ok": True, "reason": None, "trials": trials,
               "max_abs_err": 0.0, "max_rel_err": 0.0, "policy_violation": None}

    train_cases = task.cases("train")
    for trial in range(n):
        if train_cases:
            inputs = task.case_inputs(train_cases[trial % len(train_cases)], device, trial)
        else:
            inputs = task.make_inputs(device, trial)
        sol_inputs = [_clone(x) for x in inputs]
        snapshot = [_clone(x) for x in inputs]
        ref = task.reference(*inputs)
        try:
            with policy_guard(policy, policy_exempt):
                out = solution_fn(*sol_inputs)
        except PolicyViolation as exc:
            overall["ok"] = False
            overall["policy_violation"] = {"code": exc.code, "detail": exc.detail}
            overall["reason"] = "policy violation: %s%s" % (
                exc.code, (": " + exc.detail) if exc.detail else "")
            trials.append({"trial": trial, "ok": False, "reason": overall["reason"],
                           "mutated_inputs": False, "policy_violation": overall["policy_violation"]})
            break
        except Exception as exc:
            overall["ok"] = False
            overall["reason"] = "trial %d: solution raised %s: %s" % (trial, type(exc).__name__, exc)
            trials.append({"trial": trial, "ok": False, "reason": overall["reason"], "mutated_inputs": False})
            break
        res = compare_outputs(ref, out, atol, rtol, allow_nonfinite=allow_nonfinite)
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