"""Negative tests for the Phase-1 evaluation policy.

Six classes of cheating/mis-measurement are exercised on CPU:
  1. calling the task reference (dynamic task load)
  2. native torch compute (runtime TorchDispatchMode; AST is auxiliary)
  3. cross-trial output cache
  4. side stream (timing-consistency detector; CPU mechanism only)
  5. fixed-seed/shape overfit (held-out shape)
  6. NaN/boundary inputs (held-out NaN case)

Run: python scripts/policy_selftest.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness.policy import detect_cache, detect_side_stream  # noqa: E402
from harness.runner import run_evaluation  # noqa: E402

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))
        traceback.print_exc()
        return
    RESULTS["pass"].append(name)
    print("PASS  %s" % name)


def write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


POLICY = '''
    "schema_version": 1,
    "policy": {
        "forbid_native_compute": True,
        "require_kernel": False,
        "torch_fused_exempt": False,
        "forbid_reference": True,
        "forbid_numpy_compute": False,
        "allow_nonfinite": True,
        "reason": "policy selftest"
    },
    "shape_set": [
        {"name": "train", "shape": [64], "dtype": "float32", "weight": 0.7,
         "heldout": False, "input_specs": [{"kind": "randn"}]},
        {"name": "heldout_96", "shape": [96], "dtype": "float32", "weight": 0.3,
         "heldout": True, "input_specs": [{"kind": "randn"}]},
        {"name": "heldout_nan", "shape": [64], "dtype": "float32", "weight": 0.0,
         "heldout": True, "input_specs": [{"kind": "nan"}]}
    ],
'''

TASK_SOURCE = '''"""Strict policy task used by scripts/policy_selftest.py (CPU only)."""
import numpy as np

TASK = {
    "name": "strict_add",
    "level": 0,
    "description": "Policy self-test: y = x + 1.",
    "num_trials": 3,
    "warmup": 1,
    "rep": 5,
%s}

def make_inputs_for_case(device, trial, case):
    rng = np.random.default_rng(int(case.get("seed", 1234)) + int(trial))
    shape = tuple(case.get("shape") or (64,))
    kind = (case.get("input_specs") or [{}])[0].get("kind", "randn")
    x = rng.standard_normal(shape).astype(np.float32)
    if kind == "nan" and x.size:
        x.reshape(-1)[::7] = np.nan
    return (x,)

def reference(x):
    return x + 1
''' % POLICY

KERNEL_TASK_SOURCE = TASK_SOURCE.replace(
    '"require_kernel": False', '"require_kernel": True'
)


def make_task(tmp):
    return write(os.path.join(tmp, "tasks", "strict_add.py"), TASK_SOURCE)


def make_kernel_task(tmp):
    return write(os.path.join(tmp, "tasks", "strict_kernel.py"), KERNEL_TASK_SOURCE)


def run(tmp, task, name, source):
    path = write(os.path.join(tmp, "solutions", name + ".py"), source)
    return run_evaluation(
        task=task, solution=path, device="cpu", timeout=60,
        out_dir=os.path.join(tmp, "out", name),
        results_file=os.path.join(tmp, "out", "results.jsonl"),
        quiet=True,
    )


def check_reference(tmp):
    task = make_task(tmp)
    source = '''import importlib.util
_spec = importlib.util.spec_from_file_location("the_task", %r)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

def run(x):
    return _mod.reference(x)
''' % task
    res = run(tmp, task, "ref_call", source)
    assert res["status"] == "fail_policy", res.get("status")
    assert any(f.get("code") in ("dynamic_import", "reference_import")
               for f in (res.get("policy_violations") or [])), res.get("policy_violations")


def check_native(tmp):
    task = make_task(tmp)
    # Static AST is intentionally blind here: use getattr to reach aten::add.
    source = '''import torch

def run(x):
    return getattr(torch, "add")(torch.as_tensor(x), 1)
'''
    res = run(tmp, task, "native", source)
    assert res["status"] == "fail_policy", res.get("status")
    text = (res.get("error") or "").lower()
    assert "native_compute" in text or "aten::" in text, res.get("error")


def check_cache(tmp):
    task = make_task(tmp)
    source = '''import numpy as np
_cache = {}

def run(x):
    key = (tuple(x.shape), str(x.dtype))
    if key not in _cache:
        _cache[key] = x + 1
    return _cache[key]
'''
    res = run(tmp, task, "cache", source)
    assert res["status"] == "fail_correctness", res.get("status")
    assert "tolerance" in (res.get("error") or ""), res.get("error")


def check_overfit(tmp):
    task = make_task(tmp)
    source = '''import numpy as np

def run(x):
    if x.shape[-1] == 64:
        return x + 1
    return np.zeros_like(x)
'''
    res = run(tmp, task, "overfit", source)
    assert res["status"] == "fail_holdout", res.get("status")
    assert "held-out" in (res.get("error") or "").lower(), res.get("error")


def check_nan(tmp):
    task = make_task(tmp)
    source = '''import numpy as np

def run(x):
    if np.isnan(x).any():
        return np.zeros_like(x)
    return x + 1
'''
    res = run(tmp, task, "nan", source)
    assert res["status"] == "fail_holdout", res.get("status")
    assert "nan" in (res.get("error") or "").lower() or "non-finite" in (
        res.get("error") or "").lower(), res.get("error")


def check_side_stream():
    # CPU mechanism test: the real detector is used by worker only on accelerators.
    assert detect_side_stream(0.01, 2.0) is True
    assert detect_side_stream(0.02, 0.05) is False
    assert detect_cache(0.01, 1.0) is True
    assert detect_cache(1.0, 1.1) is False
    from harness.policy import kernel_evidence
    triton_src = '''import triton
import triton.language as tl

@triton.jit
def k(x):
    pass

def run(x):
    k(x)
'''
    assert kernel_evidence(triton_src)["triton"] is True
    ext_src = "from torch.utils.cpp_extension import load_inline\nload_inline(name='k', cpp_sources='')\n"
    assert kernel_evidence(ext_src)["extension"] is True


def check_kernel_evidence(tmp):
    task = make_kernel_task(tmp)
    source = '''def run(x):
    return x + 1
'''
    res = run(tmp, task, "no_kernel", source)
    assert res["status"] == "fail_policy", res.get("status")
    assert "kernel" in (res.get("error") or "").lower(), res.get("error")


def check_jsonl_migration(tmp):
    from scripts.migrate_jsonl_schema import migrate_file
    path = write(os.path.join(tmp, "jsonl", "old.jsonl"),
                 '{"iter": 1}\n{"iter": 2, "schema_version": 9}\nnot-json\n')
    report = migrate_file(path, 1)
    assert report["changed"] == 1, report
    assert report["invalid"] == 1, report
    assert report["backup"] and os.path.isfile(report["backup"]), report
    rows = [line for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    assert '"schema_version": 1' in rows[0], rows
    assert '"schema_version": 9' in rows[1], rows


def main():
    tmp = tempfile.mkdtemp(prefix="kernel-agent-policy-")
    print("temp dir: %s" % tmp)
    check("policy: reference call is rejected", lambda: check_reference(tmp))
    check("policy: native torch compute is rejected at runtime", lambda: check_native(tmp))
    check("policy: cross-trial cache is rejected", lambda: check_cache(tmp))
    check("policy: fixed-shape overfit is rejected by held-out", lambda: check_overfit(tmp))
    check("policy: NaN/boundary candidate is rejected", lambda: check_nan(tmp))
    check("policy: side-stream/cache detectors (CPU mechanism)", check_side_stream)
    check("policy: kernel evidence is required", lambda: check_kernel_evidence(tmp))
    check("schema: JSONL migration adds version + backup", lambda: check_jsonl_migration(tmp))
    print("note: side-stream/cache conclusions are CPU mechanism checks only")
    print("\n==================================================================")
    print("passed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    print("temp artifacts: %s" % tmp)
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())