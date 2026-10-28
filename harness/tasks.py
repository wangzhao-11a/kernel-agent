"""Task loading.

A task is a standalone Python module:

    TASK = {                       # metadata
        "name": "vector_add",
        "level": 1,
        "description": "...",
        "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "num_trials": 3, "warmup": 25, "rep": 100,
        "entrypoint": "run",       # name the solution must define
        "baseline": "eager",       # eager | compile
        "expected_speedup": 1.1,   # optional reporting target
        "input_specs": [...],      # optional; else define make_inputs()
    }

    def make_inputs(device, trial):   # optional when input_specs is set
        ...

    def reference(*inputs):           # the reference implementation
        ...

The harness only reads these three things (TASK, make_inputs, reference) and never
executes anything else from the task module.
"""
from __future__ import annotations

import importlib.util
import os

from .policy import normalize_policy

DEFAULT_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}

# Phase-1 schema for a weighted, held-out-aware shape set:
#
#   "shape_set": [
#     {"name": "prefill_2k", "shape": [2, 2048], "dtype": "float16", "dim": -1,
#      "weight": 0.7, "heldout": false,
#      "input_specs": [{"kind": "randn", "shape": [2, 2048], "dtype": "float16"}]},
#     {"name": "decode_1k", "shape": [1024, 128], "dtype": "float16", "dim": -1,
#      "weight": 0.3, "heldout": false, "input_specs": [...]},
#     {"name": "heldout_odd", "shape": [17, 257], "dtype": "float16", "dim": -1,
#      "weight": 0.0, "heldout": true, "input_specs": [...]}
#   ]
#
# `input_specs` is optional when the task module defines
# make_inputs_for_case(device, trial, case).  A task without shape_set keeps the
# legacy single-shape make_inputs(device, trial) contract.  Moment weighting and
# the worst-shape gate are deliberately left for Phase 4; Phase 1 only fixes the
# schema, held-out split and per-case seeds.
SHAPE_SET_KEYS = ("name", "shape", "dtype", "dim", "weight", "heldout", "input_specs", "seed")


def parse_shape_set(raw) -> list:
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        raise TaskLoadError("TASK['shape_set'] must be a list")
    cases, names = [], set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TaskLoadError("shape_set[%d] must be an object" % i)
        name = str(item.get("name") or "case%d" % i)
        if name in names:
            raise TaskLoadError("duplicate shape_set name: %s" % name)
        names.add(name)
        shape = item.get("shape")
        if shape is not None:
            try:
                shape = [int(x) for x in shape]
            except (TypeError, ValueError):
                raise TaskLoadError("shape_set[%s].shape must be a list of ints" % name)
        specs = item.get("input_specs")
        if specs is not None and not isinstance(specs, (list, tuple)):
            raise TaskLoadError("shape_set[%s].input_specs must be a list" % name)
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            raise TaskLoadError("shape_set[%s].weight must be numeric" % name)
        if weight < 0:
            raise TaskLoadError("shape_set[%s].weight must be >= 0" % name)
        cases.append({
            "name": name,
            "shape": shape,
            "dtype": str(item.get("dtype") or ""),
            "dim": item.get("dim", -1),
            "weight": weight,
            "heldout": bool(item.get("heldout", False)),
            "input_specs": list(specs) if specs else None,
            "seed": int(item.get("seed", 1234)),
        })
    if not any(not c["heldout"] for c in cases):
        raise TaskLoadError("shape_set needs at least one training (heldout=false) case")
    return cases

_DTYPE_NAMES = {
    "float32": "float32", "float": "float32", "fp32": "float32",
    "float16": "float16", "half": "float16", "fp16": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "int32": "int32", "int64": "int64", "int8": "int8", "bool": "bool",
}


class TaskLoadError(Exception):
    pass


class Task:
    def __init__(self, path: str, module, meta: dict):
        self.path = os.path.abspath(path)
        self.module = module
        self.meta = meta
        self.name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
        self.level = int(meta.get("level", 1))
        self.description = meta.get("description", "")
        self.entrypoint = meta.get("entrypoint", "run")
        self.tolerance = dict(DEFAULT_TOLERANCE)
        self.tolerance.update(meta.get("tolerance", {}) or {})
        self.num_trials = int(meta.get("num_trials", 3))
        self.warmup = int(meta.get("warmup", 25))
        self.rep = int(meta.get("rep", 100))
        self.baseline = meta.get("baseline", "eager")
        self.flush_l2 = bool(meta.get("flush_l2", True))
        self.expected_speedup = float(meta.get("expected_speedup", 1.0))
        self.input_specs = meta.get("input_specs")
        self.schema_version = int(meta.get("schema_version", 1))
        self.policy = normalize_policy(meta)
        self.shape_set = parse_shape_set(meta.get("shape_set"))
        self.source = _read_source(self.path)

    @property
    def reference(self):
        fn = getattr(self.module, "reference", None)
        if not callable(fn):
            raise TaskLoadError("task %s defines no callable reference()" % self.path)
        return fn

    def make_inputs(self, device: str, trial: int, case: dict | None = None):
        fn = getattr(self.module, "make_inputs_for_case", None)
        if case is not None and callable(fn):
            return tuple(fn(device, trial, case))
        if case is not None and case.get("input_specs"):
            return _inputs_from_specs(case["input_specs"], device, trial,
                                      seed=case.get("seed"), default_shape=case.get("shape"))
        fn = getattr(self.module, "make_inputs", None)
        if callable(fn):
            return tuple(fn(device, trial))
        specs = (case or {}).get("input_specs") or self.input_specs
        if specs:
            return _inputs_from_specs(specs, device, trial,
                                      seed=(case or {}).get("seed"),
                                      default_shape=(case or {}).get("shape"))
        raise TaskLoadError(
            "task %s defines neither make_inputs() nor TASK['input_specs']" % self.path
        )

    def cases(self, split: str | None = None) -> list:
        if not self.shape_set:
            return []
        if split is None:
            return list(self.shape_set)
        want_heldout = split == "heldout"
        return [c for c in self.shape_set if c["heldout"] == want_heldout]

    def case_inputs(self, case: dict, device: str, trial: int):
        return self.make_inputs(device, trial, case=case)


def load_task(path: str) -> Task:
    if not os.path.isfile(path):
        raise TaskLoadError("task file not found: %s" % path)
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location("kernel_agent_task_%s" % stem, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise TaskLoadError("cannot load task module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = dict(getattr(module, "TASK", {}) or {})
    return Task(path, module, meta)


def _read_source(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _inputs_from_specs(specs, device: str, trial: int, seed=None, default_shape=None):
    from .device import torch_module

    t = torch_module()
    if t is None:
        raise TaskLoadError("TASK['input_specs'] needs torch; define make_inputs() instead")
    gen = t.Generator(device="cpu")
    gen.manual_seed(int(seed if seed is not None else 1234) + int(trial))
    out = []
    for s in specs:
        dtype = _DTYPE_NAMES.get(str(s.get("dtype", "float32")).lower(), "float32")
        dt = getattr(t, dtype)
        shape = tuple(int(d) for d in (s.get("shape") or default_shape or ()))
        kind = s.get("kind", "randn")
        if kind == "randn":
            x = t.randn(shape, generator=gen, dtype=dt)
        elif kind == "rand":
            x = t.rand(shape, generator=gen, dtype=dt)
        elif kind == "zeros":
            x = t.zeros(shape, dtype=dt)
        elif kind == "ones":
            x = t.ones(shape, dtype=dt)
        elif kind == "full":
            x = t.full(shape, float(s.get("value", 0.0)), dtype=dt)
        elif kind == "empty":
            x = t.empty(shape, dtype=dt)
        elif kind == "arange":
            x = t.arange(int(s.get("n", shape[0] if shape else 0)), dtype=dt).reshape(shape)
        elif kind in ("nan", "inf", "extreme"):
            fdt = dt if dt in (t.float16, t.float32, t.float64, t.bfloat16) else t.float32
            x = t.randn(shape, generator=gen, dtype=fdt)
            flat = x.view(-1)
            if flat.numel():
                if kind == "nan":
                    flat[::7] = float("nan")
                elif kind == "inf":
                    flat[::11] = float("inf")
                    flat[1::13] = float("-inf")
                else:
                    lim = float(t.finfo(fdt).max) / 8.0
                    flat.mul_(lim)
            x = x.to(dt)
        else:
            raise TaskLoadError("unknown input kind: %r" % kind)
        scale = s.get("scale")
        if scale is not None and x.is_floating_point():
            x = x * float(scale)
        out.append(x.to(device))
    return tuple(out)