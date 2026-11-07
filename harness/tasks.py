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

import contextlib
import importlib.util
import json
import os

DEFAULT_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}
HOLDOUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "holdout.json")

_DTYPE_NAMES = {
    "float32": "float32", "float": "float32", "fp32": "float32",
    "float16": "float16", "half": "float16", "fp16": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "int32": "int32", "int64": "int64", "int8": "int8", "bool": "bool",
}


class TaskLoadError(Exception):
    pass


class Task:
    def __init__(self, path: str, module, meta: dict, holdout_cases=None):
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
        self.holdout_cases = list(holdout_cases or [])
        self.source = _read_source(self.path)

    @property
    def reference(self):
        fn = getattr(self.module, "reference", None)
        if not callable(fn):
            raise TaskLoadError("task %s defines no callable reference()" % self.path)
        return fn

    def make_inputs(self, device: str, trial: int):
        fn = getattr(self.module, "make_inputs", None)
        if callable(fn):
            return tuple(fn(device, trial))
        if self.input_specs:
            return _inputs_from_specs(self.input_specs, device, trial)
        raise TaskLoadError(
            "task %s defines neither make_inputs() nor TASK['input_specs']" % self.path
        )

    def holdout_inputs(self, case: dict, device: str, trial: int):
        """Inputs for one holdout case.

        Holdout cases live in configs/holdout.json (not in the task source), so the
        agent's prompt and training loop never see them.  Two forms:

        - ``{"shapes": [...]}``: overrides `input_specs` shapes (spec-based tasks);
        - ``{"constants": {"T": 64}}``: overrides module-level shape constants for
          the duration of the call (tasks with a custom ``make_inputs``).
        """
        case = case or {}
        shapes = list(case.get("shapes") or [])
        if self.input_specs and shapes:
            specs = []
            for index, spec in enumerate(self.input_specs):
                item = dict(spec)
                shape = shapes[index] if index < len(shapes) else shapes[-1]
                item["shape"] = list(shape)
                specs.append(item)
            return _inputs_from_specs(specs, device, 10000 + int(trial))
        with self.holdout_constants(case):
            fn = getattr(self.module, "make_inputs", None)
            if not callable(fn):
                raise TaskLoadError(
                    "task %s defines neither input_specs nor make_inputs() for holdout" % self.path)
            return tuple(fn(device, 10000 + int(trial)))

    @contextlib.contextmanager
    def holdout_constants(self, case: dict):
        """Temporarily apply a holdout case's module constants (e.g. T=64)."""
        constants = dict((case or {}).get("constants") or {})
        if not constants:
            yield
            return
        saved = {}
        for name, value in constants.items():
            if hasattr(self.module, name):
                saved[name] = getattr(self.module, name)
                setattr(self.module, name, value)
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(self.module, name, value)


def _holdout_cases(task_name: str, holdout_config: str = None) -> list:
    path = holdout_config or HOLDOUT_PATH
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    entry = ((data or {}).get("tasks") or {}).get(task_name) or {}
    return list(entry.get("cases") or [])


def load_task(path: str, holdout_config: str = None) -> Task:
    if not os.path.isfile(path):
        raise TaskLoadError("task file not found: %s" % path)
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location("kernel_agent_task_%s" % stem, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise TaskLoadError("cannot load task module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = dict(getattr(module, "TASK", {}) or {})
    name = meta.get("name") or stem
    return Task(path, module, meta, holdout_cases=_holdout_cases(name, holdout_config))


def _read_source(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _inputs_from_specs(specs, device: str, trial: int):
    from .device import torch_module

    t = torch_module()
    if t is None:
        raise TaskLoadError("TASK['input_specs'] needs torch; define make_inputs() instead")
    gen = t.Generator(device="cpu")
    gen.manual_seed(1234 + int(trial))
    out = []
    for s in specs:
        dtype = _DTYPE_NAMES.get(str(s.get("dtype", "float32")).lower(), "float32")
        dt = getattr(t, dtype)
        shape = tuple(int(d) for d in s.get("shape", ()))
        kind = s.get("kind", "randn")
        if kind == "randn":
            x = t.randn(shape, generator=gen, dtype=dt)
        elif kind == "rand":
            x = t.rand(shape, generator=gen, dtype=dt)
        elif kind == "zeros":
            x = t.zeros(shape, dtype=dt)
        elif kind == "arange":
            x = t.arange(int(s.get("n", shape[0] if shape else 0)), dtype=dt).reshape(shape)
        else:
            raise TaskLoadError("unknown input kind: %r" % kind)
        scale = s.get("scale")
        if scale is not None and x.is_floating_point():
            x = x * float(scale)
        out.append(x.to(device))
    return tuple(out)