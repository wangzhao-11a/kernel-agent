"""Evaluator-integrity guards.

Four independent anti-cheat signals used by the worker:

1. ``snapshot_torch``: importing a solution must not replace torch's compute
   functions (a solution that slows the baseline down or speeds itself up by
   monkeypatching ``torch.softmax`` & co. is caught by an ``id()`` snapshot
   taken immediately before and after the import).
2. ``snapshot_harness``: the solution must not rebind anything inside the
   evaluator modules either — e.g. swapping ``harness.benchmark.time`` for a
   fake clock.  Only callables and modules are compared, so legitimate lazy
   caches (``harness.device._TORCH``) are not flagged.
3. ``audit_source``: static AST findings for solutions that reach into the
   evaluator (task/harness imports, ``reference()`` calls, dynamic imports,
   frame/GC/ctypes introspection).
4. ``detect_side_stream``: current-stream event timing vs a globally
   synchronized wall clock; work on a side stream escapes the former only.

``harness.benchmark`` additionally binds the real ``perf_counter`` at import and
records whether the module-level clock agrees with it, which the worker uses as
a runtime clock audit.
"""
from __future__ import annotations

import ast
import types
from importlib import import_module

# Compute entry points a solution could replace to poison the baseline.  Keep
# this list to functions the reference implementations actually use.
_TORCH_FUNCS = (
    "softmax", "log_softmax", "exp", "log", "sqrt", "rsqrt", "pow",
    "sum", "mean", "max", "min", "amax", "amin", "cumsum",
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "linear",
    "add", "sub", "mul", "div", "abs", "neg",
    "tanh", "sigmoid", "relu", "gelu", "silu", "erf",
    "layer_norm", "group_norm", "batch_norm", "rms_norm", "norm",
    "where", "cat", "stack", "topk", "sort", "gather", "scatter",
    "index_select", "scaled_dot_product_attention",
)
_FUNCTIONAL_FUNCS = (
    "softmax", "log_softmax", "relu", "gelu", "silu", "sigmoid", "tanh",
    "linear", "layer_norm", "group_norm", "rms_norm", "normalize", "pad",
    "scaled_dot_product_attention", "embedding", "dropout",
)
_TENSOR_METHODS = (
    "__matmul__", "__add__", "__sub__", "__mul__", "__truediv__",
    "sum", "mean", "max", "min", "exp", "softmax", "contiguous", "clone",
)

BLOCKING_CODES = frozenset({
    "reference_import", "reference_call", "dynamic_import", "dynamic_exec",
    "module_introspection", "evaluator_introspection",
})

# Modules whose attributes are part of the measurement path.  A solution that
# rebinds any callable or module here is doing evaluator surgery.
_HARNESS_MODULES = (
    "harness.benchmark", "harness.measure", "harness.stats", "harness.noise",
    "harness.correctness", "harness.guard", "harness.worker", "harness.device",
    "harness.tasks",
)


def snapshot_torch() -> dict:
    """Map torch attribute name -> id(), for every present guarded attribute."""
    from . import device as dev

    t = dev.torch_module()
    if t is None:
        return {}
    out = {}
    for name in _TORCH_FUNCS:
        value = getattr(t, name, None)
        if value is not None:
            out["torch.%s" % name] = id(value)
    nn = getattr(t, "nn", None)
    functional = getattr(nn, "functional", None) if nn is not None else None
    if functional is not None:
        for name in _FUNCTIONAL_FUNCS:
            value = getattr(functional, name, None)
            if value is not None:
                out["torch.nn.functional.%s" % name] = id(value)
    tensor_type = getattr(t, "Tensor", None)
    if tensor_type is not None:
        for name in _TENSOR_METHODS:
            value = getattr(tensor_type, name, None)
            if value is not None:
                out["torch.Tensor.%s" % name] = id(value)
    return out


def snapshot_runtime() -> dict:
    """Map evaluator runtime helpers name -> id() (patched clocks/statistics)."""
    import statistics
    import time

    return {
        "time.perf_counter": id(time.perf_counter),
        "time.time": id(time.time),
        "statistics.median": id(statistics.median),
        "statistics.fmean": id(statistics.fmean),
    }


def snapshot_harness() -> dict:
    """Map harness module attribute name -> id(), for callables and modules only.

    Plain data attributes are skipped on purpose: lazy caches such as
    ``harness.device._TORCH`` legitimately change when torch is first imported.
    """
    out = {}
    for name in _HARNESS_MODULES:
        try:
            module = import_module(name)
        except Exception:
            continue
        for attr, value in vars(module).items():
            if isinstance(value, types.ModuleType) or callable(value):
                out["%s.%s" % (name, attr)] = id(value)
    return out


def diff_snapshots(before: dict, after: dict) -> list:
    """Names whose function object changed or disappeared during an import."""
    changed = []
    for name, value in (before or {}).items():
        new = (after or {}).get(name)
        if new is None:
            changed.append("%s (removed)" % name)
        elif new != value:
            changed.append(name)
    return changed


def _call_name(node) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        bits = []
        cur = f
        while isinstance(cur, ast.Attribute):
            bits.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            bits.append(cur.id)
        return ".".join(reversed(bits))
    return ""


def audit_source(source: str, task_stem: str = "") -> list:
    """Static evaluator-tampering findings (conservative; each is a hard signal)."""
    findings = []
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return [{"severity": "error", "code": "syntax", "message": str(exc)}]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "harness" or name.startswith("harness."):
                    findings.append({"severity": "error", "code": "reference_import",
                                     "message": "imports evaluator/harness: %s" % name})
                if task_stem and name.split(".")[-1] == task_stem:
                    findings.append({"severity": "error", "code": "reference_import",
                                     "message": "imports the task module: %s" % name})
                if name in ("importlib", "inspect", "gc", "ctypes"):
                    findings.append({"severity": "error", "code": "evaluator_introspection",
                                     "message": "imports %s (evaluator introspection)" % name})
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "harness" or module.startswith("harness."):
                findings.append({"severity": "error", "code": "reference_import",
                                 "message": "imports evaluator/harness: %s" % module})
            if task_stem and module.split(".")[-1] == task_stem:
                findings.append({"severity": "error", "code": "reference_import",
                                 "message": "imports the task module: %s" % module})
            if module.split(".")[0] in ("importlib", "inspect", "gc", "ctypes"):
                findings.append({"severity": "error", "code": "evaluator_introspection",
                                 "message": "imports from %s (evaluator introspection)" % module})
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "reference" or name.endswith(".reference"):
                findings.append({"severity": "error", "code": "reference_call",
                                 "message": "calls reference(): %s" % name})
            elif (name in ("getattr", "builtins.getattr") and len(node.args) >= 2
                  and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value == "reference"):
                findings.append({"severity": "error", "code": "reference_call",
                                 "message": "resolves reference() via getattr"})
            if any(part in name for part in ("spec_from_file_location", "module_from_spec",
                                             "exec_module", "run_path")):
                findings.append({"severity": "error", "code": "dynamic_import",
                                 "message": "loads another module dynamically: %s" % name})
            if name in ("exec", "eval", "compile", "builtins.exec", "builtins.eval"):
                findings.append({"severity": "error", "code": "dynamic_exec",
                                 "message": "uses dynamic code execution: %s" % name})
            if name in ("__import__", "builtins.__import__"):
                findings.append({"severity": "error", "code": "dynamic_import",
                                 "message": "uses __import__: %s" % name})
            if name.startswith(("importlib.", "inspect.")) or name in (
                    "sys._getframe", "sys._current_frames", "gc.get_objects",
                    "gc.get_referrers", "gc.get_referents"):
                findings.append({"severity": "error", "code": "evaluator_introspection",
                                 "message": "evaluator introspection via %s" % name})
        elif isinstance(node, ast.Attribute):
            if node.attr == "reference":
                findings.append({"severity": "warn", "code": "reference_ref",
                                 "message": "references task.reference"})
            if (node.attr == "modules" and isinstance(node.value, ast.Name)
                    and node.value.id == "sys"):
                findings.append({"severity": "error", "code": "module_introspection",
                                 "message": "walks sys.modules (looks for evaluator modules)"})
            if node.attr in ("_getframe", "_current_frames"):
                findings.append({"severity": "error", "code": "evaluator_introspection",
                                 "message": "walks interpreter frames (%s)" % node.attr})

    seen, out = set(), []
    for finding in findings:
        key = (finding["severity"], finding["code"], finding["message"])
        if key not in seen:
            seen.add(key)
            out.append(finding)
    return out


def blocking_reasons(findings: list) -> list:
    return [f["message"] for f in findings or () if f.get("code") in BLOCKING_CODES]


def detect_side_stream(event_ms: float, wall_ms: float,
                       min_wall_ms: float = 0.5, ratio: float = 0.2) -> bool:
    """Work on a side stream escapes a current-stream event: wall >> event."""
    try:
        event_ms, wall_ms = float(event_ms), float(wall_ms)
    except (TypeError, ValueError):
        return False
    return wall_ms >= min_wall_ms and event_ms > 0 and event_ms < ratio * wall_ms
