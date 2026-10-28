"""Task-level evaluation policy: forbid native compute and require real kernels.

The RSI layer is only as trustworthy as the measurement.  A task can opt into a
strict policy:

    "policy": {
        "forbid_native_compute": true,   # torch/numpy compute ops are rejected
        "require_kernel": true,          # a triton/C++/CANN kernel must be present
        "torch_fused_exempt": false,     # explicit legacy exemption
        "forbid_reference": true,        # never call the task's reference()
        "allow_nonfinite": false,        # NaN/Inf references are errors unless set
        "allowed_torch_ops": ["aten::copy_"],  # optional extra whitelist
        "reason": "kernel-only task"
    }

Runtime enforcement is a TorchDispatchMode with an allow-list of allocation,
view, copy and metadata ops.  Static AST checks are auxiliary: they catch
reference imports and give the agent an actionable error before a run.
"""
from __future__ import annotations

import ast
import contextlib
import os
import re

DEFAULT_POLICY = {
    "forbid_native_compute": False,
    "require_kernel": False,
    "torch_fused_exempt": True,   # legacy tasks are torch/triton reference tasks
    "forbid_reference": True,
    "forbid_numpy_compute": False,
    "allow_nonfinite": False,
    "allowed_torch_ops": [],
    "reason": "",
}

# Non-compute ATen ops are safe to allow: allocation, metadata, views, copies,
# synchronization and device transfer.  Anything not listed is rejected when
# forbid_native_compute is on.  Keep this list intentionally small.
_ALLOWED_ATEN = {
    "aten::empty", "aten::empty_like", "aten::empty_strided",
    "aten::new_empty", "aten::new_empty_strided",
    "aten::zeros", "aten::zeros_like", "aten::ones", "aten::ones_like",
    "aten::full", "aten::full_like", "aten::new_zeros", "aten::new_ones",
    "aten::new_full", "aten::empty_permuted",
    "aten::to", "aten::_to_copy", "aten::copy_", "aten::clone",
    "aten::contiguous", "aten::detach", "aten::alias",
    "aten::view", "aten::_unsafe_view", "aten::view_as", "aten::reshape",
    "aten::reshape_as", "aten::as_strided", "aten::permute", "aten::transpose",
    "aten::t", "aten::slice", "aten::select", "aten::narrow", "aten::expand",
    "aten::expand_as", "aten::unsqueeze", "aten::squeeze", "aten::flatten",
    "aten::set_", "aten::resize_", "aten::resize_as_", "aten::record_stream",
    "aten::is_contiguous", "aten::numel", "aten::size", "aten::stride",
    "aten::dim", "aten::is_floating_point", "aten::device", "aten::dtype",
    "aten::_local_scalar_dense", "aten::zero_", "aten::fill_",
    "aten::pin_memory", "aten::share_memory_", "aten::is_nonzero",
}

# Native compute calls that are unambiguously cheating in a kernel-only task.
# Used by the auxiliary static scan and the C++ review.
_TORCH_COMPUTE = re.compile(
    r"\btorch\.(?:matmul|mm|bmm|add|sub|mul|div|pow|exp|log|sum|mean|softmax|"
    r"silu|sigmoid|gelu|relu|tanh|layer_norm|rms_norm|norm|conv\w*|"
    r"einsum|scaled_dot_product_attention|baddbmm|addmm|linear|topk|sort|gather)\b"
)
_NUMPY_COMPUTE = re.compile(
    r"\b(?:np|numpy)\.(?:add|subtract|multiply|divide|power|exp|log|sum|mean|"
    r"matmul|dot|einsum|softmax|tanh|maximum|minimum|where|sort|argsort)\b"
)


class PolicyViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail) if detail else code)


def normalize_policy(meta: dict | None) -> dict:
    """Merge a task's policy with legacy defaults and validate contradictions."""
    raw = dict((meta or {}).get("policy") or {})
    p = dict(DEFAULT_POLICY)
    p.update(raw)
    p["forbid_native_compute"] = bool(p["forbid_native_compute"])
    p["require_kernel"] = bool(p["require_kernel"])
    p["torch_fused_exempt"] = bool(p["torch_fused_exempt"])
    p["forbid_reference"] = bool(p["forbid_reference"])
    p["forbid_numpy_compute"] = bool(p["forbid_numpy_compute"])
    p["allow_nonfinite"] = bool(p["allow_nonfinite"])
    p["allowed_torch_ops"] = [str(x) for x in (p.get("allowed_torch_ops") or [])]
    if p["torch_fused_exempt"]:
        p["forbid_native_compute"] = False
        p["require_kernel"] = False
    if p["require_kernel"]:
        p["forbid_native_compute"] = True
    if p["forbid_native_compute"]:
        p["torch_fused_exempt"] = False
    return p


def _op_name(func) -> str:
    schema = getattr(func, "_schema", None)
    name = getattr(schema, "name", "") if schema is not None else ""
    if not name:
        name = str(func)
    # aten::add.Tensor -> aten::add
    return name.split(".", 1)[0]


def allowed_torch_op(name: str, extra=()) -> bool:
    base = name.split(".", 1)[0]
    if base in _ALLOWED_ATEN:
        return True
    for item in extra or ():
        item = str(item).split(".", 1)[0]
        if item and (base == item or base.startswith(item.rstrip("*"))):
            return True
    return False


@contextlib.contextmanager
def policy_guard(policy: dict | None, exempt: bool = False):
    """Reject native torch compute while the candidate runs.

    The dispatch mode only wraps the candidate call; the reference and the
    harness's own accounting run outside it.  On a machine without torch this is
    a no-op (the static scan still applies).
    """
    p = normalize_policy({"policy": policy} if policy and "policy" not in policy else policy)
    if exempt or not p["forbid_native_compute"]:
        yield
        return
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:
        yield
        return

    class _PolicyMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = _op_name(func)
            if not allowed_torch_op(name, p.get("allowed_torch_ops")):
                raise PolicyViolation("native_compute", name)
            return func(*args, **(kwargs or {}))

    with _PolicyMode():
        yield


def _decorator_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


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


def kernel_evidence(source: str, module=None) -> dict:
    """Auxiliary evidence that a real kernel (not torch eager) exists in the file."""
    out = {"triton": False, "extension": False, "cann": False, "any": False}
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return out
    triton_kernel_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "cpp_extension" in alias.name:
                    out["extension"] = True
        elif isinstance(node, ast.ImportFrom):
            if "cpp_extension" in (node.module or ""):
                out["extension"] = True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if _decorator_name(dec) == "jit" or _call_name(dec).endswith("triton.jit"):
                    triton_kernel_names.add(node.name)
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if "cpp_extension" in name and ("load" in name or "load_inline" in name):
                out["extension"] = True
            if name.endswith("_C") or "cann_bench" in name:
                out["cann"] = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in triton_kernel_names:
            out["triton"] = True
            break
    if module is not None:
        for name, value in vars(module).items():
            if callable(value) and getattr(value, "__module__", "").startswith("triton"):
                out["triton"] = True
    out["any"] = out["triton"] or out["extension"] or out["cann"]
    return out


def audit_source(source: str, task_stem: str = "") -> list:
    """Static findings: reference access, native compute, and kernel evidence.

    These are auxiliary to the runtime dispatch mode; they are intentionally
    conservative (a finding is a signal, not a proof).
    """
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
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "harness" or module.startswith("harness."):
                findings.append({"severity": "error", "code": "reference_import",
                                 "message": "imports evaluator/harness: %s" % module})
            if task_stem and module.split(".")[-1] == task_stem:
                findings.append({"severity": "error", "code": "reference_import",
                                 "message": "imports the task module: %s" % module})
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "reference" or name.endswith(".reference"):
                findings.append({"severity": "error", "code": "reference_call",
                                 "message": "calls reference(): %s" % name})
            if any(part in name for part in ("spec_from_file_location", "module_from_spec",
                                             "run_path", "exec_module")):
                findings.append({"severity": "error", "code": "dynamic_import",
                                 "message": "loads another module dynamically: %s" % name})
            if name in ("exec", "eval", "compile", "builtins.exec", "builtins.eval"):
                findings.append({"severity": "error", "code": "dynamic_exec",
                                 "message": "uses dynamic code execution: %s" % name})
            if name.startswith("torch.") or name.startswith("torch.nn.functional."):
                if _TORCH_COMPUTE.search(name):
                    findings.append({"severity": "error", "code": "native_compute",
                                     "message": "calls native torch compute: %s" % name})
            if name.startswith("np.") or name.startswith("numpy."):
                if _NUMPY_COMPUTE.search(name):
                    findings.append({"severity": "error", "code": "numpy_compute",
                                     "message": "calls numpy compute: %s" % name})
        elif isinstance(node, ast.Attribute):
            if node.attr == "reference":
                findings.append({"severity": "warn", "code": "reference_ref",
                                 "message": "references task.reference"})
    # de-duplicate exact findings
    seen, out = set(), []
    for f in findings:
        key = (f["severity"], f["code"], f["message"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def cpp_policy_findings(files: dict) -> list:
    """Scan edited C++ sources for vendor/eager compute escapes (auxiliary)."""
    findings = []
    banned = re.compile(
        r"\b(?:aclnn[A-Z]\w*|torch::(?:matmul|mm|bmm|add|mul|div|softmax|silu|"
        r"sigmoid|gelu|sum|mean|exp|linear)|at::(?:matmul|mm|add|mul|softmax))\b")
    for rel, body in (files or {}).items():
        if not str(rel).lower().endswith((".cpp", ".cc", ".cxx", ".h", ".hpp")):
            continue
        for m in banned.finditer(str(body or "")):
            findings.append({"severity": "error", "code": "native_compute",
                             "message": "%s: [native_compute] forbidden native compute `%s`"
                                        % (rel, m.group(0))})
    return findings


def detect_side_stream(event_ms: float, wall_ms: float,
                       min_wall_ms: float = 0.5, ratio: float = 0.2) -> bool:
    """A side stream escapes a current-stream event: wall >> event."""
    try:
        event_ms, wall_ms = float(event_ms), float(wall_ms)
    except (TypeError, ValueError):
        return False
    return wall_ms >= min_wall_ms and event_ms > 0 and event_ms < ratio * wall_ms


def detect_cache(first_ms: float, fresh_ms: float,
                 max_ratio: float = 0.25, min_gain: float = 1.5) -> bool:
    """Benchmark output cache: the fixed-input pass is far faster than fresh inputs."""
    try:
        first_ms, fresh_ms = float(first_ms), float(fresh_ms)
    except (TypeError, ValueError):
        return False
    if first_ms <= 0 or fresh_ms <= 0:
        return False
    return first_ms < max_ratio * fresh_ms and (fresh_ms / first_ms) >= min_gain