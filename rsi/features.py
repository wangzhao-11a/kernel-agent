"""Task / hardware feature extraction used for condition matching in the RSI layer."""
from __future__ import annotations

HARDWARE_ALIASES = (
    ("ascend", "ascend-910b"),
    ("npu", "ascend-910b"),
    ("cuda", "nvidia-gpu"),
    ("cpu", "cpu"),
)

_OP_CLASS_RULES = (
    ("attention", ("attention", "flash", "mla", "sdpa")),
    ("moe", ("moe", "expert", "routing", "finalize")),
    ("gemm", ("matmul", "gemm", "grouped")),
    ("reduction", ("softmax", "rmsnorm", "layernorm", "norm", "reduce")),
    ("elementwise", ("vector_add", "add", "elementwise", "activation")),
)

_BOUND_RULES = (
    ("compute-bound", ("attention", "gemm", "matmul", "mla", "moe")),
    ("memory-bound", ("softmax", "rmsnorm", "layernorm", "vector", "elementwise", "gather")),
)


def hardware_from_device(device: str, override: str = "") -> str:
    if override:
        return override
    d = str(device).lower()
    for prefix, label in HARDWARE_ALIASES:
        if d.startswith(prefix):
            return label
    return "unknown"


def _meta_of(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    meta = getattr(obj, "meta", None)
    return meta if isinstance(meta, dict) else {}


def _infer_tags(name: str, description: str, meta: dict) -> list:
    text = ("%s %s" % (name, description)).lower()
    tags = []
    for op_class, needles in _OP_CLASS_RULES:
        if any(n in text for n in needles):
            tags.append(op_class)
    for bound, needles in _BOUND_RULES:
        if any(n in text for n in needles):
            tags.append(bound)
    for spec in meta.get("input_specs") or []:
        dt = str(spec.get("dtype", "")).lower()
        if dt and dt not in tags:
            tags.append(dt)
    if meta.get("level"):
        tags.append("level%d" % int(meta["level"]))
    if meta.get("hardware"):
        tags.append(str(meta["hardware"]))
    return tags


def _guess_op_class(meta: dict, tags: list) -> str:
    explicit = meta.get("op_class")
    if explicit:
        return str(explicit)
    for op_class, _ in _OP_CLASS_RULES:
        if op_class in tags:
            return op_class
    return ""


def task_features(task_like) -> dict:
    """Extract matching features from a task module/Task/dict(TASK meta)."""
    meta = _meta_of(task_like)
    name = str(meta.get("name") or getattr(task_like, "name", "") or "")
    description = str(meta.get("description") or "")
    dtypes = [str(d).lower() for d in (meta.get("dtypes") or [])]
    for spec in meta.get("input_specs") or []:
        dt = str(spec.get("dtype", "float32")).lower()
        if dt not in dtypes:
            dtypes.append(dt)
    declared = [str(t) for t in (meta.get("tags") or [])]
    tags = sorted(set(declared + _infer_tags(name, description, meta)))
    return {
        "name": name,
        "tags": tags,
        "op_class": _guess_op_class(meta, tags),
        "dtypes": dtypes or ["float32"],
        "level": meta.get("level"),
        "hardware": str(meta.get("hardware") or ""),
    }