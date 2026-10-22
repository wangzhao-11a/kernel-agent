"""Device backends (cpu / cuda / npu) with a lazy torch import.

torch is imported lazily so the pipeline (and its self-tests) can run on machines
without torch installed; accelerator backends are resolved from the device string.
"""
from __future__ import annotations

_TORCH = None
_TORCH_CHECKED = False


def torch_module():
    """Import torch on first use; return None when it is not installed."""
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        _TORCH_CHECKED = True
        try:
            import torch as _t
        except Exception:
            _t = None
        _TORCH = _t
    return _TORCH


def backend_of(device: str) -> str:
    d = str(device).lower()
    if d.startswith("npu") or d.startswith("ascend"):
        return "npu"
    if d.startswith("cuda"):
        return "cuda"
    return "cpu"


def backend_module(device: str):
    """Return torch.cuda / torch.npu for accelerator devices, else None."""
    t = torch_module()
    if t is None:
        return None
    kind = backend_of(device)
    if kind == "cpu":
        return None
    return getattr(t, kind, None)


def _ensure_backend(device: str) -> None:
    if backend_of(device) == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "device %r was requested but torch_npu is not importable: %s" % (device, exc)
            ) from exc


def _npu_available(t) -> bool:
    mod = getattr(t, "npu", None)
    if mod is None:
        try:
            import torch_npu  # noqa: F401
        except Exception:
            return False
        mod = getattr(t, "npu", None)
    if mod is None:
        return False
    try:
        return bool(mod.is_available())
    except Exception:
        return False


def resolve_device(name: str = "auto") -> str:
    if name and name not in ("auto", ""):
        _ensure_backend(name)
        return name
    t = torch_module()
    if t is not None:
        cuda = getattr(t, "cuda", None)
        if cuda is not None and cuda.is_available():
            return "cuda"
        if _npu_available(t):
            return "npu"
    return "cpu"


def is_accelerator(device: str) -> bool:
    return backend_of(device) in ("cuda", "npu")


def sync(device: str) -> None:
    mod = backend_module(device)
    if mod is not None:
        try:
            mod.synchronize()
        except Exception:
            pass


def device_info(device: str) -> dict:
    kind = backend_of(device)
    info = {"device": device, "backend": kind}
    mod = backend_module(device)
    if mod is None:
        return info
    try:
        info["count"] = mod.device_count()
        if kind == "cuda":
            t = torch_module()
            idx = t.device(device).index or 0
            props = mod.get_device_properties(idx)
            info["name"] = props.name
            info["memory_gb"] = round(props.total_memory / (1024 ** 3), 1)
            if hasattr(props, "L2_cache_size"):
                info["l2_cache_mb"] = round(props.L2_cache_size / (1024 ** 2), 1)
        else:  # npu
            info["name"] = mod.get_device_name(0)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def flush_buffer(device: str, cap_mb: int = 256):
    """Allocate (once) a scratch buffer used to evict L2 between timed iterations."""
    t = torch_module()
    if t is None or not is_accelerator(device):
        return None
    try:
        n_bytes = cap_mb * 1024 * 1024
        if backend_of(device) == "cuda":
            idx = t.device(device).index or 0
            props = t.cuda.get_device_properties(idx)
            l2 = int(getattr(props, "L2_cache_size", 0) or 0)
            if l2:
                n_bytes = int(min(max(l2, 64 * 1024 * 1024), cap_mb * 1024 * 1024))
        return t.empty(n_bytes, dtype=t.int8, device=device)
    except Exception:
        return None