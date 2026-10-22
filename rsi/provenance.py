"""Evidence provenance for performance claims.

A run can be a mechanics test (CPU, replay provider) or a performance measurement
(real NPU/GPU).  Only the latter may update strategy-pool evidence.  This module is
the single place that decides which is which, so `rsi.distill` and `rsi.evolve`
cannot disagree.
"""
from __future__ import annotations


def device_class(device: str) -> str:
    d = str(device or "").lower()
    if "npu" in d or "ascend" in d:
        return "npu"
    if "cuda" in d or "nvidia" in d or "gpu" in d:
        return "cuda"
    if "cpu" in d:
        return "cpu"
    return "unknown"


def provenance(meta: dict) -> dict:
    """Can this run's performance numbers be used as evidence for its claimed hardware?"""
    device = str((meta or {}).get("device") or "")
    claimed = str((meta or {}).get("hardware") or "").strip().lower()
    kind = device_class(device)
    inferred = {"npu": "ascend-910b", "cuda": "nvidia-gpu", "cpu": "cpu"}.get(kind, "")

    if kind == "unknown":
        return {"device_class": kind, "claimed_hardware": claimed,
                "inferred_hardware": inferred, "valid": False,
                "reason": "device is unknown; cannot attribute performance evidence"}

    if not claimed:
        return {"device_class": kind, "claimed_hardware": "",
                "inferred_hardware": inferred, "valid": kind != "cpu",
                "reason": "no hardware label; inferred %s" % inferred}

    if kind == "cpu":
        valid = claimed in ("cpu", "unknown")
        return {"device_class": kind, "claimed_hardware": claimed,
                "inferred_hardware": inferred, "valid": valid,
                "reason": ("CPU run labelled %s: mechanics-only, not %s performance evidence"
                           % (claimed, claimed)) if not valid else "CPU mechanics run"}

    if kind == "npu":
        valid = ("ascend" in claimed) or ("npu" in claimed)
        return {"device_class": kind, "claimed_hardware": claimed,
                "inferred_hardware": inferred, "valid": valid,
                "reason": "" if valid else "NPU run labelled %s: mismatched hardware" % claimed}

    if kind == "cuda":
        valid = ("nvidia" in claimed) or ("cuda" in claimed) or ("gpu" in claimed)
        return {"device_class": kind, "claimed_hardware": claimed,
                "inferred_hardware": inferred, "valid": valid,
                "reason": "" if valid else "CUDA run labelled %s: mismatched hardware" % claimed}

    return {"device_class": kind, "claimed_hardware": claimed,
            "inferred_hardware": inferred, "valid": False, "reason": "unrecognised device"}