"""Device noise amplitude (epsilon) and the statistical success gate.

`configs/noise.json` records, per device class, the 95th percentile of
|speedup - 1| measured with an honest solution whose work equals the reference.
A run is only evidence of an improvement when the lower bound of its speedup CI
clears `1 + epsilon`; a regression needs the upper bound below `1 - epsilon`.
Without a calibration for the device, the gate stays closed (neutral): no
statistics, no success claim.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "configs", "noise.json")


def load_noise(path: str = None) -> dict:
    p = path or DEFAULT_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"schema_version": 1, "devices": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "devices": {}}
    if not isinstance(data.get("devices"), dict):
        data["devices"] = {}
    return data


def device_key(device: str = None, hardware: str = None) -> str:
    """Map an actual device string (preferred) or a hardware label to a noise key."""
    d = str(device or "").strip().lower()
    if d and d != "auto":
        if "npu" in d or "ascend" in d:
            return "ascend-910b"
        if "cuda" in d or "nvidia" in d or "gpu" in d:
            return "cuda"
        if "cpu" in d:
            return "cpu"
    h = str(hardware or "").strip().lower()
    if h:
        if "ascend" in h or "npu" in h:
            return "ascend-910b"
        if "nvidia" in h or "cuda" in h or "gpu" in h:
            return "cuda"
        if "cpu" in h:
            return "cpu"
    return ""


def epsilon_for(device: str = None, hardware: str = None, path: str = None):
    """The calibrated epsilon for the measurement device, or None if uncalibrated."""
    entry = (load_noise(path).get("devices") or {}).get(device_key(device, hardware))
    if not isinstance(entry, dict):
        return None
    try:
        return float(entry.get("epsilon"))
    except (TypeError, ValueError):
        return None


def classify(speedup_ci, epsilon) -> str:
    """success / regression / neutral for a passing run with a speedup CI."""
    if epsilon is None or not speedup_ci or len(speedup_ci) < 2:
        return "neutral"
    try:
        lo, hi = float(speedup_ci[0]), float(speedup_ci[1])
    except (TypeError, ValueError):
        return "neutral"
    if lo > 1.0 + float(epsilon):
        return "success"
    if hi < 1.0 - float(epsilon):
        return "regression"
    return "neutral"


def promotion_floor(epsilon):
    """New strategies are only created when ci_lo clears 1 + 2*epsilon."""
    return None if epsilon is None else 1.0 + 2.0 * float(epsilon)
