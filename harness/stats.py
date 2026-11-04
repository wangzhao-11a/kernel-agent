"""Robust statistics for benchmark measurements.

Round-level speedups are bootstrapped (percentile method) to a 95% confidence
interval; a single run's speedup is never trusted on its own.
"""
from __future__ import annotations

import random
import statistics


def percentile(values, q: float):
    """Linear-interpolated percentile (numpy-style) without importing numpy."""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(100.0, float(q)))
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def bootstrap_ci(values, n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 20261104) -> list:
    """Percentile-bootstrap 95% CI of the median of `values`."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        v = vals[0] if vals else 0.0
        return [v, v]
    n_boot = max(100, int(n_boot))
    rng = random.Random(int(seed))
    n = len(vals)
    medians = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_index = int((alpha / 2.0) * n_boot)
    hi_index = int((1.0 - alpha / 2.0) * n_boot) - 1
    lo = medians[max(0, min(lo_index, n_boot - 1))]
    hi = medians[max(0, min(hi_index, n_boot - 1))]
    return [float(lo), float(hi)]
