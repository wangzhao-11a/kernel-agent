"""Small, dependency-free statistics helpers for evidence gating.

The harness records raw per-repetition timings.  A speedup is a ratio of two
medians; bootstrap resampling gives an interval without assuming normality.
"""
from __future__ import annotations

import random
import statistics


def _median(values):
    return float(statistics.median(values))


def bootstrap_ratio_ci(numerator, denominator, n: int = 1000, alpha: float = 0.05,
                       seed: int = 12345):
    """95% bootstrap CI for median(numerator) / median(denominator).

    For a speedup, pass the candidate's raw times as `denominator` and the
    baseline/control raw times as `numerator`.
    """
    num = [float(x) for x in (numerator or []) if x is not None]
    den = [float(x) for x in (denominator or []) if x is not None]
    if len(num) < 2 or len(den) < 2:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(max(50, int(n))):
        a = _median([num[rng.randrange(len(num))] for _ in range(len(num))])
        b = _median([den[rng.randrange(len(den))] for _ in range(len(den))])
        if b > 0:
            ratios.append(a / b)
    if not ratios:
        return None
    ratios.sort()
    lo = ratios[int((alpha / 2.0) * (len(ratios) - 1))]
    hi = ratios[int((1.0 - alpha / 2.0) * (len(ratios) - 1))]
    return [round(lo, 4), round(hi, 4)]


def ci_excludes_one(ci, threshold: float = 1.0) -> bool:
    return bool(ci and len(ci) == 2 and float(ci[0]) > float(threshold))