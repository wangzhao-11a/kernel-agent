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


def bootstrap_weighted_ratio(cases, n: int = 1000, alpha: float = 0.05, seed: int = 12345):
    """Bootstrap CI for a weighted mean of per-case median ratios.

    `cases` is a list of {"weight": w, "numerator": baseline samples,
    "denominator": candidate samples}.  Resampling is within each case, then the
    per-case medians are combined with normalized weights.
    """
    prepared = []
    for case in cases or []:
        num = [float(x) for x in (case.get("numerator") or []) if x is not None]
        den = [float(x) for x in (case.get("denominator") or []) if x is not None]
        weight = float(case.get("weight", 1.0) or 0.0)
        if len(num) < 2 or len(den) < 2 or weight <= 0:
            continue
        prepared.append((weight, num, den))
    if not prepared:
        return None
    total = sum(w for w, _n, _d in prepared)
    if total <= 0:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(max(50, int(n))):
        acc = 0.0
        for weight, num, den in prepared:
            a = _median([num[rng.randrange(len(num))] for _ in range(len(num))])
            b = _median([den[rng.randrange(len(den))] for _ in range(len(den))])
            acc += (weight / total) * (a / b if b > 0 else 0.0)
        ratios.append(acc)
    ratios.sort()
    lo = ratios[int((alpha / 2.0) * (len(ratios) - 1))]
    hi = ratios[int((1.0 - alpha / 2.0) * (len(ratios) - 1))]
    return [round(lo, 4), round(hi, 4)]


def ci_excludes_one(ci, threshold: float = 1.0) -> bool:
    return bool(ci and len(ci) == 2 and float(ci[0]) > float(threshold))