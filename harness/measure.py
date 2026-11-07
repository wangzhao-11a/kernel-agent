"""ABAB interleaved measurement with per-round medians and a bootstrap CI.

Each round measures the baseline and the candidate back to back; the order flips
every round (A-B, B-A, ...) so slow drift inside a round cannot systematically
favour either side.  The per-round speedups are bootstrapped to a 95% CI, which
is what the success gate (1 + epsilon) consumes.
"""
from __future__ import annotations

import statistics

from . import benchmark as bench
from . import device as dev
from . import stats as stats_mod

DEFAULT_ROUNDS = 5
ROUND_WARMUP = 3


def reps_per_round(rep: int) -> int:
    """Half the task rep (at least 5): two sides x N rounds stays affordable."""
    return max(5, int(rep) // 2)


def ab_benchmark(solution_fn, baseline_fn, input_sets, device: str,
                 rounds: int = DEFAULT_ROUNDS, warmup: int = 10, rep: int = 50,
                 flush_l2: bool = True, perturb: bool = True) -> dict:
    sets = [list(s) for s in input_sets or ()]
    if not sets:
        raise ValueError("ab_benchmark needs input_sets")
    rounds = max(1, int(rounds))
    per_round_rep = reps_per_round(rep)

    # Shared warmup outside every timed round: both sides pay it once.
    for i in range(max(0, int(warmup))):
        args = sets[i % len(sets)]
        baseline_fn(*args)
        solution_fn(*args)
        if perturb:
            bench.perturb_inputs(args, device, step=i + 1)
    dev.sync(device)

    per_round = []
    clock_anomalies = 0
    for r in range(rounds):
        order = ("baseline", "solution") if r % 2 == 0 else ("solution", "baseline")
        measured = {}
        for name in order:
            fn = baseline_fn if name == "baseline" else solution_fn
            measured[name] = bench.benchmark(
                fn, input_sets=sets, device=device, warmup=ROUND_WARMUP,
                rep=per_round_rep, flush_l2=flush_l2, perturb=perturb)
            clock_anomalies += int(measured[name].get("clock_anomalies") or 0)
        base_ms = measured["baseline"]["median_ms"]
        sol_ms = measured["solution"]["median_ms"]
        speedup = base_ms / sol_ms if sol_ms > 0 else 0.0
        per_round.append({
            "round": r + 1,
            "order": list(order),
            "baseline_ms": round(base_ms, 6),
            "solution_ms": round(sol_ms, 6),
            "speedup": round(speedup, 4),
        })

    speedups = [row["speedup"] for row in per_round]
    sol_medians = [row["solution_ms"] for row in per_round]
    sol_median = statistics.median(sol_medians)
    # Relative dispersion of the candidate's own timings, independent of the
    # effect size.  Compared against epsilon in the worker.
    stability = (statistics.pstdev(sol_medians) / sol_median
                 if len(sol_medians) > 1 and sol_median > 0 else 0.0)
    return {
        "clock_anomalies": clock_anomalies,
        "speedup": round(float(statistics.median(speedups)), 4),
        "speedup_ci": [round(float(x), 4) for x in stats_mod.bootstrap_ci(speedups)],
        "n_rounds": rounds,
        "rounds": per_round,
        "stability_delta": round(float(stability), 4),
        "solution": {"median_ms": round(float(sol_median), 6)},
        "baseline": {"median_ms": round(float(statistics.median(
            [row["baseline_ms"] for row in per_round])), 6)},
        "rep_per_round": per_round_rep,
        "unit": "ms",
    }
