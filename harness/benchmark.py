"""Timing: wall clock with global device syncs, rotating input sets, no caching.

Anti-noise and anti-cheat measures:

- warmup, median over `rep` iterations, optional L2 flush between timed iterations;
- K input sets rotated round-robin, so one fixed tensor is not the whole benchmark;
- input values are perturbed in place after every timed call (outside the timed
  window), so a solution that memoized its output is caught by the worker's
  post-timing probe instead of being rewarded with a ~0 ms median;
- on accelerators the reported time is wall clock bracketed by *global* device
  synchronizations.  Current-stream events miss work launched on a side stream.
"""
from __future__ import annotations

import statistics
import time

from . import device as dev

# Bind the real clock at import: the report must never depend on the module
# attribute `time`, which a solution could rebind to a fake clock.
_PERF_COUNTER = time.perf_counter

# A timed region where the module-level clock disagrees with the bound clock by
# more than this factor is flagged; the worker turns the flag into a failure.
CLOCK_DISAGREEMENT = 0.2

DEFAULT_INPUT_SETS = 4
MAX_INPUT_BYTES = 512 * 1024 * 1024


def _is_floating(x) -> bool:
    try:
        if callable(getattr(x, "is_floating_point", None)):
            return bool(x.is_floating_point())  # torch tensor
        kind = getattr(getattr(x, "dtype", None), "kind", None)
        return kind == "f"  # numpy array
    except Exception:
        return False


def _flat_view(x):
    """A flat in-place view of a torch tensor / numpy array, or None."""
    try:
        if callable(getattr(x, "data_ptr", None)):
            try:
                flat = x.view(-1)
            except Exception:
                flat = x.reshape(-1)
            return flat if flat.data_ptr() == x.data_ptr() else None
        import numpy as np

        if isinstance(x, np.ndarray):
            flat = x.reshape(-1)
            return flat if np.shares_memory(flat, x) else None
    except Exception:
        return None
    return None


def _numel(x) -> int:
    numel = getattr(x, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:
            return 0
    try:
        return int(getattr(x, "size", 0) or 0)
    except Exception:
        return 0


def perturb_inputs(inputs, device: str = "cpu", step: int = 0) -> bool:
    """Change a few input values in place so a cached output becomes stale.

    Runs outside the timed region and costs O(1) per floating-point tensor.
    Integer/boolean tensors (indices, masks) are left untouched.  Returns True
    when at least one element was changed.
    """
    changed = False
    for x in inputs or ():
        if not _is_floating(x):
            continue
        flat = _flat_view(x)
        n = _numel(flat) if flat is not None else 0
        if n <= 0:
            continue
        for j in range(3):
            idx = (int(step) * 7919 + j * 104729) % n
            try:
                flat[idx] = flat[idx] + 0.5
                changed = True
            except Exception:
                pass
    return changed


def input_bytes(inputs) -> int:
    total = 0
    for x in inputs or ():
        nbytes = getattr(x, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
            continue
        try:
            total += int(x.numel()) * int(x.element_size())
        except Exception:
            pass
    return total


def benchmark(fn, inputs=None, device: str = "cpu", warmup: int = 25, rep: int = 100,
              flush_l2: bool = True, input_sets=None, perturb: bool = True) -> dict:
    raw = input_sets if input_sets is not None else [inputs]
    if not raw or raw[0] is None:
        raise ValueError("benchmark needs inputs or input_sets")
    sets = [list(s) for s in raw]
    accelerator = dev.is_accelerator(device)
    buf = dev.flush_buffer(device) if flush_l2 else None

    def _call(i: int):
        args = sets[i % len(sets)]
        if buf is not None:
            buf.zero_()
        return fn(*args)

    def _after(i: int) -> None:
        if perturb:
            perturb_inputs(sets[i % len(sets)], device, step=i + 1)

    for i in range(max(0, int(warmup))):
        _call(i)
        if accelerator:
            dev.sync(device)
        _after(i)
    dev.sync(device)

    times = []
    clock_anomalies = 0
    for i in range(int(rep)):
        if accelerator:
            # Global sync on both sides: the flush and any work from earlier
            # iterations are drained before t0, and every stream launched by the
            # candidate is awaited inside the timed window.
            #
            # The audited window must bracket exactly the same span as the real
            # one: ending it before the trailing sync would compare launch
            # overhead against real device time and flag every iteration.
            if buf is not None:
                buf.zero_()
            dev.sync(device)
            t0 = _PERF_COUNTER()
            a0 = time.perf_counter()
            fn(*sets[i % len(sets)])
            dev.sync(device)
            a1 = time.perf_counter()
            t1 = _PERF_COUNTER()
        else:
            if buf is not None:
                buf.zero_()
            t0 = _PERF_COUNTER()
            a0 = time.perf_counter()
            fn(*sets[i % len(sets)])
            a1 = time.perf_counter()
            t1 = _PERF_COUNTER()
        real = t1 - t0
        audited = a1 - a0
        if real > 1e-4 and abs(audited - real) > CLOCK_DISAGREEMENT * max(real, audited):
            clock_anomalies += 1
        times.append(real * 1000.0)
        _after(i)

    ordered = sorted(times)
    n = len(ordered)
    return {
        "clock_anomalies": clock_anomalies,
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.fmean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "p10_ms": float(ordered[max(0, int(0.10 * (n - 1)))]),
        "std_ms": float(statistics.pstdev(ordered)) if n > 1 else 0.0,
        "reps": n,
        "n_input_sets": len(sets),
        "perturbed": bool(perturb),
        "device_sync": bool(accelerator),
    }
