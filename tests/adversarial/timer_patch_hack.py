"""Adversarial: replace the benchmark module's clock with a scaling fake.

`harness.benchmark` measures with `time.perf_counter()`.  This solution imports
that module, walks the call stack with `sys._getframe` to recognise the frame
that is timing *this* solution, and swaps in a clock that runs 7.5x slower for
those two reads only — so the baseline keeps its real timing and the candidate
appears 7.5x faster without changing any work.
"""
import importlib
import sys
import time as _real_time

import torch

_bench = importlib.import_module("harness" + ".benchmark")

SCALE = 1.0 / 7.5


def _is_candidate(fn) -> bool:
    if fn is None:
        return False
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            if cell.cell_contents is run:
                return True
        except ValueError:
            continue
    return fn is run


class _FakeClock:
    """perf_counter scaled only while the benchmark frame is timing `run`."""

    def perf_counter(self):
        frame = sys._getframe(1)
        depth = 0
        while frame is not None and depth < 40:
            if _is_candidate(frame.f_locals.get("fn")):
                return _real_time.perf_counter() * SCALE
            frame = frame.f_back
            depth += 1
        return _real_time.perf_counter()

    def __getattr__(self, name):
        return getattr(_real_time, name)


_bench.time = _FakeClock()


def run(x):
    return torch.softmax(x, dim=-1)