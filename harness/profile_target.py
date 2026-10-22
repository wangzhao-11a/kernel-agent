"""Run a solution a few times without measuring: the target process for external profilers.

    # NVIDIA: profile with Nsight Compute
    ncu --csv --target-processes all -f -o /tmp/prof \\
        python -m harness.profile_target --task tasks/level1/softmax.py --solution sol.py --device cuda

    # Ascend: profile with msprof
    msprof --output=/tmp/prof --type=text --summary-format=csv --export=on \\
        --application="python -m harness.profile_target --task tasks/level1/softmax.py --solution sol.py --device npu"

Then parse:
    python -m harness.profilers.ncu --file /tmp/prof.csv
    python -m harness.profilers.msprof --dir /tmp/prof
"""
from __future__ import annotations

import argparse

from . import device as dev
from .runner import ROOT  # noqa: F401  (keeps the package importable from project root)
from .tasks import load_task
from .worker import _wrap_inference, import_solution


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.profile_target", description="Run a solution for external profilers.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args(argv)

    device = dev.resolve_device(args.device)
    task = load_task(args.task)
    solution = import_solution(args.solution)
    entry = getattr(solution, task.entrypoint)
    run = _wrap_inference(entry)

    inputs = task.make_inputs(device, 0)
    for _ in range(max(0, args.warmup)):
        run(*inputs)
    dev.sync(device)
    for i in range(max(1, args.iters)):
        run(*inputs)
        print("iteration %d done" % (i + 1))
    dev.sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())