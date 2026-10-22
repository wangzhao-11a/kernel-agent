"""Torch-level reference solution for the SWiGLU task (CPU smoke test + semantic oracle).

The kernel form can require Triton or AscendC; this file exists so the task can be
exercised end-to-end on CPU and so the arithmetic path is pinned down.
"""
import torch


def run(input, dim=-1):
    out_dtype = input.dtype
    x = input.to(torch.float32)
    x0, x1 = x.chunk(2, dim=dim)
    return (torch.nn.functional.silu(x0) * x1).to(out_dtype)