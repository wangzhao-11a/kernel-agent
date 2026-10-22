"""Minimal triton-ascend Sigmoid: 1D flat, masked, BLOCK=4096 (route probe for the 950pr pool).

Deliberately plain: tl.load / tl.exp / tl.store only — no 2D aranges, no tl.split/reshape, no
constexpr-specialised division — so a staged-eval failure on a new pool can only come from the
triton-ascend route itself, not from a fancy code path.
"""
import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=mask)


def run(x):
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    total = out.numel()
    if total:
        _sigmoid_kernel[(triton.cdiv(total, BLOCK),)](x, out, total, BLOCK=BLOCK)
    return out
