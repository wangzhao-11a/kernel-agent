"""Template: Triton implementation of vector_add (needs triton + a supported GPU).

Kept out of solutions/examples because it cannot run on CPU-only machines; copy it,
adapt the kernel to your task, and evaluate with:

    python -m harness.runner --task tasks/level1/vector_add.py --solution my_triton.py --device cuda
"""
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


def run(x, y):
    if not HAVE_TRITON or not x.is_cuda:
        return x + y
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    _add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out