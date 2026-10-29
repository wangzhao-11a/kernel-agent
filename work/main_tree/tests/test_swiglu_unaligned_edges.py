#!/usr/bin/env python3
"""Edge cases for the padded, multi-row SwiGlu path (non-32B-aligned rows).

Covers small row paddings (half=17/33), 1023-element rows, outer < R,
outer == R, and the last tile with exactly one row (the x1 overread guard).
"""
import torch
import torch_npu

import cann_bench

CASES = [
    ((1, 1022), torch.float16, -1),    # half=511, outer=1, last rh=1
    ((15, 1022), torch.float16, -1),   # outer < R
    ((16, 1022), torch.float16, -1),   # outer == R
    ((17, 1022), torch.float16, -1),   # last rh=1 after a full tile
    ((100, 34), torch.float16, 1),     # half=17, gap=4B
    ((37, 66), torch.bfloat16, 1),     # half=33
    ((33, 34), torch.float32, 1),      # half=17 fp32
    ((9, 2046), torch.bfloat16, -1),   # half=1023
    ((7, 2046), torch.float16, -1),    # half=1023 fp16
    ((2051, 34), torch.float16, 1),    # many tiles, last rh=3
    ((3, 7, 13, 1018), torch.float32, -1),  # half=509 fp32
    ((100, 1022), torch.float32, 1),   # half=511 fp32
]

THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}


def ref(x, dim):
    xf = x.to(torch.float32)
    x0, x1 = xf.chunk(2, dim=dim)
    return (torch.nn.functional.silu(x0) * x1).to(x.dtype)


def main():
    ok_all = True
    for shape, dtype, dim in CASES:
        x = (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dtype).npu()
        gold = ref(x, dim)
        got = cann_bench.swi_glu(x, dim)
        ok = tuple(got.shape) == tuple(gold.shape) and got.dtype == dtype
        mere = mare = 0.0
        if ok:
            a = got.cpu().to(torch.float32).reshape(-1)
            g = gold.cpu().to(torch.float32).reshape(-1)
            rel = (a - g).abs() / (g.abs() + 1e-7)
            mere = float(rel.mean())
            mare = float(rel.max())
            thr = THR[dtype]
            ok = mere < thr and mare < 10 * thr
        ok_all = ok_all and ok
        print("%-20s %-9s dim=%-3d MERE=%.3e MARE=%.3e %s" % (
            str(shape), str(dtype).split(".")[-1], dim, mere, mare, "PASS" if ok else "FAIL"))
    print("SWIGLU_UNALIGNED_EDGES", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())