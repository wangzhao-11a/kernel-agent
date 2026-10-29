#!/usr/bin/env python3
"""Edge cases for the SwiGlu tiny path.

Covers half=1..15, packed uint32 Gather (even 16-bit half), arbitrary-step
vector offsets, partial tiles (rows not a multiple of rowsPerTile) and
rows fewer than the AIV core count.
"""
import torch
import torch_npu

import cann_bench

CASES = [
    ((5, 2), torch.float32, 1),        # half=1, rows < cores
    ((37, 4), torch.float16, 1),       # half=2, packed, partial core rows
    ((2051, 4), torch.float16, 1),     # half=2, partial last tile
    ((1023, 4), torch.bfloat16, -1),   # half=2, rows < rowsPerTile
    ((100, 6), torch.float16, 1),      # half=3, arbitrary step
    ((100, 10), torch.bfloat16, 1),    # half=5
    ((100, 12), torch.float16, 1),     # half=6, packed step=3
    ((100, 14), torch.bfloat16, 1),    # half=7
    ((100, 16), torch.float16, 1),     # half=8, packed step=4
    ((100, 18), torch.bfloat16, 1),    # half=9
    ((8193, 4), torch.float32, 1),     # half=2, unpacked (fp32)
    ((4099, 8), torch.bfloat16, 1),    # half=4, packed
    ((3, 6, 4), torch.float16, 1),     # outer=3, inner=4 -> half=12
    ((7, 6, 5), torch.bfloat16, 1),    # half=15, arbitrary step
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
        print("%-18s %-9s dim=%-3d MERE=%.3e MARE=%.3e %s" % (
            str(shape), str(dtype).split(".")[-1], dim, mere, mare, "PASS" if ok else "FAIL"))
    print("SWIGLU_TINY_EDGES", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())