#!/usr/bin/env python3
"""Edge cases for the batched rms_norm3 path (and its v2 fallback).

These shapes are deliberately not in the official list: non-64-multiple aligned
columns, rows fewer than one tile, rows fewer than the core count, and
non-32B-aligned rows that must fall back to v2.
"""
import torch
import torch_npu

import cann_bench

CASES = [
    ((1, 8), torch.float32),
    ((37, 8), torch.float32),
    ((40, 8), torch.float32),
    ((41, 8), torch.float32),
    ((1000, 24), torch.float32),
    ((129, 40), torch.float32),
    ((257, 48), torch.float16),
    ((63, 80), torch.bfloat16),
    ((500, 120), torch.float32),
    ((64, 256), torch.float16),
    ((65, 256), torch.bfloat16),
    ((4097, 72), torch.float16),   # 144B rows -> v2 fallback
    ((33, 200), torch.bfloat16),   # 400B rows -> v2 fallback
]

THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}


def ref(x, g, eps):
    xf = x.to(torch.float32)
    gf = g.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms * gf).to(x.dtype)


def main():
    ok_all = True
    for shape, dtype in CASES:
        x = torch.randn(*shape, dtype=dtype)
        g = torch.randn(shape[-1], dtype=dtype)
        gold = ref(x, g, 1e-6)
        got = cann_bench.rms_norm(x.npu(), g.npu(), 1e-6).cpu()
        ok = tuple(got.shape) == tuple(gold.shape) and got.dtype == dtype
        mere = mare = 0.0
        if ok:
            a = got.to(torch.float32).reshape(-1)
            b = gold.to(torch.float32).reshape(-1)
            rel = (a - b).abs() / (b.abs() + 1e-7)
            mere = float(rel.mean())
            mare = float(rel.max())
            thr = THR[dtype]
            ok = mere < thr and mare < 10 * thr
        ok_all = ok_all and ok
        print("%-18s %-9s MERE=%.3e MARE=%.3e %s" % (
            str(shape), str(dtype).split(".")[-1], mere, mare, "PASS" if ok else "FAIL"))
    print("RMSNORM_EDGES", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())