#!/usr/bin/env python3
"""Official-style RMSNorm cases for the CannBench harness.

out[r,c] = x[r,c] * rsqrt(mean_c x[r,c]^2 + eps) * gamma[c]
"""
import torch
import torch_npu

import cann_bench

CASES = [
    ((2048, 4096), torch.float32, 1e-6),
    ((1024, 1024), torch.float16, 1e-6),
    ((512, 8192), torch.bfloat16, 1e-6),
    ((17, 257), torch.float32, 1e-5),
    ((3, 5, 127), torch.float16, 1e-6),
    ((4096, 1024), torch.bfloat16, 1e-6),
]

THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}


def ref(x, g, eps):
    xf = x.to(torch.float32)
    gf = g.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms * gf).to(x.dtype)


def main():
    ok_all = True
    for shape, dtype, eps in CASES:
        x = torch.randn(*shape, dtype=dtype)
        g = torch.randn(shape[-1], dtype=dtype)
        gold = ref(x, g, eps)
        got = cann_bench.rms_norm(x.npu(), g.npu(), eps).cpu()
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
        print("%-22s %-9s dim=-1  MERE=%.3e MARE=%.3e %s" % (
            shape, str(dtype).split(".")[-1], mere, mare, "PASS" if ok else "FAIL"))
    print("RMSNORM_TEST", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())