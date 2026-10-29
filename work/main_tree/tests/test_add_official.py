#!/usr/bin/env python3
"""Official-style add cases for the CannBench harness.

Prints one case line per shape/dtype and a final ADD_TEST PASS/FAIL so
`harness.cannbench.parse_cases` can consume it.
"""
import torch
import torch_npu

import cann_bench

CASES = [
    ((1,), torch.float32),
    ((1024,), torch.float32),
    ((10000,), torch.float16),
    ((10, 10), torch.float32),
    ((256, 512), torch.float16),
    ((1000, 1000), torch.float32),
    ((5, 10, 15), torch.float16),
    ((16, 32, 64), torch.float32),
    ((4, 3, 64, 64), torch.float16),
    ((1000003,), torch.float32),
    ((1024,), torch.int32),
    ((256, 512), torch.int32),
]

THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13}


def rel_errors(got, gold):
    a = got.cpu().to(torch.float32).reshape(-1)
    g = gold.cpu().to(torch.float32).reshape(-1)
    rel = (a - g).abs() / (g.abs() + 1e-7)
    return float(rel.mean()), float(rel.max())


def main():
    ok_all = True
    for shape, dtype in CASES:
        if dtype == torch.int32:
            x = torch.randint(-1000, 1000, shape, dtype=dtype)
            y = torch.randint(-1000, 1000, shape, dtype=dtype)
        else:
            x = torch.randn(*shape, dtype=dtype)
            y = torch.randn(*shape, dtype=dtype)
        gold = x + y
        got = cann_bench.add(x.npu(), y.npu()).cpu()
        if dtype == torch.int32:
            ok = torch.equal(got, gold)
            mere = mare = 0.0
        else:
            ok = tuple(got.shape) == tuple(gold.shape) and got.dtype == dtype
            mere = mare = 0.0
            if ok:
                mere, mare = rel_errors(got, gold)
                thr = THR[dtype]
                ok = mere < thr and mare < 10 * thr
        ok_all = ok_all and ok
        print("%-22s %-9s dim=0   MERE=%.3e MARE=%.3e %s" % (
            shape, str(dtype).split(".")[-1], mere, mare, "PASS" if ok else "FAIL"))
    print("ADD_TEST", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())