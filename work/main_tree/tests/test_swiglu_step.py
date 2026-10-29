import torch, torch_npu
import cann_bench

print("=== step 1: kernel alone, tiny case ===")
x = torch.randn(1, 128, dtype=torch.float16).npu()
y = cann_bench.swi_glu(x, -1)
print("launched; out shape", tuple(y.shape))
try:
    s = float(y.cpu().to(torch.float32).sum())
    print("STEP1_OK sum=%.4f" % s)
except Exception as exc:
    print("STEP1_FAIL", type(exc).__name__, str(exc)[:200])
    raise SystemExit(1)

print("=== step 2: kernel vs CPU reference (official-ish cases) ===")
CASES = [
    ((1024, 2048), torch.float16, -1),
    ((2048, 4096), torch.float32, -1),
    ((4096, 8192), torch.bfloat16, -1),
    ((1022, 2047), torch.bfloat16, 0),
    ((2049, 1024), torch.float16, 1),
    ((11, 13, 16, 67), torch.float32, 2),
    ((2, 3, 17, 512, 100), torch.float32, -1),
    ((1000003, 2), torch.bfloat16, 1),
]
THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}
ok_all = True
for shape, dt, dim in CASES:
    x = (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dt)
    xf = x.to(torch.float32)
    x0, x1 = xf.chunk(2, dim=dim)
    gold = (torch.nn.functional.silu(x0) * x1).to(dt).reshape(-1)
    xn = x.npu()
    yn = cann_bench.swi_glu(xn, dim).cpu().to(torch.float32).reshape(-1)
    g = gold.to(torch.float32)
    rel = (yn - g).abs() / (g.abs() + 1e-7)
    mere = float(rel.mean()); mare = float(rel.max())
    thr = THR[dt]
    ok = (mere < thr) and (mare < 10 * thr)
    ok_all = ok_all and ok
    print("%-24s %-9s dim=%-3d MERE=%.3e MARE=%.3e %s" % (shape, str(dt).split('.')[-1], dim, mere, mare, "PASS" if ok else "FAIL"))
print("SWIGLU_KERNEL", "PASS" if ok_all else "FAIL")
