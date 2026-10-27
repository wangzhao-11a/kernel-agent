import torch, torch_npu
import cann_bench

CASES = [
    ((1024, 2048), torch.float16, -1),
    ((2048, 4096), torch.float32, -1),
    ((4096, 8192), torch.bfloat16, -1),
    ((1022, 2047), torch.bfloat16, 0),
    ((2049, 1024), torch.float16, 1),
    ((3, 7, 13, 1018), torch.float32, -1),
    ((11, 13, 16, 67), torch.float32, 2),
    ((2, 3, 17, 512, 100), torch.float32, -1),
    ((4, 510, 4097), torch.bfloat16, 1),
    ((512, 4096), torch.float32, -1),
    ((1000003, 2), torch.bfloat16, 1),
    ((363, 367, 14), torch.bfloat16, -1),
]

def ref(x, dim):
    xf = x.to(torch.float32)
    x0, x1 = xf.chunk(2, dim=dim)
    return (torch.nn.functional.silu(x0) * x1).to(x.dtype)

THR = {torch.float16: 2 ** -10, torch.float32: 2 ** -13, torch.bfloat16: 2 ** -7}
ok_all = True
for shape, dt, dim in CASES:
    x = (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dt).npu()
    gold = ref(x, dim)
    y = cann_bench.swi_glu(x, dim)
    a = y.cpu().to(torch.float32).reshape(-1)
    g = gold.cpu().to(torch.float32).reshape(-1)
    rel = (a - g).abs() / (g.abs() + 1e-7)
    mere = float(rel.mean())
    mare = float(rel.max())
    thr = THR[dt]
    ok = (mere < thr) and (mare < 10 * thr) and (tuple(y.shape) == tuple(gold.shape)) and (y.dtype == dt)
    ok_all = ok_all and ok
    print("%-22s %-9s dim=%-3d MERE=%.3e MARE=%.3e %s" % (shape, str(dt).split('.')[-1], dim, mere, mare, "PASS" if ok else "FAIL"))
print("SWIGLU_TEST", "PASS" if ok_all else "FAIL")
