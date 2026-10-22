"""Level 2 — W8A8 quantized matmul (int8 in, fp32 dequantized oracle, vendor-op baseline).

The correctness oracle is exact float math on the dequantized values (per-tensor
activation scale, per-channel weight scales). The *baseline* is the vendor's fused
quantized matmul (`baseline="op"`), which the harness correctness-checks before using —
so a plain dequant-then-matmul torch path will not win.

Optimization targets: keep the int8 data path (int8 loads, int32 accumulation, scale on
the epilogue), avoid materializing the dequantized fp32 copies, fuse the per-channel scale.
"""
import torch

M, K_DIM, N = 1024, 512, 512

TASK = {
    "name": "quantized_matmul_w8a8",
    "level": 2,
    "description": (
        "W8A8 quantized matmul: a int8 (1024,512), w int8 (512,512), per-tensor activation "
        "scale_a (1,), per-channel weight scale_w (512,); out fp32 (1024,512) = "
        "(a * scale_a) @ (w * scale_w). The baseline is the vendor fused op."
    ),
    "tolerance": {"atol": 1e-2, "rtol": 1e-2},
    "num_trials": 2,
    "warmup": 5,
    "rep": 20,
    "entrypoint": "run",
    "baseline": "op",
    "expected_speedup": 1.5,
    "tags": ["gemm", "quantized", "w8a8", "int8", "compute-bound"],
    "op_class": "gemm",
    "dtypes": ["int8"],
    "hardware": "ascend-910b",
}


def make_inputs(device, trial):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(8080 + trial)
    a = torch.randint(-127, 128, (M, K_DIM), generator=gen, dtype=torch.int8)
    w = torch.randint(-127, 128, (K_DIM, N), generator=gen, dtype=torch.int8)
    scale_a = torch.rand(1, generator=gen, dtype=torch.float32) * 0.01 + 0.001
    scale_w = torch.rand(N, generator=gen, dtype=torch.float32) * 0.01 + 0.001
    return (a.to(device), w.to(device), scale_a.to(device), scale_w.to(device))


def reference(a, w, scale_a, scale_w):
    return (a.to(torch.float32) * scale_a) @ (w.to(torch.float32) * scale_w)


def baseline(a, w, scale_a, scale_w):
    """Vendor fused quantized matmul, self-validated on a 4-row probe before use.

    Different torch_npu builds expose different names/argument orders; a variant is only
    accepted if it reproduces the dequantized reference on the probe. If none does, this
    raises and the harness falls back to eager with a recorded note (never a fake baseline).
    """
    import torch_npu  # noqa: F401  (raises on machines without it -> eager fallback)

    variants = (
        lambda fn, xa: fn(xa, w, scale_a, scale_w),
        lambda fn, xa: fn(xa, w, scale_a, scale_w, None),
        lambda fn, xa: fn(xa, w, scale_a, scale_w,
                          torch.zeros(xa.shape[0], N, dtype=torch.float32, device=xa.device)),
        lambda fn, xa: fn(xa, w, scale_w, scale_a),
        lambda fn, xa: fn(xa, w.t().contiguous(), scale_a, scale_w),
    )
    names = ("npu_quant_matmul", "npu_quant_matmul_v4", "npu_weight_quant_batchmatmul",
             "npu_add_quant_matmul", "npu_add_quant_matmul_")
    probe = a[:4]
    probe_ref = (probe.to(torch.float32) * scale_a) @ (w.to(torch.float32) * scale_w)
    errors = []
    for name in names:
        fn = getattr(torch_npu, name, None)
        if fn is None:
            continue
        for variant in variants:
            try:
                out = variant(fn, probe)
                err = float((out.to(torch.float32) - probe_ref).abs().max())
            except Exception as exc:
                errors.append("%s: %s" % (name, str(exc)[:90]))
                continue
            if tuple(out.shape) == (4, N) and err < 1e-1:
                return variant(fn, a)  # the same proven variant, on the full tensors
            errors.append("%s: probe mismatch (max_err=%.3e)" % (name, err))
    raise RuntimeError("no validated quantized matmul op (%s)" % ("; ".join(errors[:3]) or "no candidate API"))