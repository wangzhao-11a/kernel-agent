"""Implementation-form priors, measured on real hardware.

Qualitative result from running the same operator suite through multiple forms:

  attention / GEMM / reduction families -> Triton kernels won big
  data-movement / sort / permute family  -> Triton failed outright

Conclusion: "extreme performance" is a per-family form choice, not one form
everywhere. This module turns that observation into a deterministic hint in the
task prompt. Re-measure and update the priors for your own hardware.
"""
from __future__ import annotations

FORM_PRIORS = (
    {
        "op_classes": ("attention", "gemm", "reduction"),
        "form": "triton_kernel",
        "advice": ("Prefer a triton-ascend kernel as the main implementation. Measured on this exact "
                   "machine: 22.3x (sparse attention), 23.2x (grouped GEMM+SwiGLU), 3.3x (MoE finalize), "
                   "3.1x (norm+rope fusion), 2.7x (w8a8). Keep a torch fallback only for cpu/unsupported."),
    },
    {
        "op_classes": ("moe", "elementwise"),
        "form": "torch_fused_first",
        "advice": ("Prefer torch/torch_npu fused ops first (permute/sort/gather/index families). "
                   "Measured: a pure-torch solution reached 12.3x on moe_permute while a forced triton "
                   "kernel failed to pass at all. Only move to a kernel if the profiler shows real "
                   "headroom in the fused path."),
    },
)


def form_for(feats: dict) -> str:
    name = str(feats.get("name") or "").lower()
    if "swiglu" in name or "swi_glu" in name:
        # measured: the triton SwiGlu solution passed 20/20 and scored 75.84 on cann-bench 910c
        # (the official triton example scored 10/20); this overrides the generic elementwise prior.
        return "triton_kernel"
    op_class = feats.get("op_class") or ""
    for prior in FORM_PRIORS:
        if op_class in prior["op_classes"]:
            return prior["form"]
    return ""


def form_hint(feats: dict) -> str:
    """Deterministic form guidance for the task prompt ('' when we have no measurement)."""
    name = str(feats.get("name") or "").lower()
    if "swiglu" in name or "swi_glu" in name:
        return ("## Implementation form prior (measured on this hardware)\n"
                "- Prefer the triton-ascend kernel: the SwiGlu triton solution passed 20/20 and "
                "scored 75.84 on cann-bench 910c, while the official triton example passed only 10/20. "
                "Handle the half==1 adjacent-pair case and the general [R,C] tile separately.")
    op_class = feats.get("op_class") or ""
    for prior in FORM_PRIORS:
        if op_class in prior["op_classes"]:
            return "## Implementation form prior (measured on this hardware)\n- %s" % prior["advice"]
    return ""