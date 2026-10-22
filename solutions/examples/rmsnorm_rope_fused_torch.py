"""Starting-point solution for rmsnorm_rope_fused (sequential, no concat)."""
import torch

EPS = 1e-5


def run(x, w, cos, sin):
    rn = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * w
    half = x.shape[-1] // 2
    r1, r2 = rn[..., :half], rn[..., half:]
    out = torch.empty_like(rn)
    out[..., :half] = r1 * cos - r2 * sin
    out[..., half:] = r2 * cos + r1 * sin
    return out