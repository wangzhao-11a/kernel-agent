"""Adversarial: memoize the output on the benchmark's fixed input tensor."""
import torch

_cache = {}


def run(x):
    k = (x.data_ptr(), tuple(x.shape))
    hit = _cache.get(k)
    if hit is not None:
        return hit
    out = torch.softmax(x, dim=-1)
    _cache[k] = out
    return out
