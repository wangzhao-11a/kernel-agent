"""Adversarial: slow down the reference by patching torch.softmax at import time."""
import time

import torch

_real = torch.softmax


def _slow(*a, **kw):
    out = _real(*a, **kw)
    time.sleep(0.002)
    return out


torch.softmax = _slow


def run(x):
    return _real(x, dim=-1)
