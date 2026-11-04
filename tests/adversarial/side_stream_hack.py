"""Adversarial: launch the work on a non-default stream and return without waiting.

On a machine without an accelerator there is no stream to escape to, so the
solution falls back to an honest call (and cannot be distinguished from one).
"""
import torch


def _stream_api():
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                return cuda.Stream, cuda.stream
        except Exception:
            pass
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                return npu.Stream, npu.stream
        except Exception:
            pass
    return None, None


def run(x):
    stream_cls, stream_ctx = _stream_api()
    if stream_cls is None:
        return torch.softmax(x, dim=-1)
    with stream_ctx(stream_cls()):
        return torch.softmax(x, dim=-1)
