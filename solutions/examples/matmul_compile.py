"""Example solution: torch.compile with a graceful eager fallback.

Shows the pattern for "try the fancy path, fall back to plain torch" — useful on
machines where torch.compile has no working inductor backend (e.g. Windows without
MSVC, or Ascend NPU until the backend lands).
"""
import torch


def _run_impl(x, w):
    return x @ w


_compiled = None


def run(x, w):
    global _compiled
    if _compiled is None:
        try:
            _compiled = torch.compile(_run_impl)
        except Exception:
            _compiled = _run_impl
    try:
        return _compiled(x, w)
    except Exception:
        return _run_impl(x, w)