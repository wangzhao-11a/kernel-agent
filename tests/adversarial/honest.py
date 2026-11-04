"""Control: an honest solution with the same work as the reference.

Used to check that the hardened harness still accepts legitimate solutions and
does not report a phantom speedup for them.
"""
import torch


def run(x):
    return torch.softmax(x, dim=-1)
