"""Example solution: hand-written softmax (correct; a plausible agent first draft)."""
import torch


def run(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)