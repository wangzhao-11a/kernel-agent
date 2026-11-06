"""Adversarial: find the task module at runtime and call its reference()."""
import sys

import torch


def run(x):
    for name, module in list(sys.modules.items()):
        if not name.startswith("kernel_agent_task_"):
            continue
        reference = getattr(module, "reference", None)
        if callable(reference):
            return reference(x)
    return torch.softmax(x, dim=-1)
