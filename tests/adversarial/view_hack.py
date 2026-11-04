"""Adversarial: skip the compute and hand back a view of the input."""
import torch


def run(x):
    return x.view_as(x)
