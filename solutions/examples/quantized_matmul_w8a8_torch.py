"""Starting-point solution for quantized_matmul_w8a8 (dequantize-then-matmul; slow path)."""
import torch


def run(a, w, scale_a, scale_w):
    return (a.to(torch.float32) * scale_a) @ (w.to(torch.float32) * scale_w)