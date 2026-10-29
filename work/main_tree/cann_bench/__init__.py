#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""CANN Bench - Ascend operator benchmarks"""
__version__ = "1.0.0"
import torch
from typing import List, Optional, Tuple

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Cannot import _C. Please make sure the `cann_bench` package is properly installed. "
    ) from e

# Direct function calls: cann_bench.add(x, y)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.add(x, y)

def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.sqrt(x)

# Also accessible via torch.ops.cann_bench.add() and torch.ops.cann_bench.sqrt()

def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.ops.cann_bench.swi_glu(input, dim)


def gru(x: torch.Tensor, weight_ih: List[torch.Tensor], weight_hh: List[torch.Tensor], inputSize: int,
        hiddenSize: int, numLayers: int, bias: bool = True, batchFirst: bool = False, dropout: float = 0.0,
        bidirectional: bool = False, bias_ih: Optional[List[torch.Tensor]] = None,
        bias_hh: Optional[List[torch.Tensor]] = None,
        h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.cann_bench.gru(x, weight_ih, weight_hh, inputSize, hiddenSize, numLayers, bias,
                                    batchFirst, dropout, bidirectional, bias_ih, bias_hh, h0)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    return torch.ops.cann_bench.nms(boxes, scores, iou_threshold)


def roi_align(x: torch.Tensor, boxes: torch.Tensor, outputHeight: int, outputWidth: int,
              spatial_scale: float, sampling_ratio: int = -1, aligned: bool = False) -> torch.Tensor:
    return torch.ops.cann_bench.roi_align(x, boxes, outputHeight, outputWidth, spatial_scale,
                                          sampling_ratio, aligned)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.ops.cann_bench.softmax(x, dim)


def resize_bilinear(x: torch.Tensor, output_size: Optional[List[int]] = None, align_corners: bool = False,
                    scale_factor: Optional[List[float]] = None) -> torch.Tensor:
    return torch.ops.cann_bench.resize_bilinear(x, output_size, align_corners, scale_factor)


# ---------------------------------------------------------------- elementwise batch
def exp(x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    return torch.ops.cann_bench.exp(x, base, scale, shift)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.sigmoid(x)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return torch.ops.cann_bench.gelu(x, approximate)


def mish(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.mish(x)


def maximum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.maximum(x1, x2)


def gcd(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.gcd(x1, x2)


def masked_scale(x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.ops.cann_bench.masked_scale(x, mask, scale)


def apply_adam_w(var: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                 lr: float, beta1: float, beta2: float, weight_decay: float,
                 epsilon: float = 1e-8, step: int = 1, maximize: bool = False) -> torch.Tensor:
    return torch.ops.cann_bench.apply_adam_w(var, grad, m, v, lr, beta1, beta2, weight_decay,
                                             epsilon, step, maximize)


def _bcast_probe(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench._bcast_probe(t, ref)


# ---------------------------------------------------------------- batch2: row ops
def dynamic_quant(x: torch.Tensor):
    return torch.ops.cann_bench.dynamic_quant(x)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return torch.ops.cann_bench.rms_norm(x, gamma, epsilon)


def arg_max(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    return torch.ops.cann_bench.arg_max(input, dim, keepdim)


def cummin(input: torch.Tensor, dim: int):
    return torch.ops.cann_bench.cummin(input, dim)


def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean",
                       ignore_index: int = -100) -> torch.Tensor:
    return torch.ops.cann_bench.cross_entropy_loss(input, target, reduction, ignore_index)


def group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int,
               epsilon: float) -> torch.Tensor:
    return torch.ops.cann_bench.group_norm(x, gamma, beta, num_groups, epsilon)


# ---------------------------------------------------------------- batch3: index / rotary
def gather(x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.ops.cann_bench.gather(x, index, dim)


def scatter(data: torch.Tensor, dim: int, indices: torch.Tensor, updates: torch.Tensor,
            reduce=None) -> torch.Tensor:
    return torch.ops.cann_bench.scatter(data, dim, indices, updates, reduce)


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor,
                         num_segments: int) -> torch.Tensor:
    return torch.ops.cann_bench.unsorted_segment_sum(data, segment_ids, num_segments)


def apply_rotary_pos_emb(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         layout: int = 0, rotaryMode: str = "half"):
    return torch.ops.cann_bench.apply_rotary_pos_emb(query, key, cos, sin, layout, rotaryMode)


# ---------------------------------------------------------------- batch4: layout / foreach
def transpose(x: torch.Tensor, perm: List[int]) -> torch.Tensor:
    return torch.ops.cann_bench.transpose(x, perm)


def strided_slice(x: torch.Tensor, begin: List[int], end: List[int], strides: List[int],
                  begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
                  shrink_axis_mask: int = 0, new_axis_mask: int = 0) -> torch.Tensor:
    return torch.ops.cann_bench.strided_slice(x, begin, end, strides, begin_mask, end_mask,
                                              ellipsis_mask, shrink_axis_mask, new_axis_mask)


def adaptive_avg_pool_3d(x: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    return torch.ops.cann_bench.adaptive_avg_pool_3d(x, output_size)


def foreach_addcdiv_scalar(x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor],
                           scalar: float) -> List[torch.Tensor]:
    return torch.ops.cann_bench.foreach_addcdiv_scalar(x1, x2, x3, scalar)


def foreach_norm(x: List[torch.Tensor], scalar: float) -> List[torch.Tensor]:
    return torch.ops.cann_bench.foreach_norm(x, scalar)


# ---------------------------------------------------------------- batch5
def mhc_sinkhorn(comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6) -> torch.Tensor:
    return torch.ops.cann_bench.mhc_sinkhorn(comb, iter_step, eps)


def add_rms_norm_dynamic_quant(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor,
                               epsilon: float = 1e-6):
    return torch.ops.cann_bench.add_rms_norm_dynamic_quant(x1, x2, gamma, epsilon)


def moe_gating_top_k_softmax(x: torch.Tensor, finished=None, k: int = 1):
    return torch.ops.cann_bench.moe_gating_top_k_softmax(x, finished, k)


def dilation_2d(x: torch.Tensor, filter: torch.Tensor, strides, rates, padding_mode: str = "SAME",
                pads=None, ceil_mode: bool = False, data_format: str = "NHWC") -> torch.Tensor:
    return torch.ops.cann_bench.dilation_2d(x, filter, strides, rates, padding_mode, pads, ceil_mode, data_format)


def moe_re_routing(tokens: torch.Tensor, expert_token_num_per_rank: torch.Tensor,
                   per_token_scales=None, expert_token_num_type: int = 1, idx_type: int = 0):
    return torch.ops.cann_bench.moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales,
                                               expert_token_num_type, idx_type)


def kv_cache_append(k_cache: torch.Tensor, v_cache: torch.Tensor, new_k: torch.Tensor,
                    new_v: torch.Tensor, slot_mapping: torch.Tensor):
    return torch.ops.cann_bench.kv_cache_append(k_cache, v_cache, new_k, new_v, slot_mapping)


def aggregate_hidden(input: torch.Tensor, weight: torch.Tensor, mask=None):
    return torch.ops.cann_bench.aggregate_hidden(input, weight, mask)


def dequant_swiglu_quant(x: torch.Tensor, weight_scale=None, activation_scale=None, quant_scale=None,
                         activate_left: bool = False):
    return torch.ops.cann_bench.dequant_swiglu_quant(x, weight_scale, activation_scale, quant_scale, activate_left)


def engram_gate_fusion(keys, hidden_states, value, norm1_weight, norm2_weight, conv_norm_weight, conv_weight,
                       conv_state=None, hc_mult: int = 4, hidden_size: int = 1024, kernel_size: int = 4,
                       dilation: int = 3, norm_eps: float = 1e-5):
    return torch.ops.cann_bench.engram_gate_fusion(keys, hidden_states, value, norm1_weight, norm2_weight,
                                                   conv_norm_weight, conv_weight, conv_state, hc_mult,
                                                   hidden_size, kernel_size, dilation, norm_eps)


def selective_scan(u, delta, A, B_mat, C_mat, D_skip, z):
    return torch.ops.cann_bench.selective_scan(u, delta, A, B_mat, C_mat, D_skip, z)


_MXQ_RM = {"rint": 0, "even": 0, "floor": 1, "round": 2, "nearest": 2, "ceil": 2, "trunc": 1}


def _mxq_view(t: torch.Tensor, name: str) -> torch.Tensor:
    dt = getattr(torch, name, None)
    if dt is None:
        return t
    try:
        return t.view(dt)
    except Exception:
        return t


def dynamic_mx_quant(x: torch.Tensor, axis: int = -1, round_mode: str = "rint", dst_type: int = 40,
                     blocksize: int = 32):
    nd = x.dim()
    ax = axis + nd if axis < 0 else axis
    rm = round_mode if isinstance(round_mode, str) else "rint"
    y_flat, s_flat = torch.ops.cann_bench.dynamic_mx_quant(x, ax, rm, int(dst_type), int(blocksize))
    is_fp4 = int(dst_type) in (40, 41)
    yshape = list(x.shape)
    if is_fp4:
        yshape[-1] = yshape[-1] // 2
    sshape = list(x.shape)
    sshape[ax] = -(-int(x.shape[ax]) // int(blocksize))
    y = y_flat.reshape(yshape)
    s = s_flat.reshape(sshape)
    if is_fp4:
        s = _mxq_view(s, "float8_e8m0fnu")
    if is_fp4:
        return y, s
    return y, s


def conv_2d(x: torch.Tensor, filter: torch.Tensor, bias=None, strides=(1, 1), pads=(0, 0, 0, 0),
            dilations=(1, 1)):
    if bias is None:
        bias = torch.zeros((filter.shape[0],), dtype=x.dtype, device=x.device)
    return torch.ops.cann_bench.conv_2d(x, filter, bias, list(strides), list(pads), list(dilations))


def fused_causal_conv1d(x, weight, conv_states, query_start_loc=None, cache_indices=None,
                        num_accepted_tokens=None, num_computed_tokens=None,
                        block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None,
                        initial_state_idx=None, pad_slot_id: int = -1, max_query_len: int = -1,
                        residual_connection: int = 1, block_size: int = 128, conv_mode: int = 1,
                        inplace: bool = False, activation="none", run_mode: int = 0):
    return torch.ops.cann_bench.fused_causal_conv1d(
        x, weight, conv_states, query_start_loc, cache_indices, num_accepted_tokens,
        num_computed_tokens, block_idx_first_scheduled_token, block_idx_last_scheduled_token,
        initial_state_idx, int(pad_slot_id), int(max_query_len), int(residual_connection),
        int(block_size), int(conv_mode), bool(inplace), "" if activation is None else str(activation),
        int(run_mode))


def depthwise_conv_2d(x, weight, bias, kernelSize, stride, padding, dilation, groups):
    if bias is None:
        bias = torch.zeros((weight.shape[0],), dtype=x.dtype, device=x.device)
    return torch.ops.cann_bench.depthwise_conv_2d(x, weight, bias, list(kernelSize), list(stride),
                                                  list(padding), list(dilation), int(groups))


def mhc_post_grad(grad_output, x, h_res, h_out, h_post):
    return torch.ops.cann_bench.mhc_post_grad(grad_output, x, h_res, h_out, h_post)


def aggregate_hidden_grad(grad_output, input, weight, mask=None):
    return torch.ops.cann_bench.aggregate_hidden_grad(grad_output, input, weight, mask)

# build-tag: v98-conv-batch

# build-tag: v99-ahg-mhcpg

# build-tag: v101-fcc1d-tmplfix

# build-tag: v102-dmxq

# build-tag: auto-conv-1792773459

# build-tag: v103-shadow-fix

# build-tag: v104-dmxq-verified

# build-tag: v105-mhcpg-verified

# build-tag: v106-dmxq

# build-tag: v107-mhcpg

# build-tag: v108-mhcpg-nullfix

# build-tag: v109-dmxq-nullfix

# build-tag: v110-uss

# build-tag: v111-dmxq-fp8dtype

# build-tag: v112-uss-cast


def top_k(x: torch.Tensor, k: int, dim: int, largest: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.cann_bench.top_k(x, k, dim, largest)

# build-tag: v113-topk

# build-tag: v113-lstm


def lstm(x: torch.Tensor, weight_ih: List[torch.Tensor], weight_hh: List[torch.Tensor],
         inputSize: int, hiddenSize: int, numLayers: int = 1, bias: bool = True,
         batchFirst: bool = False, dropout: float = 0.0, bidirectional: bool = False,
         projSize: int = 0, bias_ih: Optional[List[torch.Tensor]] = None,
         bias_hh: Optional[List[torch.Tensor]] = None,
         weight_hr: Optional[List[torch.Tensor]] = None,
         h0: Optional[torch.Tensor] = None,
         c0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.cann_bench.lstm(x, weight_ih, weight_hh, int(inputSize), int(hiddenSize),
                                     int(numLayers), bool(bias), bool(batchFirst), float(dropout),
                                     bool(bidirectional), int(projSize), bias_ih, bias_hh,
                                     weight_hr, h0, c0)


def unique(x, return_inverse: bool = False):
    y, inv = torch.ops.cann_bench.unique(x, bool(return_inverse))
    return (y, inv) if return_inverse else y

# build-tag: v113-unique

# build-tag: v113-topk


def lstm_dbg_math(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.lstm_dbg_math(x, w, s)


# build-tag: v114-c3dbp


def conv_3d_backprop_filter(x: torch.Tensor, grad: torch.Tensor, strides=(1, 1, 1),
                            pads=(0, 0, 0, 0, 0, 0), dilations=(1, 1, 1), groups: int = 1,
                            filter_size=None) -> torch.Tensor:
    if filter_size is None:
        raise ValueError("filter_size is required")
    return torch.ops.cann_bench.conv_3d_backprop_filter(
        x, grad, list(strides), list(pads), list(dilations), int(groups),
        [int(v) for v in filter_size])

# build-tag: v114-grouped-matmul


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, bias=None, group_list=None,
                   split_item: int = 0, transpose_weight: bool = False) -> List[torch.Tensor]:
    """Grouped matmul: y[rows_g] = x[rows_g] @ weight[g] (+ bias[g]), group_list is cumsum."""
    if group_list is None:
        group_list = []
    if torch.is_tensor(group_list):
        group_list = group_list.tolist()
    return torch.ops.cann_bench.grouped_matmul(x, weight, bias, [int(v) for v in group_list],
                                               int(split_item), bool(transpose_weight))

# build-tag: v114-unique

# build-tag: v114b-unique

# build-tag: v115-c3dbp

# build-tag: v116-gmm

# build-tag: v117-lstm

# build-tag: v118-unique-only

# build-tag: v119-c3dbp-only
