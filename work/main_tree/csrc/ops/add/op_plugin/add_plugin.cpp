/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_api.cpp
 * \brief Add API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/add_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("add(Tensor x, Tensor y) -> Tensor");
}

torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("add", add_meta);
}

torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = add_meta(x, y);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_add_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            ADD_KERNEL_LAUNCH_FLOAT(x_ptr, y_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        } else if (x.scalar_type() == torch::kFloat16) {
            ADD_KERNEL_LAUNCH_HALF(x_ptr, y_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        } else if (x.scalar_type() == torch::kInt32) {
            ADD_KERNEL_LAUNCH_INT32(x_ptr, y_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Add", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("add", add_npu);
}

} // namespace cann_bench