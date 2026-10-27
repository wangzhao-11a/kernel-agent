#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <cstdlib>

#include "../op_kernel/swiglu_launch.h"

#ifndef SWIGLU_HAS_SIMT
#define SWIGLU_HAS_SIMT 0
#endif

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("swi_glu(Tensor input, int dim=-1) -> Tensor");
}

torch::Tensor swi_glu_meta(const torch::Tensor &input, int64_t dim)
{
    int64_t ndim = input.dim();
    TORCH_CHECK(ndim >= 1, "swi_glu expects at least one dimension");
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");
    TORCH_CHECK(input.size(dim) % 2 == 0, "the split dimension must have an even size");
    auto sizes = input.sizes().vec();
    sizes[dim] /= 2;
    return torch::empty(sizes, input.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("swi_glu", swi_glu_meta);
}

torch::Tensor swi_glu_npu(const torch::Tensor &input, int64_t dim)
{
    const c10::OptionalDeviceGuard guard(input.device());
    auto x = input.contiguous();
    auto y = swi_glu_meta(x, dim);
    int64_t ndim = x.dim();
    if (dim < 0) {
        dim += ndim;
    }
    int64_t half = x.size(dim) / 2;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        half *= x.size(i);
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer *= x.size(i);
    }
    int64_t total = y.numel();
    if (total == 0) {
        return y;
    }
    int64_t numBlocks = 0;
    int64_t rowsPerBlock = 0;
    uint32_t tileBytes = 0;
    std::tie(numBlocks, rowsPerBlock, tileBytes) = calc_swiglu_tiling_params(outer, half, static_cast<int64_t>(x.element_size()));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto dtype = x.scalar_type();

    bool aligned = ((half * static_cast<int64_t>(x.element_size())) % 32 == 0);
    int64_t rowBytes = half * static_cast<int64_t>(x.element_size());
    int64_t simtBlocks = swiglu_core_num();   // SIMT wants every AIV core, not the tile count
    // test hook: SWIGLU_FORCE_TILE=1 routes non-aligned rows through the tile/R=1 path for A/B runs
    bool forceTile = (std::getenv("SWIGLU_FORCE_TILE") != nullptr);
    // Non-aligned rows take the tile/R=1 path only when a row is long enough to amortize the
    // per-tile DMA setup; short rows (R=1 => one row per tile) are far faster on SIMT.
    // Measured on 950pr: 2MB rows tile 8.2us vs simt 17.5us; 1KB rows tile 485us vs simt 6.8us.
    bool simtPath = (!aligned) && (rowBytes < 64 * 1024);
    // 910B has no SIMT. Tiny rows (half*sizeof(T) < 32B, many rows) make the tile path
    // degenerate to one element per DataCopyPad; use the Gather-based tiny path instead.
    bool tinyPath = (!aligned) && (!simtPath || !SWIGLU_HAS_SIMT) && (rowBytes < 32) && (outer >= 4096);
    int64_t tinyBlocks = 0;
    int64_t tinyRowsPerBlock = 0;
    uint32_t tinyTileBytes = 0;
    if (tinyPath) {
        std::tie(tinyBlocks, tinyRowsPerBlock, tinyTileBytes) =
            calc_swiglu_tiny_tiling_params(outer, half, static_cast<int64_t>(x.element_size()));
    }
    auto acl_call = [=]() -> int {
#if SWIGLU_HAS_SIMT
        if (simtPath && !forceTile) {   // SIMT for short non-aligned rows, tile/DMA otherwise
        if (dtype == torch::kFloat32) {
                launch_swiglu_simt_float(x_ptr, y_ptr, total, half, simtBlocks, stream);
            } else if (dtype == torch::kFloat16) {
                launch_swiglu_simt_half(x_ptr, y_ptr, total, half, simtBlocks, stream);
            } else if (dtype == torch::kBFloat16) {
                launch_swiglu_simt_bf16(x_ptr, y_ptr, total, half, simtBlocks, stream);
            }
            return 0;
        }
#endif  // SWIGLU_HAS_SIMT
        if (tinyPath) {
            if (dtype == torch::kFloat32) {
                launch_swiglu_tiny_float(x_ptr, y_ptr, outer, half, tinyRowsPerBlock, tinyTileBytes, tinyBlocks, stream);
            } else if (dtype == torch::kFloat16) {
                launch_swiglu_tiny_half(x_ptr, y_ptr, outer, half, tinyRowsPerBlock, tinyTileBytes, tinyBlocks, stream);
            } else if (dtype == torch::kBFloat16) {
                launch_swiglu_tiny_bf16(x_ptr, y_ptr, outer, half, tinyRowsPerBlock, tinyTileBytes, tinyBlocks, stream);
            } else {
                TORCH_CHECK(false, "swi_glu supports float32/float16/bfloat16");
            }
            return 0;
        }
        if (dtype == torch::kFloat32) {
            launch_swiglu_kernel_float(x_ptr, y_ptr, outer, half, rowsPerBlock, tileBytes, numBlocks, stream);
        } else if (dtype == torch::kFloat16) {
            launch_swiglu_kernel_half(x_ptr, y_ptr, outer, half, rowsPerBlock, tileBytes, numBlocks, stream);
        } else if (dtype == torch::kBFloat16) {
            launch_swiglu_kernel_bf16(x_ptr, y_ptr, outer, half, rowsPerBlock, tileBytes, numBlocks, stream);
        } else {
            TORCH_CHECK(false, "swi_glu supports float32/float16/bfloat16");
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("SwiGlu", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("swi_glu", swi_glu_npu);
}

} // namespace cann_bench
