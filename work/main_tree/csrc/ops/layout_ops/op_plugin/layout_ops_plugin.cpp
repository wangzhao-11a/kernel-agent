#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include <acl/acl.h>

#include "../op_kernel/layout_ops_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("transpose(Tensor x, int[] perm) -> Tensor");
    m.def("strided_slice(Tensor x, int[] begin, int[] end, int[] strides, int begin_mask, int end_mask, "
          "int ellipsis_mask, int shrink_axis_mask, int new_axis_mask) -> Tensor");
    m.def("adaptive_avg_pool_3d(Tensor x, int[] output_size) -> Tensor");
    m.def("foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[]");
    m.def("foreach_norm(Tensor[] x, float scalar) -> Tensor[]");
}

static int64_t layout_tile(int64_t inner)
{
    return std::get<0>(calc_layout_tiling(inner));
}

// KAG_TR_HW=0 disables the 16x16 hardware-transpose path (default: on for 16-bit dtypes)
static bool env_on(const char *name, bool dflt)
{
    const char *v = std::getenv(name);
    if (v == nullptr) return dflt;
    return !(v[0] == '0' && v[1] == '\0');
}

// ------------------------------------------------------------------ transpose
torch::Tensor transpose_npu(const torch::Tensor &x, std::vector<int64_t> perm)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    const int64_t nd = xc.dim();
    std::vector<int64_t> oshape;
    for (auto p : perm) oshape.push_back(xc.size(p >= 0 ? p : p + nd));
    auto y = torch::empty(oshape, xc.options());
    if (xc.numel() == 0) return y;
    int64_t st[8] = {0}, sh[8] = {1}, osz[8] = {1};
    for (int k = 0; k < nd; ++k) {
        st[k] = xc.stride(perm[k]);
        sh[k] = oshape[k];
    }
    const int64_t inner = oshape[nd - 1];
    const int64_t outer = xc.numel() / std::max<int64_t>(inner, 1);
    const int64_t tile = layout_tile(inner);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)xc.data_ptr();
    auto yp = (GM_ADDR)y.data_ptr();
    // "last two output dims swapped, everything else identical" -> batched 2-D transpose of
    //   nBatch matrices (H x W -> W x H); 16-bit dtypes use the 16x16 hardware transpose.
    bool swapLast = (nd >= 2) && (perm[nd - 1] == nd - 2) && (perm[nd - 2] == nd - 1);
    for (int64_t k = 0; k + 2 < nd && swapLast; ++k) {
        if (perm[k] != k) swapLast = false;
    }
    if (swapLast && env_on("KAG_TR_HW", true) && xc.element_size() == 2) {
        const int64_t H = xc.size(nd - 2), W = xc.size(nd - 1);
        if (H >= 16 && W >= 16) {
            const int64_t nBatch = xc.numel() / (H * W);
            bool done = true;
#define TRHW_CALL(SUF) launch_transpose2d_hw_##SUF(xp, yp, nBatch, H, W, stream)
            switch (xc.scalar_type()) {
                case torch::kFloat16: TRHW_CALL(f16); break;
                case torch::kBFloat16: TRHW_CALL(bf16); break;
                case torch::kInt16: TRHW_CALL(i16); break;
                default: done = false; break;
            }
#undef TRHW_CALL
            if (done) return y;
        }
    }
    // 2-D transpose: tiled UB transpose (64x64 blocks) is far faster than the scalar path
    if (nd == 2 && perm[0] == 1 && perm[1] == 0 && xc.size(0) >= 64 && xc.size(1) >= 64) {
        const int64_t H = xc.size(0), W = xc.size(1);
        const int64_t T = 64;
#define TR2_CALL(SUF) launch_transpose2d_tiled_##SUF(xp, yp, H, W, T, stream)
        switch (xc.scalar_type()) {
            case torch::kFloat32: TR2_CALL(f32); break;
            case torch::kFloat16: TR2_CALL(f16); break;
            case torch::kBFloat16: TR2_CALL(bf16); break;
            case torch::kInt8: TR2_CALL(i8); break;
            case torch::kInt16: TR2_CALL(i16); break;
            case torch::kInt32: TR2_CALL(i32); break;
            case torch::kInt64: TR2_CALL(i64); break;
            default: break;
        }
#undef TR2_CALL
        return y;
    }
    const int64_t trStride = env_on("KAG_TR_CONTIG", true) ? st[nd - 1] : -st[nd - 1];
#define TR_CALL(SUF)                                                                                        \
    launch_transpose_##SUF(xp, yp, outer, inner, trStride, nd, st[0], st[1], st[2], st[3], st[4], st[5],      \
                           st[6], st[7], sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], sh[6], sh[7], tile, stream)
    switch (xc.scalar_type()) {
        case torch::kFloat32: TR_CALL(float); break;
        case torch::kFloat16: TR_CALL(half); break;
        case torch::kBFloat16: TR_CALL(bf16); break;
        case torch::kInt8: TR_CALL(i8); break;
        case torch::kInt16: TR_CALL(i16); break;
        case torch::kInt32: TR_CALL(i32); break;
        case torch::kInt64: TR_CALL(i64); break;
        default: TORCH_CHECK(false, "transpose: unsupported dtype");
    }
#undef TR_CALL
    return y;
}

// ------------------------------------------------------------------ strided slice
torch::Tensor strided_slice_npu(const torch::Tensor &x, std::vector<int64_t> begin, std::vector<int64_t> end,
                                std::vector<int64_t> strides, int64_t begin_mask, int64_t end_mask,
                                int64_t ellipsis_mask, int64_t shrink_axis_mask, int64_t new_axis_mask)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    const int64_t nd = xc.dim();
    const int64_t np = static_cast<int64_t>(begin.size());
    int64_t ell = -1;
    for (int i = 0; i < 32; ++i) {
        if (ellipsis_mask & (1LL << i)) { ell = i; break; }
    }
    int64_t nNew = 0;
    for (int64_t i = 0; i < np; ++i) {
        if (new_axis_mask & (1LL << i)) nNew++;
    }
    const int64_t nEll = (ell >= 0) ? (nd - (np - nNew - 1)) : 0;

    // output dims, in order: srcDim (-1 for a new axis), start, step
    std::vector<int64_t> srcDim, startV, stepV;
    std::vector<int64_t> oshape;
    int64_t inDim = 0;
    int64_t p = 0;
    int64_t base0 = 0;
    while (inDim < nd || p < np) {
        if (p < np && (new_axis_mask & (1LL << p))) {
            srcDim.push_back(-1);
            startV.push_back(0);
            stepV.push_back(1);
            oshape.push_back(1);
            p++;
            continue;
        }
        if (ell >= 0 && p == ell) {
            for (int64_t e = 0; e < nEll; ++e) {
                srcDim.push_back(inDim);
                startV.push_back(0);
                stepV.push_back(1);
                oshape.push_back(xc.size(inDim));
                inDim++;
            }
            p++;
            continue;
        }
        if (inDim < nd && p < np) {
            const int64_t dimSize = xc.size(inDim);
            int64_t b = (p < static_cast<int64_t>(begin.size())) ? begin[p] : 0;
            int64_t e = (p < static_cast<int64_t>(end.size())) ? end[p] : dimSize;
            int64_t s = (p < static_cast<int64_t>(strides.size())) ? strides[p] : 1;
            if (b < 0) b += dimSize;
            if (e < 0) e += dimSize;
            if (begin_mask & (1LL << p)) b = (s > 0) ? 0 : dimSize - 1;
            if (end_mask & (1LL << p)) e = (s > 0) ? dimSize : -1;
            if (s > 0) {
                if (b < 0) b = 0;
                if (b > dimSize) b = dimSize;
                if (e < 0) e = 0;
                if (e > dimSize) e = dimSize;
            }
            if (shrink_axis_mask & (1LL << p)) {
                base0 += b * xc.stride(inDim);
            } else {
                int64_t cnt = (s > 0) ? ((e - b + s - 1) / s) : ((b - e - s - 1) / (-s));
                if (cnt < 0) cnt = 0;
                srcDim.push_back(inDim);
                startV.push_back(b);
                stepV.push_back(s);
                oshape.push_back(cnt);
            }
            inDim++;
            p++;
        } else if (inDim < nd) {
            srcDim.push_back(inDim);
            startV.push_back(0);
            stepV.push_back(1);
            oshape.push_back(xc.size(inDim));
            inDim++;
        } else {
            p++;
        }
    }
    auto y = torch::empty(oshape, xc.options());
    if (y.numel() == 0) return y;
    const int64_t no = static_cast<int64_t>(oshape.size());
    int64_t st[8] = {0}, sp[8] = {0}, sh[8] = {1};
    for (int64_t k = 0; k < no; ++k) {
        sh[k] = oshape[k];
        if (srcDim[k] < 0) {
            st[k] = 0;
            sp[k] = 0;
        } else {
            st[k] = xc.stride(srcDim[k]);
            sp[k] = stepV[k];
        }
    }
    const int64_t inner = oshape[no - 1];
    const int64_t outer = y.numel() / std::max<int64_t>(inner, 1);
    const int64_t lastSrc = (srcDim[no - 1] < 0) ? 0 : xc.stride(srcDim[no - 1]);
    const int64_t tile = layout_tile(inner);
    // negative tile => scalar strided path (A/B switch), see slice_kernel
    const int64_t slTile = env_on("KAG_SL_SPAN", true) ? tile : -tile;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)xc.data_ptr();
    auto yp = (GM_ADDR)y.data_ptr();
#define SL_CALL(SUF)                                                                                        \
    launch_slice_##SUF(xp, yp, outer, inner, startV[no - 1], stepV[no - 1], lastSrc, no, base0, st[0],    \
                       st[1], st[2], st[3], st[4], st[5], st[6], st[7], sp[0], sp[1], sp[2], sp[3], sp[4],     \
                       sp[5], sp[6], sp[7], sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], sh[6], sh[7], slTile,     \
                       stream)
    switch (xc.scalar_type()) {
        case torch::kFloat32: SL_CALL(float); break;
        case torch::kFloat16: SL_CALL(half); break;
        case torch::kBFloat16: SL_CALL(bf16); break;
        case torch::kInt8: SL_CALL(i8); break;
        case torch::kInt32: SL_CALL(i32); break;
        case torch::kInt64: SL_CALL(i64); break;
        default: TORCH_CHECK(false, "strided_slice: unsupported dtype");
    }
#undef SL_CALL
    return y;
}

// ------------------------------------------------------------------ adaptive avg pool 3d
torch::Tensor adaptive_avg_pool_3d_npu(const torch::Tensor &x, std::vector<int64_t> output_size)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    TORCH_CHECK(xc.dim() == 5, "adaptive_avg_pool_3d expects 5D input");
    const int64_t N = xc.size(0), C = xc.size(1), D = xc.size(2), H = xc.size(3), W = xc.size(4);
    const int64_t OD = output_size[0], OH = output_size[1], OW = output_size[2];
    auto y = torch::empty({N, C, OD, OH, OW}, xc.options());
    if (y.numel() == 0) return y;
    const int64_t tile = layout_tile(OW);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    if (xc.scalar_type() == torch::kFloat32) {
        launch_adapt_pool_float((GM_ADDR)xc.data_ptr(), (GM_ADDR)y.data_ptr(), N, C, D, H, W, OD, OH, OW, tile, stream);
    } else if (xc.scalar_type() == torch::kFloat16) {
        launch_adapt_pool_half((GM_ADDR)xc.data_ptr(), (GM_ADDR)y.data_ptr(), N, C, D, H, W, OD, OH, OW, tile, stream);
    } else if (xc.scalar_type() == torch::kBFloat16) {
        launch_adapt_pool_bf16((GM_ADDR)xc.data_ptr(), (GM_ADDR)y.data_ptr(), N, C, D, H, W, OD, OH, OW, tile, stream);
    } else {
        TORCH_CHECK(false, "adaptive_avg_pool_3d: unsupported dtype");
    }
    return y;
}

// ------------------------------------------------------------------ foreach addcdiv scalar
std::vector<torch::Tensor> foreach_addcdiv_scalar_npu(c10::List<at::Tensor> x1, c10::List<at::Tensor> x2,
                                                      c10::List<at::Tensor> x3, double scalar)
{
    std::vector<torch::Tensor> out;
    const c10::OptionalDeviceGuard guard(x1.get(0).device());
    for (size_t i = 0; i < x1.size(); ++i) {
        auto a = x1.get(i).contiguous();
        auto b = x2.get(i).contiguous();
        auto c = x3.get(i).contiguous();
        auto y = torch::empty_like(a);
        const int64_t total = a.numel();
        if (total > 0) {
            int64_t cores = 40;
            int64_t blocks = std::min<int64_t>(cores, (total + 1023) / 1024);
            if (blocks < 1) blocks = 1;
            int64_t blockLen = (total + blocks - 1) / blocks;
            int64_t tile = std::get<0>(calc_layout_tiling(blockLen));
            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            if (a.scalar_type() == torch::kFloat32) {
                launch_addcdiv_float((GM_ADDR)a.data_ptr(), (GM_ADDR)b.data_ptr(), (GM_ADDR)c.data_ptr(),
                                     (GM_ADDR)y.data_ptr(), total, blocks, blockLen, tile,
                                     static_cast<float>(scalar), stream);
            } else if (a.scalar_type() == torch::kFloat16) {
                launch_addcdiv_half((GM_ADDR)a.data_ptr(), (GM_ADDR)b.data_ptr(), (GM_ADDR)c.data_ptr(),
                                    (GM_ADDR)y.data_ptr(), total, blocks, blockLen, tile,
                                    static_cast<float>(scalar), stream);
            } else {
                TORCH_CHECK(false, "foreach_addcdiv_scalar: unsupported dtype");
            }
        }
        out.push_back(y);
    }
    return out;
}

// ------------------------------------------------------------------ foreach norm
std::vector<torch::Tensor> foreach_norm_npu(c10::List<at::Tensor> xs, double p)
{
    std::vector<torch::Tensor> out;
    float pv = static_cast<float>(p);
    int32_t mode = 0;
    if (std::isinf(p)) mode = (p > 0) ? 1 : 2;
    for (size_t i = 0; i < xs.size(); ++i) {
        auto x = xs.get(i).contiguous();
        auto y = torch::empty({}, x.options());
        const int64_t total = x.numel();
        int64_t cores = 40;
        int64_t blocks = std::min<int64_t>(cores, (total + 1023) / 1024);
        if (blocks < 1) blocks = 1;
        int64_t blockLen = (total + blocks - 1) / blocks;
        int64_t tile = std::get<0>(calc_layout_tiling(blockLen));
        auto ws = torch::empty({blocks}, x.options().dtype(torch::kFloat32));
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        if (x.scalar_type() == torch::kFloat32) {
            launch_norm_partial_float((GM_ADDR)x.data_ptr(), (GM_ADDR)ws.data_ptr(), total, blocks, blockLen, tile,
                                      pv, mode, stream);
        } else if (x.scalar_type() == torch::kFloat16) {
            launch_norm_partial_half((GM_ADDR)x.data_ptr(), (GM_ADDR)ws.data_ptr(), total, blocks, blockLen, tile,
                                     pv, mode, stream);
        } else {
            TORCH_CHECK(false, "foreach_norm: unsupported dtype");
        }
        launch_norm_finalize((GM_ADDR)ws.data_ptr(), (GM_ADDR)y.data_ptr(), blocks, pv, mode,
                             x.scalar_type() == torch::kFloat32 ? 1 : 0, stream);
        out.push_back(y);
    }
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("transpose", transpose_npu);
    m.impl("strided_slice", strided_slice_npu);
    m.impl("adaptive_avg_pool_3d", adaptive_avg_pool_3d_npu);
    m.impl("foreach_addcdiv_scalar", foreach_addcdiv_scalar_npu);
    m.impl("foreach_norm", foreach_norm_npu);
}

}  // namespace cann_bench
