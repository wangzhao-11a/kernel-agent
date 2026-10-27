#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include <acl/acl.h>

#include "../op_kernel/row_ops_launch.h"
#include "../../layout_ops/op_kernel/layout_ops_launch.h"

namespace cann_bench {
namespace {

// parameters for the layout_ops transpose primitive: y[r, i] = x[base(r) + i*inStrideInner]
struct RowSwapTr {
    int64_t st[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t sh[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t inner = 1;
    int64_t outer = 0;
    int64_t isi = 1;
    int64_t nd = 1;
    int64_t tile = 256;
};

inline RowSwapTr rowSwapParams(const torch::Tensor &t, int64_t dim)
{
    RowSwapTr p;
    const int64_t nd = t.dim();
    p.nd = nd;
    for (int64_t k = 0; k < nd; ++k) {
        const int64_t src = (k == dim) ? (nd - 1) : ((k == nd - 1) ? dim : k);
        p.st[k] = t.stride(src);
        p.sh[k] = t.size(src);
    }
    p.inner = p.sh[nd - 1];
    p.outer = (p.inner > 0) ? (t.numel() / p.inner) : 0;
    p.isi = p.st[nd - 1];
    p.tile = p.inner < 256 ? p.inner : 256;
    p.tile = (p.tile / 8) * 8;
    if (p.tile < 8) p.tile = 8;
    return p;
}

}  // namespace


TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("dynamic_quant(Tensor x) -> (Tensor, Tensor)");
    m.def("rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor");
    m.def("arg_max(Tensor input, int dim, bool keepdim=False) -> Tensor");
    m.def("cummin(Tensor input, int dim) -> (Tensor, Tensor)");
    m.def("cross_entropy_loss(Tensor input, Tensor target, str reduction=\"mean\", int ignore_index=-100) -> Tensor");
    m.def("group_norm(Tensor x, Tensor gamma, Tensor beta, int num_groups, float epsilon) -> Tensor");
}

static std::tuple<int64_t, int64_t, int64_t> row_tile(int64_t cols, int64_t es)
{
    int64_t a, b, c;
    std::tie(a, b, c) = calc_row_tiling(cols, es);
    return std::make_tuple(a, b, c);
}

// ---- optimisation switches (A/B through env vars, no rebuild needed)
//   KAG_RMS2=0    -> old rms_norm kernel + old tiling          (default: v2 on)
//   KAG_AM2=0     -> old arg_max kernel + old tiling           (default: v2 on)
//   KAG_AM_SPR=0  -> arg_max v2 kernel but Compare/Select index (default: SPR index)
static bool env_on(const char *name, bool dflt)
{
    const char *v = std::getenv(name);
    if (v == nullptr) return dflt;
    return !(v[0] == '0' && v[1] == '\0');
}

static std::tuple<int64_t, int64_t, int64_t> row_tile2(int64_t cols, int64_t es)
{
    int64_t a, b, c;
    std::tie(a, b, c) = calc_row_tiling2(cols, es);
    return std::make_tuple(a, b, c);
}

static std::tuple<int64_t, int64_t, int64_t> dq_tile(int64_t cols, int64_t es)
{
    int64_t a, b, c;
    std::tie(a, b, c) = calc_dq_tiling(cols, es);
    return std::make_tuple(a, b, c);
}

static std::tuple<int64_t, int64_t, int64_t> argmax_tile(int64_t cols, int64_t es)
{
    int64_t a, b, c;
    std::tie(a, b, c) = calc_argmax_tiling(cols, es);
    return std::make_tuple(a, b, c);
}

torch::Tensor rms_norm_npu(const torch::Tensor &x, const torch::Tensor &gamma, double epsilon)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto g = gamma.contiguous();
    auto y = torch::empty_like(xc);
    const int64_t cols = xc.size(-1);
    const int64_t rows = xc.numel() / cols;
    if (cols == 0 || rows == 0) return y;
    // unverified on device -> default off (A/B payload turns it on); OFF path = original kernel
    const bool v2 = env_on("KAG_RMS2", true);
    // v3 batches R rows per UB tile. It needs 32B-aligned rows and a u8 row repeat stride;
    // otherwise fall back to the per-row v2 path.
    const bool v3 = env_on("KAG_RMS3", v2);
    // v4 keeps one full row resident for cols that do not fit the v2 tile (the v2 path
    // would otherwise read x twice). Only used when v3 cannot batch rows.
    const bool v4 = env_on("KAG_RMS4", v2);
    int64_t tile = v2 ? std::get<0>(row_tile2(cols, xc.element_size()))
                      : std::get<0>(row_tile(cols, xc.element_size()));
    const int64_t es = xc.element_size();
    int64_t tile3 = 0;
    std::tie(tile3, std::ignore, std::ignore) = calc_row_tiling3(cols, es);
    // Batched reduce accumulates full 64-column groups; a partial trailing group is
// not handled correctly yet, so keep those shapes on v2.
    const bool batched = v3 && (cols < 64 || cols % 64 == 0) && (cols / 8) <= 255 &&
                         (cols * es) % 32 == 0 && (tile3 / cols) >= 2;
    const bool fullRow = v4 && !batched && cols > tile && rms_full_row_fits(cols, es);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)xc.data_ptr();
    auto gp = (GM_ADDR)g.data_ptr();
    auto yp = (GM_ADDR)y.data_ptr();
    const float eps = static_cast<float>(epsilon);
    at_npu::native::OpCommand::RunOpApi("RmsNorm", [=]() -> int {
    switch (xc.scalar_type()) {
        case torch::kFloat32: if (batched) launch_rms_norm3_float(xp, gp, yp, rows, cols, tile3, eps, stream);
                              else if (fullRow) launch_rms_norm4_float(xp, gp, yp, rows, cols, cols, eps, stream);
                              else if (v2) launch_rms_norm2_float(xp, gp, yp, rows, cols, tile, eps, stream);
                              else launch_rms_norm_float(xp, gp, yp, rows, cols, tile, eps, stream); break;
        case torch::kFloat16: if (batched) launch_rms_norm3_half(xp, gp, yp, rows, cols, tile3, eps, stream);
                              else if (fullRow) launch_rms_norm4_half(xp, gp, yp, rows, cols, cols, eps, stream);
                              else if (v2) launch_rms_norm2_half(xp, gp, yp, rows, cols, tile, eps, stream);
                              else launch_rms_norm_half(xp, gp, yp, rows, cols, tile, eps, stream); break;
        case torch::kBFloat16: if (batched) launch_rms_norm3_bf16(xp, gp, yp, rows, cols, tile3, eps, stream);
                              else if (fullRow) launch_rms_norm4_bf16(xp, gp, yp, rows, cols, cols, eps, stream);
                              else if (v2) launch_rms_norm2_bf16(xp, gp, yp, rows, cols, tile, eps, stream);
                              else launch_rms_norm_bf16(xp, gp, yp, rows, cols, tile, eps, stream); break;
        default: TORCH_CHECK(false, "rms_norm: unsupported dtype");
    }
        return 0;
    });
    return y;
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_quant_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    TORCH_CHECK(xc.dim() >= 2, "dynamic_quant expects >= 2 dims");
    auto y = torch::empty(xc.sizes(), xc.options().dtype(torch::kInt8));
    auto shape = xc.sizes().vec();
    shape.pop_back();
    auto scale = torch::empty(shape, xc.options().dtype(torch::kFloat32));
    const int64_t cols = xc.size(-1);
    const int64_t rows = xc.numel() / cols;
    if (cols == 0 || rows == 0) return {y, scale};
    // v2 path: UB-sized fp32 row cache + vector int8 narrowing. Each step can be switched off
    // from the environment so one build can A/B them on the device.
    int64_t tile = std::get<0>(row_tile(cols, xc.element_size()));
    int32_t flags = 0;
    if (env_on("KAG_DQ2", true)) {
        std::tie(tile, std::ignore, std::ignore) = dq_tile(cols, xc.element_size());
        if (env_on("KAG_DQ_1PASS", true)) flags |= 4;
        if (env_on("KAG_DQ_I8", true)) flags |= 1;
        if (env_on("KAG_DQ_MULS", false)) flags |= 2;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)xc.data_ptr();
    auto yp = (GM_ADDR)y.data_ptr();
    auto sp = (GM_ADDR)scale.data_ptr();
    if (xc.scalar_type() == torch::kFloat16) {
        launch_dynamic_quant_half(xp, yp, sp, rows, cols, tile, flags, stream);
    } else if (xc.scalar_type() == torch::kBFloat16) {
        launch_dynamic_quant_bf16(xp, yp, sp, rows, cols, tile, flags, stream);
    } else {
        TORCH_CHECK(false, "dynamic_quant: only fp16/bf16");
    }
    return {y, scale};
}

torch::Tensor arg_max_npu(const torch::Tensor &input, int64_t dim, bool keepdim)
{
    const c10::OptionalDeviceGuard guard(input.device());
    auto x = input.contiguous();
    const int64_t nd = x.dim();
    if (dim < 0) dim += nd;
    TORCH_CHECK(dim >= 0 && dim < nd, "arg_max: dim out of range");
    torch::Tensor src = x;
    // the reduction length is x.size(dim) whichever path is taken (src is transposed later)
    const int64_t cols = x.size(dim);
    const int64_t rows = x.numel() / cols;
    std::vector<int64_t> oshape;
    for (int64_t i = 0; i < nd; ++i) {
        if (i == dim) {
            if (keepdim) oshape.push_back(1);
        } else {
            oshape.push_back(x.size(i));
        }
    }
    // the kernel writes the destination slot itself (output-order remap for a moved dim),
    // so allocate the final contiguous output and skip the torch-side permute entirely
    auto out = torch::empty(oshape, x.options().dtype(torch::kInt64));
    if (cols == 0 || rows == 0) return out;
    auto stream0 = c10_npu::getCurrentNPUStream().stream(false);
    if (dim != nd - 1 && !env_on("KAG_AM_AXIS", true)) {
        src = x.transpose(dim, nd - 1).contiguous();
        aclrtSynchronizeDevice();
    }
    if (dim != nd - 1 && env_on("KAG_AM_AXIS", true)) {
        // no torch transpose: treat the tensor as [O, K, S] and reduce along K with the
        // contiguous runs along S (the transposed .contiguous() copy fails on the runner)
        int64_t S = 1;
        for (int64_t i = dim + 1; i < nd; ++i) S *= x.size(i);
        const int64_t O = x.numel() / (S * cols);
        int64_t tile = S < 4096 ? S : 4096;
        tile = (tile / 8) * 8;
        if (tile < 8) tile = 8;
        int64_t dbg = 0;
        if (const char *d = std::getenv("KAG_AM_DBG")) {
            // Parse without strtoll: the bisheng sysroot maps it to __isoc23_strtoll
            // (GLIBC_2.38), which the glibc 2.35 runtime container cannot resolve.
            bool neg = (*d == '-');
            for (const char *p = neg ? d + 1 : d; *p >= '0' && *p <= '9'; ++p) {
                dbg = dbg * 10 + (*p - '0');
            }
            if (neg) dbg = -dbg;
        }
        auto xp0 = (GM_ADDR)x.data_ptr();
        auto op0 = (GM_ADDR)out.data_ptr();
        at_npu::native::OpCommand::RunOpApi("ArgMax", [=]() -> int {
            switch (x.scalar_type()) {
                case torch::kFloat32: launch_argmax_axis_float(xp0, op0, O, cols, S, tile, dbg, stream0); break;
                case torch::kFloat16: launch_argmax_axis_half(xp0, op0, O, cols, S, tile, dbg, stream0); break;
                case torch::kBFloat16: launch_argmax_axis_bf16(xp0, op0, O, cols, S, tile, dbg, stream0); break;
                case torch::kInt32: launch_argmax_axis_i32(xp0, op0, O, cols, S, tile, dbg, stream0); break;
                case torch::kInt64: TORCH_CHECK(false, "arg_max: int64 with a moved dim is unsupported"); break;
                default: TORCH_CHECK(false, "arg_max: unsupported dtype");
            }
            return 0;
        });
        return out;
    }
    int64_t rsz[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t ost[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int64_t i = 0; i < nd - 1; ++i) rsz[i] = src.size(i);
    for (int64_t k = 0; k < nd - 1; ++k) {
        // out axis for row axis k: same index before dim, shifted after it, and the
        // swapped-in x.size(nd-1) axis lands on the last output position
        const int64_t pos = (k < dim) ? k : ((k > dim) ? k - 1 : nd - 2);
        int64_t st = 1;
        for (int64_t q = pos + 1; q < static_cast<int64_t>(oshape.size()); ++q) st *= oshape[q];
        ost[k] = st;
    }
    const bool v2 = env_on("KAG_AM2", false);
    // SPR index extraction needs GetReduceRepeatMaxMinSpr semantics confirmed on device;
    // default off so a blind build keeps the known-good Compare/Select index
    const int32_t useSpr = env_on("KAG_AM_SPR", false) ? 1 : 0;
    int64_t tile = v2 ? std::get<0>(argmax_tile(cols, src.element_size()))
                      : std::get<0>(row_tile(cols, src.element_size()));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)src.data_ptr();
    auto op = (GM_ADDR)out.data_ptr();
    at_npu::native::OpCommand::RunOpApi("ArgMax", [=]() -> int {
    switch (src.scalar_type()) {
        case torch::kFloat32: if (v2) launch_argmax2_float(xp, op, rows, cols, tile, useSpr, stream);
                              else launch_argmax_float(xp, op, rows, cols, tile, nd, dim, rsz[0], rsz[1], rsz[2], rsz[3], rsz[4], rsz[5], rsz[6], rsz[7], ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7], stream); break;
        case torch::kFloat16: if (v2) launch_argmax2_half(xp, op, rows, cols, tile, useSpr, stream);
                              else launch_argmax_half(xp, op, rows, cols, tile, nd, dim, rsz[0], rsz[1], rsz[2], rsz[3], rsz[4], rsz[5], rsz[6], rsz[7], ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7], stream); break;
        case torch::kBFloat16: if (v2) launch_argmax2_bf16(xp, op, rows, cols, tile, useSpr, stream);
                              else launch_argmax_bf16(xp, op, rows, cols, tile, nd, dim, rsz[0], rsz[1], rsz[2], rsz[3], rsz[4], rsz[5], rsz[6], rsz[7], ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7], stream); break;
        case torch::kInt32: if (v2) launch_argmax2_i32(xp, op, rows, cols, tile, useSpr, stream);
                            else launch_argmax_i32(xp, op, rows, cols, tile, nd, dim, rsz[0], rsz[1], rsz[2], rsz[3], rsz[4], rsz[5], rsz[6], rsz[7], ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7], stream); break;
        case torch::kInt64: if (v2) launch_argmax2_i64(xp, op, rows, cols, tile, useSpr, stream);
                            else launch_argmax_i64(xp, op, rows, cols, tile, nd, dim, rsz[0], rsz[1], rsz[2], rsz[3], rsz[4], rsz[5], rsz[6], rsz[7], ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7], stream); break;
        default: TORCH_CHECK(false, "arg_max: unsupported dtype");
    }
        return 0;
    });
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> cummin_npu(const torch::Tensor &input, int64_t dim)
{
    const c10::OptionalDeviceGuard guard(input.device());
    auto x = input.contiguous();
    const int64_t nd = x.dim();
    if (dim < 0) dim += nd;
    torch::Tensor src = x;
    const bool moved = (dim != nd - 1);
    const bool ownTr = moved && env_on("KAG_CM_OWNTR", true);
    RowSwapTr cin, cout;
    if (moved && !ownTr) {
        src = x.transpose(dim, nd - 1).contiguous();
        aclrtSynchronizeDevice();
    }
    if (ownTr) {
        cin = rowSwapParams(x, dim);
        src = torch::empty(std::vector<int64_t>(cin.sh, cin.sh + nd), x.options());
        cout = rowSwapParams(src, dim);
    }
    auto vals = torch::empty(x.sizes(), x.options());
    auto idx = torch::empty(x.sizes(), x.options().dtype(torch::kInt64));
    torch::Tensor vs = vals, is = idx;
    if (moved) {
        vs = torch::empty(src.sizes(), x.options());
        is = torch::empty(src.sizes(), x.options().dtype(torch::kInt64));
    }
    const int64_t cols = src.size(-1);
    const int64_t rows = src.numel() / cols;
    if (cols == 0 || rows == 0) return {vals, idx};
    int64_t tile = std::get<0>(row_tile(cols, src.element_size()));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)src.data_ptr();
    auto vp = (GM_ADDR)vs.data_ptr();
    auto ip = (GM_ADDR)is.data_ptr();
    at_npu::native::OpCommand::RunOpApi("Cummin", [=]() -> int {
    if (ownTr) {
#define CMTR_IN(SUF)                                                                                       \
        launch_transpose_##SUF((GM_ADDR)x.data_ptr(), (GM_ADDR)src.data_ptr(), cin.outer, cin.inner,        \
                               cin.isi, cin.nd, cin.st[0], cin.st[1], cin.st[2], cin.st[3], cin.st[4],      \
                               cin.st[5], cin.st[6], cin.st[7], cin.sh[0], cin.sh[1], cin.sh[2], cin.sh[3], \
                               cin.sh[4], cin.sh[5], cin.sh[6], cin.sh[7], cin.tile, stream)
        switch (x.scalar_type()) {
            case torch::kFloat32: CMTR_IN(float); break;
            case torch::kFloat16: CMTR_IN(half); break;
            case torch::kBFloat16: CMTR_IN(bf16); break;
            case torch::kInt32: CMTR_IN(i32); break;
            default: TORCH_CHECK(false, "cummin: unsupported dtype");
        }
#undef CMTR_IN
    }
    switch (src.scalar_type()) {
        case torch::kFloat32: launch_cummin_float(xp, vp, ip, rows, cols, tile, stream); break;
        case torch::kFloat16: launch_cummin_half(xp, vp, ip, rows, cols, tile, stream); break;
        case torch::kBFloat16: launch_cummin_bf16(xp, vp, ip, rows, cols, tile, stream); break;
        case torch::kInt32: launch_cummin_i32(xp, vp, ip, rows, cols, tile, stream); break;
        default: TORCH_CHECK(false, "cummin: unsupported dtype");
    }
    if (ownTr) {
#define CMTR_OUT(SUF)                                                                                      \
        launch_transpose_##SUF((GM_ADDR)vs.data_ptr(), (GM_ADDR)vals.data_ptr(), cout.outer, cout.inner,     \
                               cout.isi, cout.nd, cout.st[0], cout.st[1], cout.st[2], cout.st[3],           \
                               cout.st[4], cout.st[5], cout.st[6], cout.st[7], cout.sh[0], cout.sh[1],     \
                               cout.sh[2], cout.sh[3], cout.sh[4], cout.sh[5], cout.sh[6], cout.sh[7],    \
                               cout.tile, stream)
        switch (src.scalar_type()) {
            case torch::kFloat32: CMTR_OUT(float); break;
            case torch::kFloat16: CMTR_OUT(half); break;
            case torch::kBFloat16: CMTR_OUT(bf16); break;
            case torch::kInt32: CMTR_OUT(i32); break;
            default: break;
        }
#undef CMTR_OUT
        launch_transpose_i64((GM_ADDR)is.data_ptr(), (GM_ADDR)idx.data_ptr(), cout.outer, cout.inner,
                             cout.isi, cout.nd, cout.st[0], cout.st[1], cout.st[2], cout.st[3], cout.st[4],
                             cout.st[5], cout.st[6], cout.st[7], cout.sh[0], cout.sh[1], cout.sh[2],
                             cout.sh[3], cout.sh[4], cout.sh[5], cout.sh[6], cout.sh[7], cout.tile, stream);
    }
        return 0;
    });
    if (moved && !ownTr) {
        aclrtSynchronizeDevice();
        return {vs.transpose(dim, nd - 1).contiguous(), is.transpose(dim, nd - 1).contiguous()};
    }
    return {vals, idx};
}

torch::Tensor cross_entropy_loss_npu(const torch::Tensor &input, const torch::Tensor &target,
                                     c10::string_view reduction, int64_t ignore_index)
{
    const c10::OptionalDeviceGuard guard(input.device());
    auto x = input.contiguous();
    auto t = target.contiguous();
    TORCH_CHECK(x.dim() >= 2, "cross_entropy_loss expects (N, C, ...)");
    RowSwapTr ctr;
    torch::Tensor xT;
    const bool ownTr = (x.dim() > 2) && env_on("KAG_CE_OWNTR", true);
    if (x.dim() > 2 && !ownTr) {
        x = x.transpose(1, x.dim() - 1).contiguous();
        aclrtSynchronizeDevice();
    }
    if (ownTr) {
        ctr = rowSwapParams(x, 1);
        std::vector<int64_t> tshape(ctr.sh, ctr.sh + x.dim());
        xT = torch::empty(tshape, x.options());
        x = xT;   // the kernel reads the transposed scratch; its shape matches
    }
    TORCH_CHECK(!t.is_floating_point(), "cross_entropy_loss: only hard labels supported");
    const int64_t cols = x.size(-1);
    const int64_t rows = x.numel() / cols;
    const std::string red(reduction);
    const bool isNone = (red == "none");
    const bool isMean = (red == "mean");
    auto lossF = torch::empty({rows}, x.options().dtype(torch::kFloat32));
    if (rows == 0) {
        return isNone ? torch::empty(t.sizes(), x.options()) : torch::empty({1}, x.options()).reshape({});
    }
    int64_t tile = std::get<0>(row_tile(cols, x.element_size()));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    torch::Tensor xOrig = input.contiguous();
    auto xp = (GM_ADDR)x.data_ptr();
    auto tp = (GM_ADDR)t.data_ptr();
    auto lp = (GM_ADDR)lossF.data_ptr();
    const int32_t ig = static_cast<int32_t>(ignore_index);
    const bool t64 = (t.scalar_type() == torch::kInt64);
    const int32_t mode = isMean ? 1 : 0;
    // 0-dim NPU tensors hit the aclnn d2d copy path that fails on the runner, so keep the
    // staging tensor at rank 1 and reshape at the very end (metadata only)
    auto scalarF = torch::empty({1}, x.options().dtype(torch::kFloat32));
    auto sp = (GM_ADDR)scalarF.data_ptr();
    // loss/mask outputs: cast fp32 -> x dtype with our own kernel (aclnn Cast is broken on the runner)
    auto lossT = torch::empty({isNone ? rows : 1}, x.options());
    auto lpT = (GM_ADDR)lossT.data_ptr();
    at_npu::native::OpCommand::RunOpApi("CrossEntropyLoss", [=]() -> int {
    if (ownTr) {
#define CETR_CALL(SUF)                                                                                     \
        launch_transpose_##SUF((GM_ADDR)xOrig.data_ptr(), (GM_ADDR)xT.data_ptr(), ctr.outer, ctr.inner,     \
                               ctr.isi, ctr.nd, ctr.st[0], ctr.st[1], ctr.st[2], ctr.st[3], ctr.st[4],      \
                               ctr.st[5], ctr.st[6], ctr.st[7], ctr.sh[0], ctr.sh[1], ctr.sh[2], ctr.sh[3], \
                               ctr.sh[4], ctr.sh[5], ctr.sh[6], ctr.sh[7], ctr.tile, stream)
        switch (xOrig.scalar_type()) {
            case torch::kFloat32: CETR_CALL(float); break;
            case torch::kFloat16: CETR_CALL(half); break;
            case torch::kBFloat16: CETR_CALL(bf16); break;
            default: TORCH_CHECK(false, "cross_entropy_loss: unsupported dtype");
        }
#undef CETR_CALL
    }
    if (x.scalar_type() == torch::kFloat32) {
        if (t64) launch_ce_float_int64_t(xp, tp, lp, rows, cols, tile, ig, stream);
        else TORCH_CHECK(false, "cross_entropy_loss: int32 target not wired");
    } else if (x.scalar_type() == torch::kFloat16) {
        if (t64) launch_ce_half_int64_t(xp, tp, lp, rows, cols, tile, ig, stream);
        else TORCH_CHECK(false, "cross_entropy_loss: int32 target not wired");
    } else {
        if (t64) launch_ce_bf16_int64_t(xp, tp, lp, rows, cols, tile, ig, stream);
        else TORCH_CHECK(false, "cross_entropy_loss: int32 target not wired");
    }
    if (!isNone) {
        launch_ce_reduce_f32(lp, tp, sp, rows, ig, mode, stream);
    }
    {
        const int64_t nCast = isNone ? rows : 1;
        auto cs = isNone ? lp : sp;
        if (x.scalar_type() == torch::kFloat32) launch_row_cast_f32f32(cs, lpT, nCast, stream);
        else if (x.scalar_type() == torch::kFloat16) launch_row_cast_f32f16(cs, lpT, nCast, stream);
        else launch_row_cast_f32bf16(cs, lpT, nCast, stream);
    }
    return 0;
    });
    // return freshly produced tensors instead of copy_ into a preallocated one: the platform
    // runner fails d2d copy_ with copy_d2d_baseformat_opapi
    if (isNone) {
        return lossT.view(t.sizes());
    }
    return lossT.reshape({});
}

torch::Tensor group_norm_npu(const torch::Tensor &x, const torch::Tensor &gamma, const torch::Tensor &beta,
                             int64_t num_groups, double epsilon)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto g = gamma.contiguous();
    auto b = beta.contiguous();
    auto gf = (g.scalar_type() == torch::kFloat32) ? g : g.to(torch::kFloat32).contiguous();
    auto bf = (b.scalar_type() == torch::kFloat32) ? b : b.to(torch::kFloat32).contiguous();
    if (gf.scalar_type() != g.scalar_type() || bf.scalar_type() != b.scalar_type()) {
        aclrtSynchronizeDevice();
    }
    auto y = torch::empty_like(xc);
    const int64_t batch = xc.size(0);
    const int64_t channels = xc.size(1);
    const int64_t spatial = xc.numel() / (batch * channels);
    TORCH_CHECK(channels % num_groups == 0, "group_norm: channels % groups != 0");
    int64_t tile = std::get<0>(row_tile(spatial, xc.element_size()));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)xc.data_ptr();
    auto gp = (GM_ADDR)gf.data_ptr();
    auto bp = (GM_ADDR)bf.data_ptr();
    auto yp = (GM_ADDR)y.data_ptr();
    const float eps = static_cast<float>(epsilon);
    at_npu::native::OpCommand::RunOpApi("GroupNorm", [=]() -> int {
    switch (xc.scalar_type()) {
        case torch::kFloat32: launch_group_norm_float(xp, gp, bp, yp, batch, channels, num_groups, spatial, tile, eps, stream); break;
        case torch::kFloat16: launch_group_norm_half(xp, gp, bp, yp, batch, channels, num_groups, spatial, tile, eps, stream); break;
        case torch::kBFloat16: launch_group_norm_bf16(xp, gp, bp, yp, batch, channels, num_groups, spatial, tile, eps, stream); break;
        default: TORCH_CHECK(false, "group_norm: unsupported dtype");
    }
        return 0;
    });
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("dynamic_quant", dynamic_quant_npu);
    m.impl("rms_norm", rms_norm_npu);
    m.impl("arg_max", arg_max_npu);
    m.impl("cummin", cummin_npu);
    m.impl("cross_entropy_loss", cross_entropy_loss_npu);
    m.impl("group_norm", group_norm_npu);
}

}  // namespace cann_bench
