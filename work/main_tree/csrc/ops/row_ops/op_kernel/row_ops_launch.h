#ifndef ROW_OPS_LAUNCH_H
#define ROW_OPS_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_row_tiling(int64_t cols, int64_t elemBytes);
std::tuple<int64_t, int64_t, int64_t> calc_dq_tiling(int64_t cols, int64_t elemBytes);
std::tuple<int64_t, int64_t, int64_t> calc_row_tiling2(int64_t cols, int64_t elemBytes);
std::tuple<int64_t, int64_t, int64_t> calc_row_tiling3(int64_t cols, int64_t elemBytes);
bool rms_full_row_fits(int64_t cols, int64_t elemBytes);
std::tuple<int64_t, int64_t, int64_t> calc_argmax_tiling(int64_t cols, int64_t elemBytes);

extern "C" {
void launch_row_cast_f32i32(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream);
void launch_row_cast_f32i64(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream);
void launch_row_cast_f32f32(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream);
void launch_row_cast_f32f16(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream);
void launch_row_cast_f32bf16(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream);
void launch_argmax_axis_float(GM_ADDR x, GM_ADDR idx, int64_t O, int64_t K, int64_t S, int64_t tile, int64_t dbg, void *stream);
void launch_argmax_axis_half(GM_ADDR x, GM_ADDR idx, int64_t O, int64_t K, int64_t S, int64_t tile, int64_t dbg, void *stream);
void launch_argmax_axis_bf16(GM_ADDR x, GM_ADDR idx, int64_t O, int64_t K, int64_t S, int64_t tile, int64_t dbg, void *stream);
void launch_argmax_axis_i32(GM_ADDR x, GM_ADDR idx, int64_t O, int64_t K, int64_t S, int64_t tile, int64_t dbg, void *stream);
void launch_rms_norm2_float(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm2_bf16(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm2_half(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm3_float(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, int64_t useBcast, void *stream);
void launch_rms_norm3_bf16(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, int64_t useBcast, void *stream);
void launch_rms_norm3_half(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, int64_t useBcast, void *stream);
void launch_rms_norm4_float(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm4_bf16(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm4_half(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_argmax2_float(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int32_t useSpr, void *stream);
void launch_argmax2_bf16(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int32_t useSpr, void *stream);
void launch_argmax2_half(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int32_t useSpr, void *stream);
void launch_argmax2_i32(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int32_t useSpr, void *stream);
void launch_argmax2_i64(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int32_t useSpr, void *stream);
void launch_dynamic_quant_float(GM_ADDR x, GM_ADDR y, GM_ADDR s, int64_t rows, int64_t cols, int64_t tile, int32_t flags, void *stream);
void launch_dynamic_quant_bf16(GM_ADDR x, GM_ADDR y, GM_ADDR s, int64_t rows, int64_t cols, int64_t tile, int32_t flags, void *stream);
void launch_dynamic_quant_half(GM_ADDR x, GM_ADDR y, GM_ADDR s, int64_t rows, int64_t cols, int64_t tile, int32_t flags, void *stream);
void launch_rms_norm_float(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm_bf16(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_rms_norm_half(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile, float eps, void *stream);
void launch_argmax_float(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int64_t nd,
                          int64_t dim, int64_t r0, int64_t r1, int64_t r2, int64_t r3, int64_t r4,
                          int64_t r5, int64_t r6, int64_t r7, int64_t o0, int64_t o1, int64_t o2,
                          int64_t o3, int64_t o4, int64_t o5, int64_t o6, int64_t o7, void *stream);
void launch_argmax_bf16(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int64_t nd,
                          int64_t dim, int64_t r0, int64_t r1, int64_t r2, int64_t r3, int64_t r4,
                          int64_t r5, int64_t r6, int64_t r7, int64_t o0, int64_t o1, int64_t o2,
                          int64_t o3, int64_t o4, int64_t o5, int64_t o6, int64_t o7, void *stream);
void launch_argmax_half(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int64_t nd,
                          int64_t dim, int64_t r0, int64_t r1, int64_t r2, int64_t r3, int64_t r4,
                          int64_t r5, int64_t r6, int64_t r7, int64_t o0, int64_t o1, int64_t o2,
                          int64_t o3, int64_t o4, int64_t o5, int64_t o6, int64_t o7, void *stream);
void launch_argmax_i32(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int64_t nd,
                          int64_t dim, int64_t r0, int64_t r1, int64_t r2, int64_t r3, int64_t r4,
                          int64_t r5, int64_t r6, int64_t r7, int64_t o0, int64_t o1, int64_t o2,
                          int64_t o3, int64_t o4, int64_t o5, int64_t o6, int64_t o7, void *stream);
void launch_argmax_i64(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int64_t nd,
                          int64_t dim, int64_t r0, int64_t r1, int64_t r2, int64_t r3, int64_t r4,
                          int64_t r5, int64_t r6, int64_t r7, int64_t o0, int64_t o1, int64_t o2,
                          int64_t o3, int64_t o4, int64_t o5, int64_t o6, int64_t o7, void *stream);
void launch_cummin_float(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t cols, int64_t tile, void *stream);
void launch_cummin_bf16(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t cols, int64_t tile, void *stream);
void launch_cummin_half(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t cols, int64_t tile, void *stream);
void launch_cummin_i32(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t cols, int64_t tile, void *stream);
void launch_ce_float_int64_t(GM_ADDR x, GM_ADDR t, GM_ADDR l, int64_t rows, int64_t cols, int64_t tile, int32_t ignoreIndex, void *stream);
void launch_ce_bf16_int64_t(GM_ADDR x, GM_ADDR t, GM_ADDR l, int64_t rows, int64_t cols, int64_t tile, int32_t ignoreIndex, void *stream);
void launch_ce_half_int64_t(GM_ADDR x, GM_ADDR t, GM_ADDR l, int64_t rows, int64_t cols, int64_t tile, int32_t ignoreIndex, void *stream);
void launch_ce_reduce_f32(GM_ADDR loss, GM_ADDR target, GM_ADDR out, int64_t rows, int32_t ignoreIndex,
                          int32_t reduceMode, void *stream);
void launch_group_norm_bf16(GM_ADDR x, GM_ADDR g, GM_ADDR b, GM_ADDR y, int64_t batch, int64_t channels,
                            int64_t groups, int64_t spatial, int64_t tile, float eps, void *stream);
void launch_group_norm_float(GM_ADDR x, GM_ADDR g, GM_ADDR b, GM_ADDR y, int64_t batch, int64_t channels,
                             int64_t groups, int64_t spatial, int64_t tile, float eps, void *stream);
void launch_group_norm_half(GM_ADDR x, GM_ADDR g, GM_ADDR b, GM_ADDR y, int64_t batch, int64_t channels,
                            int64_t groups, int64_t spatial, int64_t tile, float eps, void *stream);
}

#endif  // ROW_OPS_LAUNCH_H
