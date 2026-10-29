#ifndef LAYOUT_OPS_LAUNCH_H
#define LAYOUT_OPS_LAUNCH_H
#include <cstdint>
#include <tuple>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif
std::tuple<int64_t, int64_t, int64_t> calc_layout_tiling(int64_t inner);
extern "C" {
void launch_transpose2d_hw_f16(GM_ADDR x, GM_ADDR y, int64_t nBatch, int64_t H, int64_t W, void *stream);
void launch_transpose2d_hw_bf16(GM_ADDR x, GM_ADDR y, int64_t nBatch, int64_t H, int64_t W, void *stream);
void launch_transpose2d_hw_i16(GM_ADDR x, GM_ADDR y, int64_t nBatch, int64_t H, int64_t W, void *stream);
void launch_transpose2d_tiled_f32(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose2d_tiled_f16(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose2d_tiled_bf16(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose2d_tiled_i8(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose2d_tiled_i16(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose2d_tiled_i32(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose2d_tiled_i64(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream);
void launch_transpose_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                           int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                           int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                           int64_t tile, void *stream);
void launch_transpose_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                            int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                            int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                            int64_t tile, void *stream);
void launch_transpose_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                           int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                           int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                           int64_t tile, void *stream);
void launch_transpose_i8(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                         int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                         int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                         int64_t tile, void *stream);
void launch_transpose_i16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                          int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                          int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                          int64_t tile, void *stream);
void launch_transpose_i32(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                          int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                          int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                          int64_t tile, void *stream);
void launch_transpose_i64(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner, int64_t ndim,
                          int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,
                          int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                          int64_t tile, void *stream);
void launch_slice_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart, int64_t lastStep,
                       int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0, int64_t s1, int64_t s2,
                       int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t t0, int64_t t1,
                       int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0,
                       int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                       int64_t tile, void *stream);
void launch_slice_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart, int64_t lastStep,
                        int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0, int64_t s1, int64_t s2,
                        int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t t0, int64_t t1,
                        int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0,
                        int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                        int64_t tile, void *stream);
void launch_slice_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart, int64_t lastStep,
                       int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0, int64_t s1, int64_t s2,
                       int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t t0, int64_t t1,
                       int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0,
                       int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                       int64_t tile, void *stream);
void launch_slice_i8(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart, int64_t lastStep,
                     int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0, int64_t s1, int64_t s2,
                     int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t t0, int64_t t1,
                     int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0,
                     int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                     int64_t tile, void *stream);
void launch_slice_i32(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart, int64_t lastStep,
                      int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0, int64_t s1, int64_t s2,
                      int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t t0, int64_t t1,
                      int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0,
                      int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                      int64_t tile, void *stream);
void launch_slice_i64(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart, int64_t lastStep,
                      int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0, int64_t s1, int64_t s2,
                      int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t t0, int64_t t1,
                      int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0,
                      int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                      int64_t tile, void *stream);
void launch_adapt_pool_bf16(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                            int64_t OD, int64_t OH, int64_t OW, int64_t tile, void *stream);
void launch_adapt_pool_float(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                             int64_t OD, int64_t OH, int64_t OW, int64_t tile, void *stream);
void launch_adapt_pool_half(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                            int64_t OD, int64_t OH, int64_t OW, int64_t tile, void *stream);
void launch_addcdiv_float(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, int64_t total, int64_t blocks,
                          int64_t blockLength, int64_t tile, float scalar, void *stream);
void launch_addcdiv_half(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, int64_t total, int64_t blocks,
                         int64_t blockLength, int64_t tile, float scalar, void *stream);
void launch_norm_partial_float(GM_ADDR x, GM_ADDR ws, int64_t total, int64_t blocks, int64_t blockLength,
                               int64_t tile, float p, int32_t mode, void *stream);
void launch_norm_partial_half(GM_ADDR x, GM_ADDR ws, int64_t total, int64_t blocks, int64_t blockLength,
                              int64_t tile, float p, int32_t mode, void *stream);
void launch_norm_finalize(GM_ADDR ws, GM_ADDR out, int64_t blocks, float p, int32_t mode, int32_t outIsF32,
                          void *stream);
}
#endif
