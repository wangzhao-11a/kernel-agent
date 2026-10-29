#ifndef SWIGLU_LAUNCH_H
#define SWIGLU_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

int64_t swiglu_core_num();
std::tuple<int64_t, int64_t, uint32_t> calc_swiglu_tiling_params(int64_t outer, int64_t half, int64_t elemBytes);
std::tuple<int64_t, int64_t, uint32_t> calc_swiglu_tiny_tiling_params(int64_t outer, int64_t half, int64_t elemBytes);

extern "C" {
void launch_swiglu_kernel_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half, int64_t rowsPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream);
void launch_swiglu_kernel_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half, int64_t rowsPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream);
void launch_swiglu_simt_float(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t half, int64_t numBlocks, void *stream);
void launch_swiglu_simt_half(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t half, int64_t numBlocks, void *stream);
void launch_swiglu_simt_bf16(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t half, int64_t numBlocks, void *stream);
void launch_swiglu_kernel_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half, int64_t rowsPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream);
void launch_swiglu_tiny_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half, int64_t rowsPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream);
void launch_swiglu_tiny_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half, int64_t rowsPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream);
void launch_swiglu_tiny_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half, int64_t rowsPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream);
}

#endif // SWIGLU_LAUNCH_H
