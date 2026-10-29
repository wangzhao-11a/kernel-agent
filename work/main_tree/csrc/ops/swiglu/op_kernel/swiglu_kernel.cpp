/**
 * SwiGlu (direct-launch, AscendC): out = silu(x0) * x1 with x0, x1 = chunk(2, dim).
 *
 * Flat semantics: with M = shape[dim], inner = prod(shape[dim+1:]), outer = prod(shape[:dim]) and
 * half = (M/2)*inner, out[r*half + c] = silu(X[r*2*half + c]) * X[r*2*half + half + c].
 * Each row of the output is one contiguous `half`-element block; the two input halves sit in the
 * same row at a `half`-element distance, so a strided DataCopyPad (blockCount=rows, blockLen=half,
 * srcStride=half) loads both operands with no gather, and the store is fully contiguous.
 */
#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#ifndef SWIGLU_HAS_SIMT
#define SWIGLU_HAS_SIMT 0
#endif
#ifndef SWIGLU_HAS_ADV_API
#define SWIGLU_HAS_ADV_API 0
#endif

#if SWIGLU_HAS_ADV_API
#include "adv_api/activation/swiglu.h"
#include "adv_api/activation/silu.h"
#endif

constexpr static int64_t PIPELINE_DEPTH = 2;
constexpr static int64_t BUFFER_NUM = 3;
constexpr static int64_t TINY_MAX_ROWS = 2048;

// Tiny-row path (910B, no SIMT): half*sizeof(T) < 32B makes the strided tile path
// degenerate to one element per DataCopyPad. Load rowsPerTile rows as one contiguous
// block, split even/odd columns into separate T buffers with Gather, compute, then
// store the contiguous output block.
template <typename T>
__global__ __aicore__ void swiglu_tiny_kernel(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half,
                                              int64_t tilesPerBlock, uint32_t tileBytes)
{
    AscendC::TPipe pipe;
    int64_t elemsPerTile = static_cast<int64_t>(tileBytes) / static_cast<int64_t>(sizeof(T));
    int64_t rowsPerTile = elemsPerTile / (2 * half);
    int64_t capRows = 4096 / half;
    if (capRows < 1) {
        capRows = 1;
    }
    if (capRows > TINY_MAX_ROWS) {
        capRows = TINY_MAX_ROWS;
    }
    if (rowsPerTile > capRows) {
        rowsPerTile = capRows;
    }
    if (rowsPerTile < 1) {
        rowsPerTile = 1;
    }
    // Gather/vector repeat boundaries: 128 elements for 16-bit, 64 for 32-bit.
    // A tail of 4095 elements can touch an unaligned UB address at the end.
    int64_t repeatElems = (sizeof(T) == 2) ? 128 : 64;
    while (rowsPerTile > 1 && (rowsPerTile * half) % repeatElems != 0) {
        --rowsPerTile;
    }

    int64_t inElems = rowsPerTile * 2 * half;
    int64_t outElems = rowsPerTile * half;
    uint32_t inBytes = static_cast<uint32_t>(inElems * sizeof(T));
    uint32_t outBytes = static_cast<uint32_t>(outElems * sizeof(T));
    uint32_t scratchBytes = static_cast<uint32_t>(outElems * sizeof(float));
    uint32_t offBytes = static_cast<uint32_t>(outElems * sizeof(uint32_t));

    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inX;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 2> xQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outY;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 2> tmpQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offBuf;
    pipe.InitBuffer(inX, PIPELINE_DEPTH, inBytes);
    pipe.InitBuffer(xQ, 2, outBytes);
    pipe.InitBuffer(outY, PIPELINE_DEPTH, outBytes);
    pipe.InitBuffer(tmpQ, 2, scratchBytes);
    pipe.InitBuffer(offBuf, 2 * offBytes);

    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalTiles = (outer + rowsPerTile - 1) / rowsPerTile;
    int64_t tStart = blockIdx * tilesPerBlock;
    int64_t tEnd = tStart + tilesPerBlock;
    if (tEnd > totalTiles) {
        tEnd = totalTiles;
    }
    if (tStart >= tEnd) {
        return;
    }

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // One int32 offset per gathered element. For 16-bit operands with even half, two
    // adjacent values form one uint32 word, so Gather moves half as many elements.
    auto offT = offBuf.Get<int32_t>();
    const int64_t maxN = rowsPerTile * half;
    const bool packed = (sizeof(T) == 2) && (half > 1) && ((half & 1) == 0);
    const int64_t step = packed ? (half / 2) : half;
    const int64_t nBase = packed ? (maxN / 2) : maxN;
    const int64_t elemBytes = packed ? 4 : static_cast<int64_t>(sizeof(T));
    AscendC::LocalTensor<uint32_t> off0;
    AscendC::LocalTensor<uint32_t> off1;
    auto offA = offT[0];
    auto offB = offT[static_cast<int32_t>(nBase)];
    int64_t logStep = 0;
    while ((1LL << logStep) < step) {
        ++logStep;
    }
    if ((step & (step - 1)) == 0) {
        // off0 = (i + (i / step) * step) * elemBytes; off1 = off0 + step * elemBytes.
        // i/step is a right shift because step is a power of two; step==1 needs no shift.
        int64_t shift = (step == 1) ? 0 : logStep;
        AscendC::CreateVecIndex(offA, 0, static_cast<uint32_t>(nBase));
        if (shift != 0) {
            AscendC::ShiftRight(offA, offA, static_cast<int32_t>(shift), static_cast<int32_t>(nBase));
            AscendC::Muls(offA, offA, static_cast<int32_t>(step * elemBytes), static_cast<int32_t>(nBase));
        } else {
            AscendC::Muls(offA, offA, static_cast<int32_t>(step * elemBytes), static_cast<int32_t>(nBase));
        }
        AscendC::CreateVecIndex(offB, 0, static_cast<uint32_t>(nBase));
        AscendC::Muls(offB, offB, static_cast<int32_t>(elemBytes), static_cast<int32_t>(nBase));
        AscendC::Add(offB, offB, offA, static_cast<int32_t>(nBase));
        AscendC::Adds(offA, offB, static_cast<int32_t>(step * elemBytes), static_cast<int32_t>(nBase));
        off0 = offB.template ReinterpretCast<uint32_t>();
        off1 = offA.template ReinterpretCast<uint32_t>();
    } else {
        // Arbitrary step: q = floor(i / step) through float. i <= 4096, so 0.001 is far
        // below the 1/step fractional gap and far above the float product error.
        auto fq = tmpQ.AllocTensor<float>();
        AscendC::CreateVecIndex(offA, 0, static_cast<uint32_t>(nBase));
        AscendC::Cast(fq, offA, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(nBase));
        AscendC::Muls(fq, fq, 1.0f / static_cast<float>(step), static_cast<int32_t>(nBase));
        AscendC::Adds(fq, fq, 0.001f, static_cast<int32_t>(nBase));
        AscendC::Cast(offA, fq, AscendC::RoundMode::CAST_FLOOR, static_cast<uint32_t>(nBase));
        AscendC::Muls(offA, offA, static_cast<int32_t>(step * elemBytes), static_cast<int32_t>(nBase));
        AscendC::CreateVecIndex(offB, 0, static_cast<uint32_t>(nBase));
        AscendC::Muls(offB, offB, static_cast<int32_t>(elemBytes), static_cast<int32_t>(nBase));
        AscendC::Add(offB, offB, offA, static_cast<int32_t>(nBase));
        AscendC::Adds(offA, offB, static_cast<int32_t>(step * elemBytes), static_cast<int32_t>(nBase));
        AscendC::PipeBarrier<PIPE_V>();
        tmpQ.FreeTensor(fq);
        off0 = offB.template ReinterpretCast<uint32_t>();
        off1 = offA.template ReinterpretCast<uint32_t>();
    }

    for (int64_t t = tStart; t < tEnd; ++t) {
        int64_t r0 = t * rowsPerTile;
        int64_t rows = (rowsPerTile < outer - r0) ? rowsPerTile : (outer - r0);
        int64_t n = rows * half;
        uint32_t cnt = static_cast<uint32_t>(n);
        uint32_t gcnt = static_cast<uint32_t>(packed ? (n / 2) : n);
        uint32_t nInBytes = static_cast<uint32_t>(rows * 2 * half * sizeof(T));
        uint32_t nOutBytes = static_cast<uint32_t>(n * sizeof(T));

        auto xLocal = inX.AllocTensor<T>();
        AscendC::DataCopyExtParams inParams{1, nInBytes, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[r0 * 2 * half], inParams, padParams);
        inX.EnQue(xLocal);
        xLocal = inX.DeQue<T>();

        // Distinct dst buffers: in-place Gather (or sharing x0/x1) corrupts the source.
        auto x0 = xQ.AllocTensor<T>();
        auto x1 = xQ.AllocTensor<T>();
        if (packed) {
            // Two adjacent T values = one uint32 word; the dst tensors keep the same
            // row-major T layout, so the fp32 compute below is unchanged.
            auto x0w = x0.template ReinterpretCast<uint32_t>();
            auto x1w = x1.template ReinterpretCast<uint32_t>();
            auto xlw = xLocal.template ReinterpretCast<uint32_t>();
            AscendC::Gather(x0w, xlw, off0, 0, gcnt);
            AscendC::Gather(x1w, xlw, off1, 0, gcnt);
        } else {
            AscendC::Gather(x0, xLocal, off0, 0, gcnt);
            AscendC::Gather(x1, xLocal, off1, 0, gcnt);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        inX.FreeTensor(xLocal);

        auto yLocal = outY.AllocTensor<T>();
        auto s0 = tmpQ.AllocTensor<float>();
        auto s1 = tmpQ.AllocTensor<float>();
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Muls(s1, x0, static_cast<float>(-1), cnt);
            AscendC::Exp(s0, s1, cnt);
            AscendC::Adds(s1, s0, static_cast<float>(1), cnt);
            AscendC::Div(s0, x0, s1, cnt);
            AscendC::Mul(yLocal, s0, x1, cnt);
        } else {
            bool usedSilu = false;
#if SWIGLU_HAS_ADV_API
            if ((n * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
                AscendC::Cast(s0, x0, AscendC::RoundMode::CAST_NONE, cnt);
                AscendC::Silu<float>(s1, s0, cnt);
                AscendC::Cast(s0, x1, AscendC::RoundMode::CAST_NONE, cnt);
                AscendC::Mul(s1, s1, s0, cnt);
                AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, cnt);
                usedSilu = true;
            }
#endif
            if (!usedSilu) {
                AscendC::Cast(s0, x0, AscendC::RoundMode::CAST_NONE, cnt);
                AscendC::Muls(s1, s0, static_cast<float>(-1), cnt);
                AscendC::Exp(s1, s1, cnt);
                AscendC::Adds(s1, s1, static_cast<float>(1), cnt);
                AscendC::Div(s1, s0, s1, cnt);
                AscendC::Cast(s0, x1, AscendC::RoundMode::CAST_NONE, cnt);
                AscendC::Mul(s1, s1, s0, cnt);
                AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, cnt);
            }
        }
        tmpQ.FreeTensor(s1);
        tmpQ.FreeTensor(s0);
        xQ.FreeTensor(x1);
        xQ.FreeTensor(x0);

        outY.EnQue(yLocal);
        yLocal = outY.DeQue<T>();
        AscendC::DataCopyExtParams outParams{1, nOutBytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[r0 * half], yLocal, outParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        outY.FreeTensor(yLocal);
    }
}

template <typename T>
__global__ __aicore__ void swiglu_kernel(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half,
                                         int64_t tilesPerBlock, uint32_t tileBytes)
{
    AscendC::TPipe pipe;
    bool aligned = ((half * static_cast<int64_t>(sizeof(T))) % 32 == 0);   // 32B block rule
    // tileBytes / elemsPerTile were computed by the host tiling helper (per-dtype budget)
    int64_t elemsPerTile = static_cast<int64_t>(tileBytes) / static_cast<int64_t>(sizeof(T));
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inX0;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inX1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outY;
    pipe.InitBuffer(inX0, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(inX1, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(outY, PIPELINE_DEPTH, tileBytes);
    AscendC::TQue<AscendC::QuePosition::VECCALC, 2> tmpQ;
    pipe.InitBuffer(tmpQ, 2, elemsPerTile * static_cast<int64_t>(sizeof(float)));   // 2 fp32 scratch slices
    int64_t blockIdx = AscendC::GetBlockIdx();
    // tile = R rows x W columns. W keeps copies >= 2KB whenever the row is long enough, R packs the
    // remaining tile budget with whole rows so short rows still get multi-row strided copies.
    constexpr static int64_t TILE_ELEMS = 2048;
    int64_t W = (half < elemsPerTile) ? half : elemsPerTile;   // use the full per-dtype UB capacity
    // DataCopyPad with blockCount > 1 needs the per-block byte length to be a multiple of 32 bytes;
    // otherwise rows must be copied one at a time (blockCount = 1), which is always correct.
    int64_t R = ((half * static_cast<int64_t>(sizeof(T))) % 32 == 0) ? (elemsPerTile / W) : 1;
    if (R < 1) {
        R = 1;
    }
    int64_t nCols = (half + W - 1) / W;
    const bool oneCol = (nCols == 1);
    int64_t nRowGroups = (outer + R - 1) / R;
    int64_t totalTiles = nRowGroups * nCols;
    int64_t tStart = blockIdx * tilesPerBlock;
    int64_t tEnd = tStart + tilesPerBlock;
    if (tEnd > totalTiles) {
        tEnd = totalTiles;
    }
    if (tStart >= tEnd) {
        return;
    }

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t t = tStart; t < tEnd; ++t) {
        // nCols == 1 is the common case; avoid a scalar division per tile.
        int64_t rowGroup = oneCol ? t : t / nCols;
        int64_t colBlk = oneCol ? 0 : t - rowGroup * nCols;
        int64_t r0 = rowGroup * R;
        int64_t rows = (R < outer - r0) ? R : (outer - r0);
        int64_t c0 = colBlk * W;
        int64_t w = (W < half - c0) ? W : (half - c0);
        int64_t n = rows * w;
        uint32_t nBytes = static_cast<uint32_t>(n * sizeof(T));
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(w * sizeof(T)),
            static_cast<uint32_t>((2 * half - w) * sizeof(T)), 0, 0};
        auto x0Local = inX0.AllocTensor<T>();
        auto x1Local = inX1.AllocTensor<T>();
        AscendC::DataCopyPad(x0Local, xGm[r0 * 2 * half + c0], params, padParams);
        AscendC::DataCopyPad(x1Local, xGm[r0 * 2 * half + half + c0], params, padParams);
        inX0.EnQue(x0Local);
        inX1.EnQue(x1Local);
        x0Local = inX0.DeQue<T>();
        x1Local = inX1.DeQue<T>();
        auto yLocal = outY.AllocTensor<T>();
        bool usedAdvApi = false;
#if SWIGLU_HAS_ADV_API
        // KB: AscendC::SwiGLU is correct only on 32B-aligned slices on dav-2201.
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            if ((n * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
                AscendC::SwiGLU<T>(yLocal, x1Local, x0Local, 1.0f, static_cast<uint32_t>(n));
                usedAdvApi = true;
            }
        }
#endif
        if (!usedAdvApi) {
            auto s0 = tmpQ.AllocTensor<float>();
            auto s1 = tmpQ.AllocTensor<float>();
            if constexpr (sizeof(T) == sizeof(float)) {
                AscendC::Muls(s1, x0Local, static_cast<float>(-1), static_cast<uint32_t>(n));
                AscendC::Exp(s0, s1, static_cast<uint32_t>(n));
                AscendC::Adds(s1, s0, static_cast<float>(1), static_cast<uint32_t>(n));
                AscendC::Div(s0, x0Local, s1, static_cast<uint32_t>(n));
                AscendC::Mul(yLocal, s0, x1Local, static_cast<uint32_t>(n));
            } else {
                bool usedSilu = false;
#if SWIGLU_HAS_ADV_API
                // bf16 has no advanced SwiGLU overload; fuse the SiLU chain in fp32 instead.
                if ((n * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
                    AscendC::Cast(s0, x0Local, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Silu<float>(s1, s0, static_cast<uint32_t>(n));
                    AscendC::Cast(s0, x1Local, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Mul(s1, s1, s0, static_cast<uint32_t>(n));
                    AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
                    usedSilu = true;
                }
#endif
                if (!usedSilu) {
                    // 2 fp32 scratch slices: s1 carries the silu chain in place, s0 is recycled for x1.
                    AscendC::Cast(s0, x0Local, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Muls(s1, s0, static_cast<float>(-1), static_cast<uint32_t>(n));
                    AscendC::Exp(s1, s1, static_cast<uint32_t>(n));
                    AscendC::Adds(s1, s1, static_cast<float>(1), static_cast<uint32_t>(n));
                    AscendC::Div(s1, s0, s1, static_cast<uint32_t>(n));
                    AscendC::Cast(s0, x1Local, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Mul(s1, s1, s0, static_cast<uint32_t>(n));
                    AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
                }
            }
            tmpQ.FreeTensor(s1);
            tmpQ.FreeTensor(s0);
        }
        outY.EnQue(yLocal);
        inX0.FreeTensor(x0Local);
        inX1.FreeTensor(x1Local);
        yLocal = outY.DeQue<T>();
        AscendC::DataCopyExtParams outParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(w * sizeof(T)), 0,
            static_cast<uint32_t>((half - w) * sizeof(T)), 0};
        AscendC::DataCopyPad(yGm[r0 * half + c0], yLocal, outParams);
        outY.FreeTensor(yLocal);
    }
}


// Non-aligned rows (half*sizeof(T) >= 32B but not a multiple of 32B): pad each row to
// halfA elements so a whole tile of R rows can be copied with blockCount=R and blockLen
// a 32B multiple. The vector work then runs on the contiguous padded tile; only the
// output stores stay per-row (the GM rows are 511 elements wide, not 512).
template <typename T>
__global__ __aicore__ void swiglu_unaligned_kernel(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half,
                                                   int64_t R, int64_t tilesPerBlock)
{
    AscendC::TPipe pipe;
    const int64_t es = static_cast<int64_t>(sizeof(T));
    const int64_t halfA = ((half * es + 31) / 32) * 32 / es;
    const int64_t tileElems = R * halfA;
    const uint32_t tileBytes = static_cast<uint32_t>(((tileElems * es + 31) / 32) * 32);
    const uint32_t fBytes = static_cast<uint32_t>(((tileElems * 4 + 31) / 32) * 32);

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inX0, inX1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outY;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 2> tmpQ;
    pipe.InitBuffer(inX0, 1, tileBytes);
    pipe.InitBuffer(inX1, 1, tileBytes);
    pipe.InitBuffer(outY, 1, tileBytes);
    pipe.InitBuffer(tmpQ, 2, fBytes);

    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalTiles = (outer + R - 1) / R;
    int64_t tStart = blockIdx * tilesPerBlock;
    int64_t tEnd = tStart + tilesPerBlock;
    if (tEnd > totalTiles) tEnd = totalTiles;
    if (tStart >= tEnd) return;

    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    const uint32_t rowBytes = static_cast<uint32_t>(halfA * es);
    const uint32_t gapBytes = static_cast<uint32_t>((2 * half - halfA) * es);
    const uint32_t outBytes = static_cast<uint32_t>(half * es);

    for (int64_t t = tStart; t < tEnd; ++t) {
        int64_t r0 = t * R;
        int64_t rh = (R < outer - r0) ? R : (outer - r0);
        int32_t n = static_cast<int32_t>(rh * halfA);

        auto x0 = inX0.AllocTensor<T>();
        auto x1 = inX1.AllocTensor<T>();
        AscendC::DataCopyPad(x0, xGm[r0 * 2 * half],
                             AscendC::DataCopyExtParams{static_cast<uint16_t>(rh), rowBytes, gapBytes, 0, 0},
                             padParams);
        bool lastTile = (r0 + rh == outer);
        if (lastTile) {
            // The last x1 row would read one element past the tensor; copy it exactly.
            if (rh > 1) {
                AscendC::DataCopyPad(x1, xGm[r0 * 2 * half + half],
                                     AscendC::DataCopyExtParams{static_cast<uint16_t>(rh - 1), rowBytes, gapBytes, 0, 0},
                                     padParams);
            }
            AscendC::DataCopyPad(x1[(rh - 1) * halfA], xGm[(r0 + rh - 1) * 2 * half + half],
                                 AscendC::DataCopyExtParams{1, outBytes, 0, 0, 0}, padParams);
        } else {
            AscendC::DataCopyPad(x1, xGm[r0 * 2 * half + half],
                                 AscendC::DataCopyExtParams{static_cast<uint16_t>(rh), rowBytes, gapBytes, 0, 0},
                                 padParams);
        }
        inX0.EnQue(x0);
        inX1.EnQue(x1);
        x0 = inX0.DeQue<T>();
        x1 = inX1.DeQue<T>();

        auto yLocal = outY.AllocTensor<T>();
        bool usedAdvApi = false;
#if SWIGLU_HAS_ADV_API
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            AscendC::SwiGLU<T>(yLocal, x1, x0, 1.0f, static_cast<uint32_t>(n));
            usedAdvApi = true;
        }
#endif
        if (!usedAdvApi) {
            auto s0 = tmpQ.AllocTensor<float>();
            auto s1 = tmpQ.AllocTensor<float>();
            if constexpr (sizeof(T) == sizeof(float)) {
                AscendC::Muls(s1, x0, static_cast<float>(-1), static_cast<uint32_t>(n));
                AscendC::Exp(s0, s1, static_cast<uint32_t>(n));
                AscendC::Adds(s1, s0, static_cast<float>(1), static_cast<uint32_t>(n));
                AscendC::Div(s0, x0, s1, static_cast<uint32_t>(n));
                AscendC::Mul(yLocal, s0, x1, static_cast<uint32_t>(n));
            } else {
                bool usedSilu = false;
#if SWIGLU_HAS_ADV_API
                AscendC::Cast(s0, x0, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                AscendC::Silu<float>(s1, s0, static_cast<uint32_t>(n));
                AscendC::Cast(s0, x1, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                AscendC::Mul(s1, s1, s0, static_cast<uint32_t>(n));
                AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
                usedSilu = true;
#endif
                if (!usedSilu) {
                    AscendC::Cast(s0, x0, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Muls(s1, s0, static_cast<float>(-1), static_cast<uint32_t>(n));
                    AscendC::Exp(s1, s1, static_cast<uint32_t>(n));
                    AscendC::Adds(s1, s1, static_cast<float>(1), static_cast<uint32_t>(n));
                    AscendC::Div(s1, s0, s1, static_cast<uint32_t>(n));
                    AscendC::Cast(s0, x1, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Mul(s1, s1, s0, static_cast<uint32_t>(n));
                    AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
                }
            }
            tmpQ.FreeTensor(s1);
            tmpQ.FreeTensor(s0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        inX0.FreeTensor(x0);
        inX1.FreeTensor(x1);

        outY.EnQue(yLocal);
        yLocal = outY.DeQue<T>();
        for (int64_t i = 0; i < rh; ++i) {
            AscendC::DataCopyPad(yGm[(r0 + i) * half], yLocal[i * halfA],
                                 AscendC::DataCopyExtParams{1, outBytes, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        outY.FreeTensor(yLocal);
    }
}

#if SWIGLU_HAS_SIMT
// ---------------------------------------------------------------------------
// SIMT path (non-aligned rows): one thread per output element, GM random access —
// no UB block-copy constraints at all. Idiom copied from the vendor SIMT kernels
// (opp/.../ascendc/*/arch35/*_simt.h).
// ---------------------------------------------------------------------------
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(2048) inline void SwigluSimt(GM_ADDR x, GM_ADDR y,
        uint32_t totalOut, uint32_t half, uint32_t m0, uint32_t shift0, uint32_t blocks, uint32_t blk)
{
    __gm__ T* xg = reinterpret_cast<__gm__ T*>(x);
    __gm__ T* yg = reinterpret_cast<__gm__ T*>(y);
    // block-aware start + global stride (all AIV cores share the range; the vendor idiom)
    uint32_t start = blk * static_cast<uint32_t>(AscendC::Simt::GetThreadNum())
                   + static_cast<uint32_t>(AscendC::Simt::GetThreadIdx());
    uint32_t stride = blocks * static_cast<uint32_t>(AscendC::Simt::GetThreadNum());
    // 4-way unrolled: independent iterations keep 8 loads in flight per thread (memory-level parallelism)
    uint32_t stride4 = stride * 4;
    for (uint32_t base = start; base < totalOut; base += stride4) {
        uint32_t idx[4];
        float v0[4];
        float v1[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            idx[u] = base + static_cast<uint32_t>(u) * stride;
            v0[u] = 0.0f;
            v1[u] = 0.0f;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (idx[u] < totalOut) {
                uint32_t row = AscendC::Simt::UintDiv(idx[u], m0, shift0);
                uint32_t src = idx[u] + row * half;
                v0[u] = static_cast<float>(xg[src]);
                v1[u] = static_cast<float>(xg[src + half]);
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (idx[u] < totalOut) {
                float sig = v0[u] / (1.0f + AscendC::Simt::Exp(-v0[u]));
                yg[idx[u]] = static_cast<T>(sig * v1[u]);
            }
        }
    }
}

// SIMT launch happens in a plain __aicore__ function (vendor idiom: VF_CALL must not sit in the
// __global__ entry, otherwise the linker cannot derive the kernel type attribute).
// vendor idiom (ops_cv/.../roi_pooling_with_arg_max.cpp): the __global__ entry declares its task type
// with KERNEL_TASK_TYPE_DEFAULT and then issues Simt::VF_CALL with dim3{threads,1,1}.
__global__ __aicore__ void swiglu_simt_float(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t halfSpan, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t m0 = 0;
    uint32_t shift0 = 0;
    AscendC::GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(halfSpan));
    AscendC::Simt::VF_CALL<SwigluSimt<float>>(AscendC::Simt::Dim3(2048), x, y,
                                              static_cast<uint32_t>(totalOut), static_cast<uint32_t>(halfSpan), m0, shift0, static_cast<uint32_t>(blocks), static_cast<uint32_t>(AscendC::GetBlockIdx()));
}
__global__ __aicore__ void swiglu_simt_half(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t halfSpan, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t m0 = 0;
    uint32_t shift0 = 0;
    AscendC::GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(halfSpan));
    AscendC::Simt::VF_CALL<SwigluSimt<half>>(AscendC::Simt::Dim3(2048), x, y,
                                             static_cast<uint32_t>(totalOut), static_cast<uint32_t>(halfSpan), m0, shift0, static_cast<uint32_t>(blocks), static_cast<uint32_t>(AscendC::GetBlockIdx()));
}
__global__ __aicore__ void swiglu_simt_bf16(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t halfSpan, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t m0 = 0;
    uint32_t shift0 = 0;
    AscendC::GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(halfSpan));
    AscendC::Simt::VF_CALL<SwigluSimt<bfloat16_t>>(AscendC::Simt::Dim3(2048), x, y,
                                                   static_cast<uint32_t>(totalOut), static_cast<uint32_t>(halfSpan), m0, shift0, static_cast<uint32_t>(blocks), static_cast<uint32_t>(AscendC::GetBlockIdx()));
}

#endif  // SWIGLU_HAS_SIMT

int64_t swiglu_core_num()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    return coreNum > 0 ? coreNum : 1;
}

std::tuple<int64_t, int64_t, uint32_t> calc_swiglu_tiling_params(int64_t outer, int64_t half, int64_t elemBytes)
{
    uint64_t ubSize2 = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize2);
    int64_t elemsPerTile = static_cast<int64_t>(ubSize2 * 9 / 10) / (6 * elemBytes + 8);   // 3 queues x2 deep + 2 fp32 scratch slices
    if (elemsPerTile < 256) {
        elemsPerTile = 256;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    constexpr static int64_t TILE_ELEMS = 2048;
    int64_t W = (half < elemsPerTile) ? half : elemsPerTile;   // use the full per-dtype UB capacity
    int64_t R = ((half * elemBytes) % 32 == 0) ? (elemsPerTile / W) : 1;   // see kernel: 32B block rule
    if (R < 1) {
        R = 1;
    }
    int64_t nCols = (half + W - 1) / W;
    int64_t nRowGroups = (outer + R - 1) / R;
    int64_t totalTiles = nRowGroups * nCols;
    int64_t numBlocks = std::min<int64_t>(coreNum, std::max<int64_t>(1, totalTiles));
    int64_t tilesPerBlock = (totalTiles + numBlocks - 1) / numBlocks;   // reused as tilesPerBlock
    // UB budget: 3 queues x PIPELINE_DEPTH x tileBytes  +  4 x tileBytes of fp32 scratch
    uint32_t tileBytes = static_cast<uint32_t>(elemsPerTile * elemBytes);
    return std::make_tuple(numBlocks, tilesPerBlock, tileBytes);
}

std::tuple<int64_t, int64_t, uint32_t> calc_swiglu_tiny_tiling_params(int64_t outer, int64_t half, int64_t elemBytes)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    // per row: 2*half input + half output + 2*half x0/x1 + 2*half fp32 scratch + 2*half offsets
    int64_t perRowBytes = 5 * half * elemBytes + 16 * half;
    if (perRowBytes < 1) {
        perRowBytes = 1;
    }
    int64_t rowsPerTile = static_cast<int64_t>(ubSize * 9 / 10) / perRowBytes;
    int64_t capRows = 4096 / half;
    if (capRows < 1) {
        capRows = 1;
    }
    if (capRows > TINY_MAX_ROWS) {
        capRows = TINY_MAX_ROWS;
    }
    if (rowsPerTile > capRows) {
        rowsPerTile = capRows;
    }
    if (rowsPerTile < 1) {
        rowsPerTile = 1;
    }
    // Match the kernel's 128/64-element Gather/vector repeat alignment.
    int64_t repeatElems = (elemBytes == 2) ? 128 : 64;
    while (rowsPerTile > 1 && (rowsPerTile * half) % repeatElems != 0) {
        --rowsPerTile;
    }
    int64_t totalTiles = (outer + rowsPerTile - 1) / rowsPerTile;
    int64_t numBlocks = std::min<int64_t>(coreNum, std::max<int64_t>(1, totalTiles));
    int64_t tilesPerBlock = (totalTiles + numBlocks - 1) / numBlocks;
    uint32_t tileBytes = static_cast<uint32_t>(rowsPerTile * 2 * half * elemBytes);
    return std::make_tuple(numBlocks, tilesPerBlock, tileBytes);
}

std::tuple<int64_t, int64_t, int64_t> calc_swiglu_unaligned_tiling_params(int64_t outer, int64_t half, int64_t elemBytes)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t halfA = ((half * elemBytes + 31) / 32) * 32 / elemBytes;
    int64_t perRow = halfA * (3 * elemBytes + 8);   // 3 one-deep queues + 2 fp32 scratch slices
    int64_t R = (static_cast<int64_t>(ubSize) * 8 / 10) / std::max<int64_t>(1, perRow);
    if (R > 16) {
        R = 16;
    }
    if (R < 1) {
        return std::make_tuple(0, 0, 0);   // one padded row does not fit; caller uses the main path
    }
    int64_t totalTiles = (outer + R - 1) / R;
    int64_t numBlocks = std::min<int64_t>(coreNum, std::max<int64_t>(1, totalTiles));
    int64_t tilesPerBlock = (totalTiles + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, tilesPerBlock, R);
}

extern "C" {

#if SWIGLU_HAS_SIMT
void launch_swiglu_simt_float(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t halfSpan, int64_t numBlocks, void *stream)
{
    swiglu_simt_float<<<numBlocks, nullptr, stream>>>(x, y, totalOut, halfSpan, numBlocks);
}
void launch_swiglu_simt_half(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t halfSpan, int64_t numBlocks, void *stream)
{
    swiglu_simt_half<<<numBlocks, nullptr, stream>>>(x, y, totalOut, halfSpan, numBlocks);
}
void launch_swiglu_simt_bf16(GM_ADDR x, GM_ADDR y, int64_t totalOut, int64_t halfSpan, int64_t numBlocks, void *stream)
{
    swiglu_simt_bf16<<<numBlocks, nullptr, stream>>>(x, y, totalOut, halfSpan, numBlocks);
}

#endif  // SWIGLU_HAS_SIMT

void launch_swiglu_kernel_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t tilesPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream)
{
    swiglu_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, tilesPerBlock, tileBytes);
}

void launch_swiglu_kernel_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t tilesPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream)
{
    swiglu_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, tilesPerBlock, tileBytes);
}

void launch_swiglu_kernel_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t tilesPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream)
{
    swiglu_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, tilesPerBlock, tileBytes);
}

void launch_swiglu_tiny_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t tilesPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream)
{
    swiglu_tiny_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, tilesPerBlock, tileBytes);
}

void launch_swiglu_tiny_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t tilesPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream)
{
    swiglu_tiny_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, tilesPerBlock, tileBytes);
}

void launch_swiglu_tiny_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t tilesPerBlock, uint32_t tileBytes, int64_t numBlocks, void *stream)
{
    swiglu_tiny_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, tilesPerBlock, tileBytes);
}

void launch_swiglu_unaligned_float(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t R, int64_t tilesPerBlock, int64_t numBlocks, void *stream)
{
    swiglu_unaligned_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, R, tilesPerBlock);
}

void launch_swiglu_unaligned_half(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t R, int64_t tilesPerBlock, int64_t numBlocks, void *stream)
{
    swiglu_unaligned_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, R, tilesPerBlock);
}

void launch_swiglu_unaligned_bf16(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t halfSpan, int64_t R, int64_t tilesPerBlock, int64_t numBlocks, void *stream)
{
    swiglu_unaligned_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, outer, halfSpan, R, tilesPerBlock);
}

} // extern "C"
