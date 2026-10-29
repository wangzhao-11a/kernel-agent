#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <tuple>
#include <algorithm>
#include <type_traits>

// KAG_HW_TRANSPOSE=0 replaces the 16x16 vtranspose with a scalar tile loop (escape hatch in case
// the vector-transpose intrinsic signature differs on this toolkit build)
#ifndef KAG_HW_TRANSPOSE
#define KAG_HW_TRANSPOSE 1
#endif

namespace {
constexpr int64_t kDepth = 2;
constexpr int64_t kMaxDims = 8;
}  // namespace

namespace {

// Adds(x, 0) is a local copy only for the dtypes Adds supports. bf16/int8/int64
// are not supported by Adds on dav_c220; use scalar copy for those instantiations.
template <typename T>
__aicore__ inline void CopyLocalZero(AscendC::LocalTensor<T> &dst,
                                     const AscendC::LocalTensor<T> &src, int32_t n)
{
    if constexpr (std::is_same<T, bfloat16_t>::value || sizeof(T) == 1 || sizeof(T) == 8) {
        for (int32_t i = 0; i < n; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    } else {
        AscendC::Adds(dst, src, static_cast<T>(0), n);
    }
}

template <typename T>
__aicore__ inline void ToF32c(const AscendC::LocalTensor<float> &d, const AscendC::LocalTensor<T> &s, int32_t n)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds(d, s, 0.0f, n);
    } else {
        AscendC::Cast(d, s, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
    }
}

template <typename T>
__aicore__ inline void FromF32c(const AscendC::LocalTensor<T> &d, const AscendC::LocalTensor<float> &s, int32_t n)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds(d, s, 0.0f, n);
    } else {
        AscendC::Cast(d, s, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    }
}


__aicore__ inline float Bf16ToFloatL(bfloat16_t v)
{
    union { bfloat16_t b; uint16_t u; } cv;
    cv.b = v;
    union { uint32_t u; float f; } out;
    out.u = static_cast<uint32_t>(cv.u) << 16;
    return out.f;
}

__aicore__ inline bfloat16_t FloatToBf16L(float v)
{
    union { float f; uint32_t u; } cv;
    cv.f = v;
    uint32_t r = cv.u + 0x8000u + ((cv.u >> 16) & 1u);
    union { uint16_t u; bfloat16_t b; } out;
    out.u = static_cast<uint16_t>(r >> 16);
    return out.b;
}

// ---------------------------------------------------------------- transpose
template <typename T>
__global__ __aicore__ void transpose_kernel(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner,
                                            int64_t inStrideInner, int64_t ndim, int64_t s0, int64_t s1,
                                            int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6,
                                            int64_t s7, int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                                            int64_t d4, int64_t d5, int64_t d6, int64_t d7, int64_t tile)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    int64_t st[kMaxDims] = {s0, s1, s2, s3, s4, s5, s6, s7};
    int64_t sh[kMaxDims] = {d0, d1, d2, d3, d4, d5, d6, d7};
    const int64_t nb = AscendC::GetBlockNum();
    // a negative inStrideInner disables the contiguous-run fast path (A/B switch)
    const bool contiguousRun = (inStrideInner == 1);
    const int64_t isi = (inStrideInner < 0) ? -inStrideInner : inStrideInner;
    for (int64_t r = AscendC::GetBlockIdx(); r < outer; r += nb) {
        int64_t rem = r;
        int64_t base = 0;
        for (int k = static_cast<int>(ndim) - 2; k >= 0; --k) {
            int64_t c = rem % sh[k];
            rem /= sh[k];
            base += c * st[k];
        }
        for (int64_t off = 0; off < inner; off += tile) {
            int32_t n = static_cast<int32_t>((inner - off) < tile ? (inner - off) : tile);
            auto ly = qy.AllocTensor<T>();
            if (contiguousRun) {
                // source run is contiguous: one DMA in, vector copy out (old code did one GM scalar
                // read per element, which for case 1 (64x32x512x128 fp16) is 134M scalar loads)
                auto lx = qx.AllocTensor<T>();
                AscendC::DataCopyPad(lx, xGm[base + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(lx);
                lx = qx.DeQue<T>();
                CopyLocalZero(ly, lx, n);
                qx.FreeTensor(lx);
            } else {
                for (int32_t j = 0; j < n; ++j) {
                    ly.SetValue(j, xGm.GetValue(base + (off + j) * isi));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            qy.EnQue(ly);
            ly = qy.DeQue<T>();
            AscendC::DataCopyPad(yGm[r * inner + off], ly,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
            qy.FreeTensor(ly);
        }
    }
}


// tiled 2-D transpose: out[i,j] = x[j,i]; one block-DMA per 64x64 tile in and out
template <typename T>
__global__ __aicore__ void transpose2d_tiled_kernel(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t TS)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qi;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bOff;
    const int64_t tileBytes = ((((TS * TS * sizeof(T)) + 31) / 32) * 32);
    pipe.InitBuffer(qi, 1, tileBytes);
    pipe.InitBuffer(qo, 1, tileBytes);
    pipe.InitBuffer(bOff, ((((TS * TS * sizeof(uint32_t)) + 31) / 32) * 32));
    auto offs = bOff.Get<uint32_t>();
    for (int64_t i = 0; i < TS * TS; ++i) {
        offs.SetValue(static_cast<int32_t>(i), static_cast<uint32_t>((i % TS) * TS + (i / TS)));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    const int64_t nTilesX = (W + TS - 1) / TS;
    const int64_t nTilesY = (H + TS - 1) / TS;
    const int64_t total = nTilesX * nTilesY;
    const int64_t nb = AscendC::GetBlockNum();
    const uint32_t rowBytes = static_cast<uint32_t>(TS * (int64_t)sizeof(T));
    for (int64_t t = AscendC::GetBlockIdx(); t < total; t += nb) {
        const int64_t ty = t / nTilesX;
        const int64_t tx = t % nTilesX;
        const int64_t r0 = ty * TS;          // starting row of x == starting col of y
        const int64_t c0 = tx * TS;          // starting col of x == starting row of y
        // fully-interior tiles take the fast path; edge tiles fall back to scalar copies
        if (r0 + TS <= H && c0 + TS <= W) {
            auto li = qi.AllocTensor<T>();
            AscendC::DataCopyPad(li, xGm[r0 * W + c0],
                                 AscendC::DataCopyExtParams{static_cast<uint16_t>(TS), rowBytes,
                                                            static_cast<uint32_t>((W - TS) * (int64_t)sizeof(T)), 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            qi.EnQue(li);
            li = qi.DeQue<T>();
            auto lo = qo.AllocTensor<T>();
            for (int64_t i = 0; i < TS; ++i) {
                for (int64_t j = 0; j < TS; ++j) {
                    lo.SetValue(static_cast<int32_t>(i * TS + j), li.GetValue(static_cast<int32_t>(j * TS + i)));
                }
            }
            qi.FreeTensor(li);
            AscendC::PipeBarrier<PIPE_ALL>();
            qo.EnQue(lo);
            lo = qo.DeQue<T>();
            AscendC::DataCopyPad(yGm[c0 * H + r0], lo,
                                 AscendC::DataCopyExtParams{static_cast<uint16_t>(TS), rowBytes, 0,
                                                            static_cast<uint32_t>((H - TS) * (int64_t)sizeof(T)), 0});
            qo.FreeTensor(lo);
        } else {
            const int64_t rEnd = (r0 + TS < H) ? r0 + TS : H;
            const int64_t cEnd = (c0 + TS < W) ? c0 + TS : W;
            for (int64_t j = c0; j < cEnd; ++j) {
                const int32_t n = static_cast<int32_t>(rEnd - r0);
                auto ly = qo.AllocTensor<T>();
                for (int32_t i = 0; i < n; ++i) {
                    ly.SetValue(i, xGm.GetValue((r0 + i) * W + j));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                qo.EnQue(ly);
                ly = qo.DeQue<T>();
                AscendC::DataCopyPad(yGm[j * H + r0], ly,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
                qo.FreeTensor(ly);
            }
        }
    }
}

// tiled 2-D transpose using the 16x16 hardware transpose (16-bit element types only).
// out[b][i][j] = in[b][j][i] with in[b] H x W and out[b] W x H, both row-major/contiguous.
// Interior tiles: 1 block-DMA in (16 rows x 32B) + vtranspose + 1 block-DMA out. Edge tiles keep
// the scalar path (they are < 1% of the tiles for the benchmark shapes).
template <typename T>
__global__ __aicore__ void transpose2d_hw_kernel(GM_ADDR x, GM_ADDR y, int64_t nBatch, int64_t H, int64_t W)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qi;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qo;
    const int64_t tileBytes = ((((16 * 16 * (int64_t)sizeof(T)) + 31) / 32) * 32);
    pipe.InitBuffer(qi, kDepth, tileBytes);
    pipe.InitBuffer(qo, kDepth, tileBytes);
    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    const int64_t nTx = (W + 15) / 16;      // blocks along W -> 16 output rows each
    const int64_t nTy = (H + 15) / 16;      // blocks along H -> 16 output cols each
    const int64_t perBatch = nTx * nTy;
    const int64_t total = nBatch * perBatch;
    const int64_t nb = AscendC::GetBlockNum();
    const uint32_t rowBytes = static_cast<uint32_t>(16 * (int64_t)sizeof(T));
    for (int64_t t = AscendC::GetBlockIdx(); t < total; t += nb) {
        const int64_t b = t / perBatch;
        const int64_t tt = t - b * perBatch;
        const int64_t ty = tt / nTx;
        const int64_t tx = tt - ty * nTx;
        const int64_t r0 = ty * 16;
        const int64_t c0 = tx * 16;
        const int64_t inBase = b * H * W;
        const int64_t outBase = b * W * H;
        if (r0 + 16 <= H && c0 + 16 <= W) {
            auto li = qi.AllocTensor<T>();
            AscendC::DataCopyPad(li, xGm[inBase + r0 * W + c0],
                                 AscendC::DataCopyExtParams{16, rowBytes,
                                                            static_cast<uint32_t>((W - 16) * (int64_t)sizeof(T)), 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            qi.EnQue(li);
            li = qi.DeQue<T>();
            auto lo = qo.AllocTensor<T>();
#if KAG_HW_TRANSPOSE
            AscendC::Transpose(lo, li);
#else
            for (int64_t i = 0; i < 16; ++i) {
                for (int64_t j = 0; j < 16; ++j) {
                    lo.SetValue(static_cast<int32_t>(i * 16 + j),
                                li.GetValue(static_cast<int32_t>(j * 16 + i)));
                }
            }
#endif
            qi.FreeTensor(li);
            AscendC::PipeBarrier<PIPE_V>();
            qo.EnQue(lo);
            lo = qo.DeQue<T>();
            AscendC::DataCopyPad(yGm[outBase + c0 * H + r0], lo,
                                 AscendC::DataCopyExtParams{16, rowBytes, 0,
                                                            static_cast<uint32_t>((H - 16) * (int64_t)sizeof(T)), 0});
            qo.FreeTensor(lo);
        } else {
            const int64_t rEnd = (r0 + 16 < H) ? r0 + 16 : H;
            const int64_t cEnd = (c0 + 16 < W) ? c0 + 16 : W;
            for (int64_t j = c0; j < cEnd; ++j) {
                const int32_t n = static_cast<int32_t>(rEnd - r0);
                auto ly = qo.AllocTensor<T>();
                for (int32_t i = 0; i < n; ++i) {
                    ly.SetValue(i, xGm.GetValue(inBase + (r0 + i) * W + j));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                qo.EnQue(ly);
                ly = qo.DeQue<T>();
                AscendC::DataCopyPad(yGm[outBase + j * H + r0], ly,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
                qo.FreeTensor(ly);
            }
        }
    }
}

// ---------------------------------------------------------------- strided slice
template <typename T>
__global__ __aicore__ void slice_kernel(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart,
                                        int64_t lastStep, int64_t lastInStride, int64_t ndim, int64_t base0,
                                        int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5,
                                        int64_t s6, int64_t s7, int64_t t0, int64_t t1, int64_t t2, int64_t t3,
                                        int64_t t4, int64_t t5, int64_t t6, int64_t t7, int64_t d0, int64_t d1,
                                        int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7,
                                        int64_t tile)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    const int64_t tileBytes = (((((tile < 0 ? -tile : tile) * (int64_t)sizeof(T)) + 31) / 32) * 32);
    pipe.InitBuffer(qx, kDepth, tileBytes);
    pipe.InitBuffer(qy, kDepth, tileBytes);
    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    int64_t st[kMaxDims] = {s0, s1, s2, s3, s4, s5, s6, s7};
    int64_t sp[kMaxDims] = {t0, t1, t2, t3, t4, t5, t6, t7};
    int64_t sh[kMaxDims] = {d0, d1, d2, d3, d4, d5, d6, d7};
    // a negative tile disables the span-DMA compaction path (A/B switch)
    const bool spanPath = (tile > 0) && (lastInStride == 1) && (lastStep > 1);
    const int64_t tileN = (tile < 0) ? -tile : tile;
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < outer; r += nb) {
        int64_t rem = r;
        int64_t base = base0;
        for (int k = static_cast<int>(ndim) - 2; k >= 0; --k) {
            int64_t c = rem % sh[k];
            rem /= sh[k];
            base += c * sp[k] * st[k];
        }
        int64_t off = 0;
        while (off < inner) {
            if (lastStep == 1 && lastInStride == 1) {
                int32_t n = static_cast<int32_t>((inner - off) < tileN ? (inner - off) : tileN);
                auto ly = qy.AllocTensor<T>();
                // contiguous run: one DMA into VECIN, copy to VECOUT, then store
                auto lx = qx.AllocTensor<T>();
                AscendC::DataCopyPad(lx, xGm[base + lastStart + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(lx);
                lx = qx.DeQue<T>();
                CopyLocalZero(ly, lx, n);
                qx.FreeTensor(lx);
                qy.EnQue(ly);
                ly = qy.DeQue<T>();
                AscendC::DataCopyPad(yGm[r * inner + off], ly,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
                qy.FreeTensor(ly);
                off += n;
            } else if (spanPath) {
                // strided pick out of a contiguous run: one DMA for the whole span, then compact in UB
                // (a GM scalar load per element, as the old fallback did, is ~10x slower)
                int64_t maxOut = tileN / lastStep;
                if (maxOut < 1) maxOut = 1;
                const int64_t chunkEnd = ((off + maxOut) < inner) ? (off + maxOut) : inner;
                const int32_t cn = static_cast<int32_t>(chunkEnd - off);
                const int32_t spanLen = static_cast<int32_t>((cn - 1) * lastStep + 1);
                auto ly = qy.AllocTensor<T>();
                auto lx = qx.AllocTensor<T>();
                AscendC::DataCopyPad(lx, xGm[base + lastStart + off * lastStep],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(spanLen * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(lx);
                lx = qx.DeQue<T>();
                for (int32_t j = 0; j < cn; ++j) {
                    ly.SetValue(j, lx.GetValue(j * static_cast<int32_t>(lastStep)));
                }
                qx.FreeTensor(lx);
                AscendC::PipeBarrier<PIPE_ALL>();
                qy.EnQue(ly);
                ly = qy.DeQue<T>();
                AscendC::DataCopyPad(yGm[r * inner + off], ly,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cn * (int64_t)sizeof(T)), 0, 0, 0});
                qy.FreeTensor(ly);
                off = chunkEnd;
            } else {
                int32_t n = static_cast<int32_t>((inner - off) < tileN ? (inner - off) : tileN);
                auto ly = qy.AllocTensor<T>();
                for (int32_t j = 0; j < n; ++j) {
                    const int64_t src = base + (lastStart + (off + j) * lastStep) * lastInStride;
                    ly.SetValue(j, xGm.GetValue(src));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                qy.EnQue(ly);
                ly = qy.DeQue<T>();
                AscendC::DataCopyPad(yGm[r * inner + off], ly,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
                qy.FreeTensor(ly);
                off += n;
            }
        }
    }
}

// ---------------------------------------------------------------- adaptive avg pool 3d
__aicore__ inline int64_t AdaptStart(int64_t o, int64_t in, int64_t out)
{
    return (o * in) / out;
}
__aicore__ inline int64_t AdaptEnd(int64_t o, int64_t in, int64_t out)
{
    return ((o + 1) * in + out - 1) / out;
}

template <typename T>
__global__ __aicore__ void adapt_pool_kernel(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H,
                                             int64_t W, int64_t OD, int64_t OH, int64_t OW, int64_t tile)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    const int64_t rows = N * C * OD * OH;
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        int64_t t = r;
        const int64_t oh = t % OH; t /= OH;
        const int64_t od = t % OD; t /= OD;
        const int64_t c = t % C;
        const int64_t n = t / C;
        const int64_t d0 = AdaptStart(od, D, OD);
        const int64_t d1 = AdaptEnd(od, D, OD);
        const int64_t h0 = AdaptStart(oh, H, OH);
        const int64_t h1 = AdaptEnd(oh, H, OH);
        for (int64_t off = 0; off < OW; off += tile) {
            int32_t nn = static_cast<int32_t>((OW - off) < tile ? (OW - off) : tile);
            auto ly = qy.AllocTensor<T>();
            for (int32_t j = 0; j < nn; ++j) {
                const int64_t ow = off + j;
                const int64_t w0 = AdaptStart(ow, W, OW);
                const int64_t w1 = AdaptEnd(ow, W, OW);
                float acc = 0.0f;
                for (int64_t d = d0; d < d1; ++d) {
                    for (int64_t h = h0; h < h1; ++h) {
                        const int64_t rowBase = ((n * C + c) * D + d) * H * W + h * W;
                        for (int64_t w = w0; w < w1; ++w) {
                            if constexpr (std::is_same<T, bfloat16_t>::value) {
                                acc += Bf16ToFloatL(xGm.GetValue(rowBase + w));
                            } else {
                                acc += static_cast<float>(xGm.GetValue(rowBase + w));
                            }
                        }
                    }
                }
                const float cnt = static_cast<float>((d1 - d0) * (h1 - h0) * (w1 - w0));
                const float avg = acc / cnt;
                if constexpr (std::is_same<T, bfloat16_t>::value) {
                    ly.SetValue(j, FloatToBf16L(avg));
                } else {
                    ly.SetValue(j, static_cast<T>(avg));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            qy.EnQue(ly);
            ly = qy.DeQue<T>();
            AscendC::DataCopyPad(yGm[r * OW + off], ly,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(nn * (int64_t)sizeof(T)), 0, 0, 0});
            qy.FreeTensor(ly);
        }
    }
}

// ---------------------------------------------------------------- foreach addcdiv scalar
template <typename T>
__global__ __aicore__ void addcdiv_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, int64_t total,
                                          int64_t blockLength, int64_t tile, float scalar)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> q1, q2, q3;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2;
    pipe.InitBuffer(q1, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(q2, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(q3, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b1, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b2, ((((tile * sizeof(float)) + 31) / 32) * 32));
    auto f0 = b0.Get<float>();
    auto f1 = b1.Get<float>();
    auto f2 = b2.Get<float>();
    AscendC::GlobalTensor<T> aGm, bGm, cGm, yGm;
    aGm.SetGlobalBuffer((__gm__ T *)x1);
    bGm.SetGlobalBuffer((__gm__ T *)x2);
    cGm.SetGlobalBuffer((__gm__ T *)x3);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    int64_t base = AscendC::GetBlockIdx() * blockLength;
    int64_t len = total - base;
    if (len <= 0) return;
    if (len > blockLength) len = blockLength;
    for (int64_t off = 0; off < len; off += tile) {
        int32_t n = static_cast<int32_t>((len - off) < tile ? (len - off) : tile);
        const uint32_t nb = static_cast<uint32_t>(n * (int64_t)sizeof(T));
        auto la = q1.AllocTensor<T>();
        auto lb = q2.AllocTensor<T>();
        auto lc = q3.AllocTensor<T>();
        AscendC::DataCopyPad(la, aGm[base + off], AscendC::DataCopyExtParams{1, nb, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        AscendC::DataCopyPad(lb, bGm[base + off], AscendC::DataCopyExtParams{1, nb, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        AscendC::DataCopyPad(lc, cGm[base + off], AscendC::DataCopyExtParams{1, nb, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        q1.EnQue(la);
        q2.EnQue(lb);
        q3.EnQue(lc);
        la = q1.DeQue<T>();
        lb = q2.DeQue<T>();
        lc = q3.DeQue<T>();
        ToF32c(f0, la, n);
        ToF32c(f1, lb, n);
        ToF32c(f2, lc, n);
        q1.FreeTensor(la);
        q2.FreeTensor(lb);
        q3.FreeTensor(lc);
        AscendC::Div(f2, f1, f2, n);
        if (scalar != 1.0f) {
            AscendC::Muls(f2, f2, scalar, n);
        }
        AscendC::Add(f0, f0, f2, n);
        auto ly = qy.AllocTensor<T>();
        FromF32c(ly, f0, n);
        qy.EnQue(ly);
        ly = qy.DeQue<T>();
        AscendC::DataCopyPad(yGm[base + off], ly, AscendC::DataCopyExtParams{1, nb, 0, 0, 0});
        qy.FreeTensor(ly);
    }
}

// ---------------------------------------------------------------- foreach norm
template <typename T>
__global__ __aicore__ void norm_partial_kernel(GM_ADDR x, GM_ADDR ws, int64_t total, int64_t blockLength,
                                               int64_t tile, float p, int32_t mode)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3;
    pipe.InitBuffer(qo, 1, 32);
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b1, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b2, ((((8 * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b3, ((((tile * sizeof(float)) + 31) / 32) * 32));
    auto f0 = b0.Get<float>();
    auto f1 = b1.Get<float>();
    auto red = b2.Get<float>();
    auto f3 = b3.Get<float>();
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<float> wGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    wGm.SetGlobalBuffer((__gm__ float *)ws);
    int64_t base = AscendC::GetBlockIdx() * blockLength;
    int64_t len = total - base;
    if (len <= 0) {
        auto oz = qo.AllocTensor<float>();
        oz.SetValue(0, (mode == 2) ? 3.0e38f : 0.0f);
        qo.EnQue(oz);
        oz = qo.DeQue<float>();
        AscendC::DataCopyPad(wGm[AscendC::GetBlockIdx()], oz, AscendC::DataCopyExtParams{1, 4u, 0, 0, 0});
        qo.FreeTensor(oz);
        return;
    }
    if (len > blockLength) len = blockLength;
    float acc = (mode == 2) ? 3.0e38f : 0.0f;
    for (int64_t off = 0; off < len; off += tile) {
        int32_t n = static_cast<int32_t>((len - off) < tile ? (len - off) : tile);
        auto src = qx.AllocTensor<T>();
        AscendC::DataCopyPad(src, xGm[base + off],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        qx.EnQue(src);
        src = qx.DeQue<T>();
        ToF32c(f0, src, n);
        qx.FreeTensor(src);
        AscendC::Abs(f0, f0, n);
        if (mode == 0) {                       // sum |x|^p
            if (p == 1.0f) {
                AscendC::Adds(f1, f0, 0.0f, n);
            } else if (p == 2.0f) {
                AscendC::Mul(f1, f0, f0, n);
            } else {
                AscendC::Ln(f1, f0, n);
                AscendC::Muls(f1, f1, p, n);
                AscendC::Exp(f1, f1, n);
            }
            AscendC::ReduceSum<float>(red, f1, f3, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            acc += red.GetValue(0);
        } else if (mode == 1) {                // max
            AscendC::ReduceMax<float>(red, f0, f3, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            float v = red.GetValue(0);
            if (v > acc) acc = v;
        } else {                               // min
            AscendC::ReduceMin<float>(red, f0, f3, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            float v = red.GetValue(0);
            if (v < acc) acc = v;
        }
    }
    auto o = qo.AllocTensor<float>();
    o.SetValue(0, acc);
    qo.EnQue(o);
    o = qo.DeQue<float>();
    AscendC::DataCopyPad(wGm[AscendC::GetBlockIdx()], o, AscendC::DataCopyExtParams{1, 4u, 0, 0, 0});
    qo.FreeTensor(o);
}

__global__ __aicore__ void norm_finalize_kernel(GM_ADDR ws, GM_ADDR out, int64_t blocks, float p, int32_t mode,
                                                int32_t outIsF32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::GlobalTensor<float> wGm;
    wGm.SetGlobalBuffer((__gm__ float *)ws);
    AscendC::TPipe pipeW;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qw;
    pipeW.InitBuffer(qw, 1, 256);
    float acc = (mode == 2) ? 3.0e38f : 0.0f;
    for (int64_t off = 0; off < blocks; off += 64) {
        int32_t n = static_cast<int32_t>((blocks - off) < 64 ? (blocks - off) : 64);
        auto w = qw.AllocTensor<float>();
        AscendC::DataCopyPad(w, wGm[off],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * 4), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        qw.EnQue(w);
        w = qw.DeQue<float>();
        for (int32_t i = 0; i < n; ++i) {
            float v = w.GetValue(i);
            if (mode == 0) acc += v;
            else if (mode == 1) { if (v > acc) acc = v; }
            else { if (v < acc) acc = v; }
        }
        qw.FreeTensor(w);
    }
    float res = acc;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qo;
    pipe.InitBuffer(b0, 64);
    pipe.InitBuffer(qo, 1, 64);
    auto f = b0.Get<float>();
    for (int i = 0; i < 8; ++i) f.SetValue(i, acc);
    AscendC::PipeBarrier<PIPE_ALL>();
    if (mode == 0) {
        if (p == 1.0f) {
            res = acc;
        } else if (p == 2.0f) {
            AscendC::Sqrt(f, f, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            res = f.GetValue(0);
        } else {
            AscendC::Ln(f, f, 8);
            AscendC::Muls(f, f, 1.0f / p, 8);
            AscendC::Exp(f, f, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            res = f.GetValue(0);
        }
    }
    if (outIsF32) {
        AscendC::GlobalTensor<float> oF;
        oF.SetGlobalBuffer((__gm__ float *)out);
        oF.SetValue(0, res);
    } else {
        for (int i = 0; i < 8; ++i) f.SetValue(i, res);
        AscendC::PipeBarrier<PIPE_ALL>();
        auto o = qo.AllocTensor<half>();
        AscendC::Cast(o, f, AscendC::RoundMode::CAST_NONE, 8u);
        qo.EnQue(o);
        o = qo.DeQue<half>();
        AscendC::GlobalTensor<half> oH;
        oH.SetGlobalBuffer((__gm__ half *)out);
        AscendC::DataCopyPad(oH[0], o, AscendC::DataCopyExtParams{1, 2u, 0, 0, 0});
        qo.FreeTensor(o);
    }
}

}  // namespace

// grid size: never fewer blocks than the 40 we used before, but use every AIV the platform reports
static inline int64_t kag_block_num()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t n = plat->GetCoreNumAiv();
    return (n > 40) ? n : 40;
}

std::tuple<int64_t, int64_t, int64_t> calc_layout_tiling(int64_t inner)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    int64_t tile = static_cast<int64_t>(ub) / 2 / 32;
    tile = (tile / 8) * 8;
    if (tile > inner) tile = (inner + 7) / 8 * 8;
    if (tile < 8) tile = 8;
    return std::make_tuple(tile, tile, tile);
}

extern "C" {
#define TRHW_DEF(SUF, TN)                                                                                    \
    void launch_transpose2d_hw_##SUF(GM_ADDR x, GM_ADDR y, int64_t nBatch, int64_t H, int64_t W, void *stream) \
    {                                                                                                        \
        transpose2d_hw_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, y, nBatch, H, W);                              \
    }
TRHW_DEF(f16, half)
TRHW_DEF(bf16, bfloat16_t)
TRHW_DEF(i16, int16_t)
#undef TRHW_DEF

#define TR2_DEF(SUF, TN)                                                                                     \
    void launch_transpose2d_tiled_##SUF(GM_ADDR x, GM_ADDR y, int64_t H, int64_t W, int64_t T, void *stream)  \
    {                                                                                                         \
        transpose2d_tiled_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, y, H, W, T);                                 \
    }
TR2_DEF(f32, float)
TR2_DEF(f16, half)
TR2_DEF(i8, int8_t)
TR2_DEF(i16, int16_t)
TR2_DEF(i32, int32_t)
TR2_DEF(i64, int64_t)
TR2_DEF(bf16, bfloat16_t)
#undef TR2_DEF

#define TR_DEF(SUF, TN)                                                                                       \
    void launch_transpose_##SUF(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t inStrideInner,      \
                                int64_t ndim, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,     \
                                int64_t s5, int64_t s6, int64_t s7, int64_t d0, int64_t d1, int64_t d2,      \
                                int64_t d3, int64_t d4, int64_t d5, int64_t d6, int64_t d7, int64_t tile,    \
                                void *stream)                                                                \
    {                                                                                                         \
        transpose_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, y, outer, inner, inStrideInner, ndim, s0, s1, s2, s3,   \
                                                      s4, s5, s6, s7, d0, d1, d2, d3, d4, d5, d6, d7, tile);   \
    }
TR_DEF(float, float)
TR_DEF(half, half)
TR_DEF(bf16, bfloat16_t)
TR_DEF(i8, int8_t)
TR_DEF(i16, int16_t)
TR_DEF(i32, int32_t)
TR_DEF(i64, int64_t)
#undef TR_DEF

#define SL_DEF(SUF, TN)                                                                                       \
    void launch_slice_##SUF(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t inner, int64_t lastStart,             \
                            int64_t lastStep, int64_t lastInStride, int64_t ndim, int64_t base0, int64_t s0,   \
                            int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7,\
                            int64_t t0, int64_t t1, int64_t t2, int64_t t3, int64_t t4, int64_t t5, int64_t t6,\
                            int64_t t7, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, \
                            int64_t d6, int64_t d7, int64_t tile, void *stream)                                \
    {                                                                                                         \
        slice_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, y, outer, inner, lastStart, lastStep, lastInStride, ndim,   \
                                                  base0, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4,   \
                                                  t5, t6, t7, d0, d1, d2, d3, d4, d5, d6, d7, tile);           \
    }
SL_DEF(float, float)
SL_DEF(half, half)
SL_DEF(bf16, bfloat16_t)
SL_DEF(i8, int8_t)
SL_DEF(i32, int32_t)
SL_DEF(i64, int64_t)
#undef SL_DEF

#define AP_DEF(SUF, TN)                                                                                       \
    void launch_adapt_pool_##SUF(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,   \
                                 int64_t OD, int64_t OH, int64_t OW, int64_t tile, void *stream)               \
    {                                                                                                         \
        adapt_pool_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, y, N, C, D, H, W, OD, OH, OW, tile);                 \
    }
AP_DEF(bf16, bfloat16_t)
AP_DEF(float, float)
AP_DEF(half, half)
#undef AP_DEF

#define AC_DEF(SUF, TN)                                                                                       \
    void launch_addcdiv_##SUF(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, int64_t total, int64_t blocks,     \
                              int64_t blockLength, int64_t tile, float scalar, void *stream)                    \
    {                                                                                                         \
        addcdiv_kernel<TN><<<blocks, nullptr, stream>>>(x1, x2, x3, y, total, blockLength, tile, scalar);       \
    }
AC_DEF(float, float)
AC_DEF(half, half)
#undef AC_DEF

#define NP_DEF(SUF, TN)                                                                                       \
    void launch_norm_partial_##SUF(GM_ADDR x, GM_ADDR ws, int64_t total, int64_t blocks, int64_t blockLength,    \
                                   int64_t tile, float p, int32_t mode, void *stream)                          \
    {                                                                                                         \
        norm_partial_kernel<TN><<<blocks, nullptr, stream>>>(x, ws, total, blockLength, tile, p, mode);         \
    }
NP_DEF(float, float)
NP_DEF(half, half)
#undef NP_DEF

void launch_norm_finalize(GM_ADDR ws, GM_ADDR out, int64_t blocks, float p, int32_t mode, int32_t outIsF32,
                          void *stream)
{
    norm_finalize_kernel<<<1, nullptr, stream>>>(ws, out, blocks, p, mode, outIsF32);
}
}
