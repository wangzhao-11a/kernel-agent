#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <tuple>
#include <algorithm>
#include <type_traits>

namespace {
constexpr int64_t kDepth = 2;
constexpr int64_t kMaxDims = 8;
}  // namespace

static inline int64_t B2CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---- pipeline sync helpers: event flags are much cheaper than a full PIPE_ALL barrier, but if the
// HardEvent enum differs on this toolkit build, set KAG_USE_EVENTS=0 and everything falls back to
// the conservative barrier.
#ifndef KAG_USE_EVENTS
#define KAG_USE_EVENTS 1
#endif
#if KAG_USE_EVENTS
#define KAG_V_S() do { AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0); } while (0)
#define KAG_S_V() do { AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0); } while (0)
#else
#define KAG_V_S() do { AscendC::PipeBarrier<PIPE_ALL>(); } while (0)
#define KAG_S_V() do { AscendC::PipeBarrier<PIPE_ALL>(); } while (0)
#endif


namespace {

template <typename T>
__aicore__ inline void ToF32(const AscendC::LocalTensor<float> &d, const AscendC::LocalTensor<T> &s, int32_t n)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds(d, s, 0.0f, n);
    } else {
        AscendC::Cast(d, s, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
    }
}

template <typename T>
__aicore__ inline void FromF32(const AscendC::LocalTensor<T> &d, const AscendC::LocalTensor<float> &s, int32_t n)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds(d, s, 0.0f, n);
    } else {
        AscendC::Cast(d, s, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    }
}

// scalar math through the vector unit (scalar libm is unavailable in aicore code)
__aicore__ inline float VecSqrt1(AscendC::LocalTensor<float> &buf, float v)
{
    for (int i = 0; i < 8; ++i) buf.SetValue(i, v);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Sqrt(buf, buf, 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    return buf.GetValue(0);
}

__aicore__ inline float VecLn1(AscendC::LocalTensor<float> &buf, float v)
{
    for (int i = 0; i < 8; ++i) buf.SetValue(i, v);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Ln(buf, buf, 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    return buf.GetValue(0);
}


__aicore__ inline float Bf16ToFloat(bfloat16_t v)
{
    union { bfloat16_t b; uint16_t u; } cv;
    cv.b = v;
    union { uint32_t u; float f; } out;
    out.u = static_cast<uint32_t>(cv.u) << 16;
    return out.f;
}

__aicore__ inline bfloat16_t FloatToBf16(float v)
{
    union { float f; uint32_t u; } cv;
    cv.f = v;
    uint32_t r = cv.u + 0x8000u + ((cv.u >> 16) & 1u);
    union { uint16_t u; bfloat16_t b; } out;
    out.u = static_cast<uint16_t>(r >> 16);
    return out.b;
}

// ---------------------------------------------------------------- dynamic quant
// v2 (2026-10-27) - every step below is gated by `flags` from the plugin so one build can A/B:
//   bit0 KAG_DQ_I8    int8 lanes come from a vector Cast chain instead of a per-element
//                     GetValue/SetValue loop (that loop cost ~20 cycles/element and dominated the
//                     whole op).  int8 is only a legal Cast *destination* for half/float sources on
//                     this arch, so the last hop is half; the f32->s32 hop in front of it keeps the
//                     rounding identical to the scalar path.  Chain: f32 -RINT-> s32 -> f32 ->
//                     half -> s8 (the last three hops are exact for |v| <= 128).  int32->half is
//                     avoided on purpose: it multiplexes g_deqValue instead of being a plain
//                     integer conversion.
//   bit1 KAG_DQ_MULS  Muls(1/sc) instead of Duplicate+Div: one vector op less, but it is 1ulp off
//                     the reference division and flips int8 near-ties, so it stays off.
//   bit2 KAG_DQ_1PASS cache the row in UB as fp32 when cols <= tile; the quantise pass then runs
//                     on the cached row instead of reading x from GM a second time.  The load of
//                     the next row is issued before the current row is processed (software
//                     pipeline) so the GM latency of narrow rows overlaps the vector work.
//   (opt3) one vector reduce + one V_S readback per row instead of a PIPE_ALL stall + GetValue per
//          tile; a multi-tile row first folds its tiles with vector Max.
// Rows whose amax is NaN/Inf: every y is 0 for those rows (x/sc is 0 or NaN, and the reference maps
// NaN to 0), so the vector path emits zeros instead of running the scalar loop.
__aicore__ inline void DqNarrowScalar(const AscendC::LocalTensor<int8_t> &dst,
                                      const AscendC::LocalTensor<int32_t> &src, int32_t n)
{
    for (int32_t i = 0; i < n; ++i) {
        dst.SetValue(i, static_cast<int8_t>(src.GetValue(i)));
    }
}

// one 4-byte scale per row, DMA'd out of a 1-element queue (a scalar sGm.SetValue per row was
// measured: 1/20, i.e. broken here, and only 4% faster - do not use)
__aicore__ inline void DqStoreScale(AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> &qs,
                                    AscendC::GlobalTensor<float> &sGm, int64_t r, float sc)
{
    auto lsc = qs.AllocTensor<float>();
    lsc.SetValue(0, sc);
    qs.EnQue(lsc);
    lsc = qs.DeQue<float>();
    AscendC::DataCopyPad(sGm[r], lsc, AscendC::DataCopyExtParams{1, 4u, 0, 0, 0});
    qs.FreeTensor(lsc);
}

__aicore__ inline void DqZeros(AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> &qy,
                               AscendC::GlobalTensor<int8_t> &yGm,
                               const AscendC::LocalTensor<float> &f0,
                               const AscendC::LocalTensor<float> &f1,
                               const AscendC::LocalTensor<int32_t> &i32,
                               const AscendC::LocalTensor<half> &hlf,
                               int64_t dstOff, int32_t n)
{
    AscendC::Duplicate(f0, 0.0f, n);
    AscendC::Cast(i32, f0, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    AscendC::Cast(f1, i32, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    AscendC::Cast(hlf, f1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    auto dst = qy.AllocTensor<int8_t>();
    AscendC::Cast(dst, hlf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    qy.EnQue(dst);
    dst = qy.DeQue<int8_t>();
    AscendC::DataCopyPad(yGm[dstOff], dst, AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n), 0, 0, 0});
    qy.FreeTensor(dst);
}

// scale + clamp + round n fp32 values that are already sitting in f0 and write the int8 chunk
__aicore__ inline void DqQuantChunk(AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> &qy,
                                    AscendC::GlobalTensor<int8_t> &yGm,
                                    const AscendC::LocalTensor<float> &f0,
                                    const AscendC::LocalTensor<float> &f1,
                                    const AscendC::LocalTensor<int32_t> &i32,
                                    const AscendC::LocalTensor<half> &hlf,
                                    int64_t dstOff, int32_t n, float sc, bool vecI8, bool muls)
{
    if (muls) {
        AscendC::Muls(f0, f0, 1.0f / sc, n);
    } else {
        AscendC::Duplicate(f1, sc, n);
        AscendC::Div(f0, f0, f1, n);
    }
    AscendC::Maxs(f0, f0, -128.0f, n);
    AscendC::Mins(f0, f0, 127.0f, n);
    auto dst = qy.AllocTensor<int8_t>();
    AscendC::Cast(i32, f0, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    if (vecI8) {
        AscendC::Cast(f1, i32, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
        AscendC::Cast(hlf, f1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
        AscendC::Cast(dst, hlf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
    } else {
        KAG_V_S();
        DqNarrowScalar(dst, i32, n);
    }
    qy.EnQue(dst);
    dst = qy.DeQue<int8_t>();
    AscendC::DataCopyPad(yGm[dstOff], dst, AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n), 0, 0, 0});
    qy.FreeTensor(dst);
}

template <typename T>
__aicore__ inline void DqLoadTile(AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> &qx,
                                  AscendC::GlobalTensor<T> &xGm, int64_t srcOff, int32_t n,
                                  AscendC::LocalTensor<T> &src)
{
    src = qx.AllocTensor<T>();
    AscendC::DataCopyPad(src, xGm[srcOff],
                         AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                         AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
    qx.EnQue(src);
    src = qx.DeQue<T>();
}

template <typename T>
__global__ __aicore__ void dynamic_quant_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR scale, int64_t rows,
                                                int64_t cols, int64_t tile, int32_t flags)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy, qs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3, b4, b5;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(int8_t)) + 31) / 32) * 32));
    pipe.InitBuffer(qs, 1, 32);
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));    // f0: row cache / tile
    pipe.InitBuffer(b1, ((((tile * sizeof(float)) + 31) / 32) * 32));    // f1: abs + running max
    pipe.InitBuffer(b2, ((((8 * sizeof(float)) + 31) / 32) * 32));       // red
    pipe.InitBuffer(b3, ((((tile * sizeof(float)) + 31) / 32) * 32));    // reduce work / abs tmp
    pipe.InitBuffer(b4, ((((tile * sizeof(int32_t)) + 31) / 32) * 32));  // i32 rounded lanes
    pipe.InitBuffer(b5, ((((tile * sizeof(half)) + 31) / 32) * 32));     // half narrowing hop
    auto f0 = b0.Get<float>();
    auto f1 = b1.Get<float>();
    auto red = b2.Get<float>();
    auto tmp = b3.Get<float>();
    auto i32 = b4.Get<int32_t>();
    auto hlf = b5.Get<half>();

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<float> sGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ int8_t *)y);
    sGm.SetGlobalBuffer((__gm__ float *)scale);

    const int64_t nb = AscendC::GetBlockNum();
    const bool vecI8 = (flags & 1) != 0;
    const bool muls = (flags & 2) != 0;
    const bool onePass = ((flags & 4) != 0) && (cols <= tile);
    const int32_t nRow = static_cast<int32_t>(cols);

    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        if (onePass) {
            AscendC::LocalTensor<T> src;
            DqLoadTile(qx, xGm, r * cols, nRow, src);
            ToF32(f0, src, nRow);
            qx.FreeTensor(src);
            AscendC::Abs(f1, f0, nRow);
            AscendC::ReduceMax<float>(red, f1, tmp, nRow);
            KAG_V_S();
            const float mx0 = red.GetValue(0);
            const float mx = (mx0 < 1.0e-12f) ? 1.0e-12f : mx0;
            const float sc = mx / 127.0f;
            DqStoreScale(qs, sGm, r, sc);
            const bool finite = (mx0 < 3.0e38f);  // NaN and +Inf both fail this test
            if (vecI8 && !finite) {
                DqZeros(qy, yGm, f0, f1, i32, hlf, r * cols, nRow);  // reference output is all zeros
            } else {
                DqQuantChunk(qy, yGm, f0, f1, i32, hlf, r * cols, nRow, sc, vecI8, muls);
            }
        } else {
            bool first = true;
            for (int64_t off = 0; off < cols; off += tile) {
                const int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                AscendC::LocalTensor<T> src;
                DqLoadTile(qx, xGm, r * cols + off, n, src);
                ToF32(f0, src, n);
                qx.FreeTensor(src);
                if (first) {
                    AscendC::Abs(f1, f0, n);
                    first = false;
                } else {
                    AscendC::Abs(tmp, f0, n);
                    AscendC::Max(f1, f1, tmp, n);
                }
            }
            // slots [0, min(tile, cols)) always hold abs values of this row; the rest is stale
            const int32_t rn = static_cast<int32_t>((cols < tile) ? cols : tile);
            AscendC::ReduceMax<float>(red, f1, tmp, rn);
            KAG_V_S();
            const float mx0 = red.GetValue(0);
            const float mx = (mx0 < 1.0e-12f) ? 1.0e-12f : mx0;
            const float sc = mx / 127.0f;
            DqStoreScale(qs, sGm, r, sc);
            const bool ok = vecI8 && (mx0 < 3.0e38f);
            for (int64_t off = 0; off < cols; off += tile) {
                const int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                if (vecI8 && !(mx0 < 3.0e38f)) {
                    DqZeros(qy, yGm, f0, f1, i32, hlf, r * cols + off, n);
                } else {
                    AscendC::LocalTensor<T> src;
                    DqLoadTile(qx, xGm, r * cols + off, n, src);
                    ToF32(f0, src, n);
                    qx.FreeTensor(src);
                    DqQuantChunk(qy, yGm, f0, f1, i32, hlf, r * cols + off, n, sc, ok, muls);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- rms norm v2
// Differences vs rms_norm_kernel:
//   * tile sized from the real UB budget (calc_row_tiling2: 5*es+16 bytes/elem instead of 8*es+32)
//   * when the whole row fits (cacheRow) gamma is loaded ONCE per core instead of once per row
//     (the old path re-read gamma for every row -> for 4096 rows it moved as many gamma bytes as x)
//   * vector->scalar waits use SetFlag/WaitFlag<HardEvent::V_S> instead of a full PIPE_ALL barrier
template <typename T>
__global__ __aicore__ void rms_norm2_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, int64_t rows, int64_t cols,
                                            int64_t tile, float eps)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx, qg;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3, b4;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qg, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));   // float row
    pipe.InitBuffer(b1, ((((tile * sizeof(float)) + 31) / 32) * 32));   // gamma (float, resident)
    pipe.InitBuffer(b2, ((((tile * sizeof(float)) + 31) / 32) * 32));   // x*x
    pipe.InitBuffer(b3, ((((tile * sizeof(float)) + 31) / 32) * 32));   // ReduceSum work
    pipe.InitBuffer(b4, 64);                                            // 8-lane scalar maths
    auto row = b0.Get<float>();
    auto gam = b1.Get<float>();
    auto sq = b2.Get<float>();
    auto work = b3.Get<float>();
    auto sm = b4.Get<float>();

    AscendC::GlobalTensor<T> xGm, gGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    gGm.SetGlobalBuffer((__gm__ T *)gamma);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    const bool cacheRow = (cols <= tile);
    if (cacheRow) {
        // one gamma load per core; gamma stays in UB for every row this core owns
        auto g = qg.AllocTensor<T>();
        AscendC::DataCopyPad(g, gGm[0],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cols * (int64_t)sizeof(T)), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        qg.EnQue(g);
        g = qg.DeQue<T>();
        ToF32(gam, g, static_cast<int32_t>(cols));
        qg.FreeTensor(g);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        float ss = 0.0f;
        if (cacheRow) {
            int32_t n = static_cast<int32_t>(cols);
            auto src = qx.AllocTensor<T>();
            AscendC::DataCopyPad(src, xGm[r * cols],
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            qx.EnQue(src);
            src = qx.DeQue<T>();
            ToF32(row, src, n);
            qx.FreeTensor(src);
            AscendC::Mul(sq, row, row, n);
            AscendC::ReduceSum<float>(sm, sq, work, n);
            KAG_V_S();
            ss = sm.GetValue(0);
        } else {
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(row, src, n);
                qx.FreeTensor(src);
                AscendC::Mul(sq, row, row, n);
                AscendC::ReduceSum<float>(sm, sq, work, n);
                KAG_V_S();
                ss += sm.GetValue(0);
            }
        }
        const float mean = ss / static_cast<float>(cols);
        for (int i = 0; i < 8; ++i) sm.SetValue(i, mean + eps);
        KAG_S_V();
        AscendC::Sqrt(sm, sm, 8);
        KAG_V_S();
        const float rstd = 1.0f / sm.GetValue(0);
        if (cacheRow) {
            int32_t n = static_cast<int32_t>(cols);
            AscendC::Muls(row, row, rstd, n);
            AscendC::Mul(row, row, gam, n);
            auto dst = qy.AllocTensor<T>();
            FromF32(dst, row, n);
            qy.EnQue(dst);
            dst = qy.DeQue<T>();
            AscendC::DataCopyPad(yGm[r * cols], dst,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
            qy.FreeTensor(dst);
        } else {
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                auto g = qg.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                AscendC::DataCopyPad(g, gGm[off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                qg.EnQue(g);
                src = qx.DeQue<T>();
                g = qg.DeQue<T>();
                ToF32(row, src, n);
                ToF32(gam, g, n);
                qx.FreeTensor(src);
                qg.FreeTensor(g);
                AscendC::Muls(row, row, rstd, n);
                AscendC::Mul(row, row, gam, n);
                auto dst = qy.AllocTensor<T>();
                FromF32(dst, row, n);
                qy.EnQue(dst);
                dst = qy.DeQue<T>();
                AscendC::DataCopyPad(yGm[r * cols + off], dst,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
                qy.FreeTensor(dst);
            }
        }
    }
}

// Batched cached-row RMSNorm: one tile holds R rows, so each tile needs one input copy, one
// output copy and one multi-repeat WholeReduceSum instead of one of each per row. The per-row
// normalisation still uses a scalar rstd (read from UB), but there is no per-row DMA or
// PIPE_ALL barrier. Only selected for 32B-aligned rows with cols/8 <= 255 (repeat stride is u8).
template <typename T>
__global__ __aicore__ void rms_norm3_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, int64_t rows, int64_t cols,
                                            int64_t tile, float eps, int64_t useBcast)
{
    AscendC::TPipe pipe;
    constexpr int64_t kMaxRows = 128;
    int64_t R = tile / cols;
    if (R > kMaxRows) R = kMaxRows;
    if (R < 1) R = 1;
    const int64_t tileElems = R * cols;
    const int64_t tileBytes = tileElems * static_cast<int64_t>(sizeof(T));
    const int64_t tileBytes32 = ((tileBytes + 31) / 32) * 32;
    const int64_t f32Bytes = ((tileElems * 4 + 31) / 32) * 32;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qx, qg;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bX, bSq, bRed, bRstd, bG;
    pipe.InitBuffer(qx, 1, tileBytes32);
    pipe.InitBuffer(qg, 1, ((cols * (int64_t)sizeof(T) + 31) / 32) * 32);
    pipe.InitBuffer(qy, 1, tileBytes32);
    pipe.InitBuffer(bX, f32Bytes);
    pipe.InitBuffer(bSq, f32Bytes);
    pipe.InitBuffer(bRed, ((R * 64 * 4 + 31) / 32) * 32);    // one 64-float sum lane per row
    pipe.InitBuffer(bRstd, ((R * 4 + 31) / 32) * 32);
    pipe.InitBuffer(bG, ((cols * 4 + 31) / 32) * 32);
    auto xf = bX.Get<float>();
    auto sq = bSq.Get<float>();
    auto red = bRed.Get<float>();
    auto rstd = bRstd.Get<float>();
    auto gam = bG.Get<float>();

    AscendC::GlobalTensor<T> xGm, gGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    gGm.SetGlobalBuffer((__gm__ T *)gamma);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    auto g = qg.AllocTensor<T>();
    AscendC::DataCopyPad(g, gGm[0],
                         AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cols * (int64_t)sizeof(T)), 0, 0, 0},
                         AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
    qg.EnQue(g);
    g = qg.DeQue<T>();
    ToF32(gam, g, static_cast<int32_t>(cols));
    qg.FreeTensor(g);
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t nb = AscendC::GetBlockNum();
    const int64_t rowsPerCore = (rows + nb - 1) / nb;
    const int64_t rStart = AscendC::GetBlockIdx() * rowsPerCore;
    int64_t rEnd = rStart + rowsPerCore;
    if (rEnd > rows) rEnd = rows;
    if (rStart >= rEnd) return;

    const AscendC::BinaryRepeatParams rowAdd{1, 1, 1, 8, static_cast<uint8_t>(cols / 8), 8};
    const uint32_t rowBytes = static_cast<uint32_t>(cols * (int64_t)sizeof(T));
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t r0 = rStart; r0 < rEnd; r0 += R) {
        int64_t rh = rEnd - r0;
        if (rh > R) rh = R;
        const int32_t n = static_cast<int32_t>(rh * cols);
        const uint8_t reps = static_cast<uint8_t>(rh);

        auto src = qx.AllocTensor<T>();
        AscendC::DataCopyPad(src, xGm[r0 * cols],
                             AscendC::DataCopyExtParams{static_cast<uint16_t>(rh), rowBytes, 0, 0, 0}, padParams);
        qx.EnQue(src);
        src = qx.DeQue<T>();
        ToF32(xf, src, n);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        qx.FreeTensor(src);

        AscendC::Mul(sq, xf, xf, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(red, 0.0f, static_cast<int32_t>(R * 64));
        AscendC::PipeBarrier<PIPE_V>();
        int64_t c = 0;
        for (; c + 64 <= cols; c += 64) {
            AscendC::Add(red, sq[c], red, 64, reps, rowAdd);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (c < cols) {
            AscendC::Add(red, sq[c], red, static_cast<int32_t>(cols - c), reps, rowAdd);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // One sum per row; the result is packed contiguously in rstd[0..rh).
        AscendC::WholeReduceSum<float, false>(rstd, red, 64, reps, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(rstd, rstd, 1.0f / static_cast<float>(cols), static_cast<int32_t>(rh));
        AscendC::Adds(rstd, rstd, eps, static_cast<int32_t>(rh));
        AscendC::Sqrt(rstd, rstd, static_cast<int32_t>(rh));
        AscendC::Duplicate(red, 1.0f, static_cast<int32_t>(rh));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(rstd, red, rstd, static_cast<int32_t>(rh));
        KAG_V_S();
        if (useBcast) {
            // A/B alternative: expand [rh,1] -> [rh,cols] once, then one contiguous Mul.
            uint32_t dstShape[2] = {static_cast<uint32_t>(rh), static_cast<uint32_t>(cols)};
            uint32_t srcShape[2] = {static_cast<uint32_t>(rh), 1};
            AscendC::BroadCast<float, 2, 1>(sq, rstd, dstShape, srcShape);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(xf, xf, sq, static_cast<int32_t>(rh * cols));
        } else {
            for (int64_t i = 0; i < rh; ++i) {
                float rv = rstd.GetValue(static_cast<int32_t>(i));
                KAG_S_V();
                AscendC::Muls(xf[i * cols], xf[i * cols], rv, static_cast<int32_t>(cols));
            }
        }
        // gamma is identical for every row: one strided Mul per 64-column group reuses
        // the same gamma block across rows (src1RepStride=0). For tall tiles this
        // replaces rh per-row instructions with cols/64.
        if (rh > cols / 64) {
            const uint8_t rowStride = static_cast<uint8_t>(cols / 8);
            const AscendC::BinaryRepeatParams gamRow{1, 1, 1, rowStride, rowStride, 0};
            int64_t c = 0;
            for (; c + 64 <= cols; c += 64) {
                AscendC::Mul(xf[c], xf[c], gam[c], 64, reps, gamRow);
            }
            if (c < cols) {
                AscendC::Mul(xf[c], xf[c], gam[c], static_cast<int32_t>(cols - c), reps, gamRow);
            }
        } else {
            for (int64_t i = 0; i < rh; ++i) {
                AscendC::Mul(xf[i * cols], xf[i * cols], gam, static_cast<int32_t>(cols));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        auto dst = qy.AllocTensor<T>();
        FromF32(dst, xf, n);
        qy.EnQue(dst);
        dst = qy.DeQue<T>();
        AscendC::DataCopyPad(yGm[r0 * cols], dst,
                             AscendC::DataCopyExtParams{static_cast<uint16_t>(rh), rowBytes, 0, 0, 0});
        qy.FreeTensor(dst);
    }
}

__aicore__ inline int32_t KagFindPow2(int32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// In-place tree reduction: no cols-sized work buffer, so a full 8192-column row fits
// in UB. The order matches the official ReduceSumHalfInterval helper.
__aicore__ inline float KagReduceSumInPlace(AscendC::LocalTensor<float> &src,
                                            AscendC::LocalTensor<float> &sm, int32_t count)
{
    int32_t body = KagFindPow2(count);
    int32_t tail = count - body;
    if (tail > 0) {
        AscendC::Add(src, src, src[body], tail);
        AscendC::PipeBarrier<PIPE_V>();
    }
    while (body > 64) {
        body >>= 1;
        AscendC::Add(src, src, src[body], body);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::WholeReduceSum<float, false>(sm, src, 64, 1, 1, 1, 8);
    KAG_V_S();
    return sm.GetValue(0);
}

// Single-pass RMSNorm for one large row that does not fit the v2 tile: load x once,
// keep the fp32 row in UB, reduce in place, normalize, store. 1-deep queues are what
// make cols up to ~8192 (16-bit) fit.
template <typename T>
__global__ __aicore__ void rms_norm4_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, int64_t rows, int64_t cols,
                                            int64_t tile, float eps)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qx, qg;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bRow, bSq, bG, bSm;
    const int64_t tBytes = ((cols * (int64_t)sizeof(T) + 31) / 32) * 32;
    const int64_t fBytes = ((cols * 4 + 31) / 32) * 32;
    pipe.InitBuffer(qx, 1, tBytes);
    pipe.InitBuffer(qg, 1, tBytes);
    pipe.InitBuffer(qy, 1, tBytes);
    pipe.InitBuffer(bRow, fBytes);
    pipe.InitBuffer(bSq, fBytes);
    pipe.InitBuffer(bG, fBytes);
    pipe.InitBuffer(bSm, 64);
    auto row = bRow.Get<float>();
    auto sq = bSq.Get<float>();
    auto gam = bG.Get<float>();
    auto sm = bSm.Get<float>();

    AscendC::GlobalTensor<T> xGm, gGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    gGm.SetGlobalBuffer((__gm__ T *)gamma);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    auto g = qg.AllocTensor<T>();
    AscendC::DataCopyPad(g, gGm[0],
                         AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cols * (int64_t)sizeof(T)), 0, 0, 0},
                         padParams);
    qg.EnQue(g);
    g = qg.DeQue<T>();
    ToF32(gam, g, static_cast<int32_t>(cols));
    qg.FreeTensor(g);
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        auto src = qx.AllocTensor<T>();
        AscendC::DataCopyPad(src, xGm[r * cols],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cols * (int64_t)sizeof(T)), 0, 0, 0},
                             padParams);
        qx.EnQue(src);
        src = qx.DeQue<T>();
        ToF32(row, src, static_cast<int32_t>(cols));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        qx.FreeTensor(src);

        AscendC::Mul(sq, row, row, static_cast<int32_t>(cols));
        AscendC::PipeBarrier<PIPE_V>();
        const float ss = KagReduceSumInPlace(sq, sm, static_cast<int32_t>(cols));
        const float mean = ss / static_cast<float>(cols);
        for (int i = 0; i < 8; ++i) sm.SetValue(i, mean + eps);
        KAG_S_V();
        AscendC::Sqrt(sm, sm, 8);
        KAG_V_S();
        const float rstd = 1.0f / sm.GetValue(0);
        AscendC::Muls(row, row, rstd, static_cast<int32_t>(cols));
        AscendC::Mul(row, row, gam, static_cast<int32_t>(cols));

        auto dst = qy.AllocTensor<T>();
        FromF32(dst, row, static_cast<int32_t>(cols));
        qy.EnQue(dst);
        dst = qy.DeQue<T>();
        AscendC::DataCopyPad(yGm[r * cols], dst,
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cols * (int64_t)sizeof(T)), 0, 0, 0});
        // 1-deep out queue: wait for the store to finish reading UB before reuse.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        qy.FreeTensor(dst);
    }
}

// ---------------------------------------------------------------- rms norm
template <typename T>
__global__ __aicore__ void rms_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, int64_t rows, int64_t cols,
                                           int64_t tile, float eps)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx, qg;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3, b4;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qg, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    const int64_t scratch = (cols < tile) ? cols : tile;
    const int64_t rowElems = ((scratch + 7) / 8) * 8;
    const int64_t rowBytes = (((rowElems * sizeof(float)) + 31) / 32) * 32;
    pipe.InitBuffer(b0, rowBytes);   // cached float row
    pipe.InitBuffer(b1, rowBytes);   // gamma float copy
    pipe.InitBuffer(b2, rowBytes);   // ReduceSum work buffer
    pipe.InitBuffer(b3, rowBytes);   // temp
    pipe.InitBuffer(b4, 64);         // scalar math
    auto row = b0.Get<float>();
    auto gam = b1.Get<float>();
    auto work = b2.Get<float>();
    auto tmp = b3.Get<float>();
    auto sm = b4.Get<float>();

    AscendC::GlobalTensor<T> xGm, gGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    gGm.SetGlobalBuffer((__gm__ T *)gamma);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    const bool cacheRow = (cols <= tile);
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        float ss = 0.0f;
        if (cacheRow) {
            int32_t n = static_cast<int32_t>(cols);
            auto src = qx.AllocTensor<T>();
            AscendC::DataCopyPad(src, xGm[r * cols],
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            auto g = qg.AllocTensor<T>();
            AscendC::DataCopyPad(g, gGm[0],
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            qx.EnQue(src);
            qg.EnQue(g);
            src = qx.DeQue<T>();
            g = qg.DeQue<T>();
            ToF32(row, src, n);
            ToF32(gam, g, n);
            qx.FreeTensor(src);
            qg.FreeTensor(g);
            AscendC::Mul(tmp, row, row, n);
            AscendC::ReduceSum<float>(sm, tmp, work, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            ss = sm.GetValue(0);
        } else {
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(row, src, n);
                qx.FreeTensor(src);
                AscendC::Mul(tmp, row, row, n);
                AscendC::ReduceSum<float>(sm, tmp, work, n);
                AscendC::PipeBarrier<PIPE_ALL>();
                ss += sm.GetValue(0);
            }
        }
        const float mean = ss / static_cast<float>(cols);
        for (int i = 0; i < 8; ++i) sm.SetValue(i, mean + eps);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sqrt(sm, sm, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float rstd = 1.0f / sm.GetValue(0);
        if (cacheRow) {
            int32_t n = static_cast<int32_t>(cols);
            AscendC::Muls(row, row, rstd, n);
            AscendC::Mul(row, row, gam, n);
            auto dst = qy.AllocTensor<T>();
            FromF32(dst, row, n);
            qy.EnQue(dst);
            dst = qy.DeQue<T>();
            AscendC::DataCopyPad(yGm[r * cols], dst,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
            qy.FreeTensor(dst);
        } else {
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                auto g = qg.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                AscendC::DataCopyPad(g, gGm[off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                qg.EnQue(g);
                src = qx.DeQue<T>();
                g = qg.DeQue<T>();
                ToF32(row, src, n);
                ToF32(gam, g, n);
                qx.FreeTensor(src);
                qg.FreeTensor(g);
                AscendC::Muls(row, row, rstd, n);
                AscendC::Mul(row, row, gam, n);
                auto dst = qy.AllocTensor<T>();
                FromF32(dst, row, n);
                qy.EnQue(dst);
                dst = qy.DeQue<T>();
                AscendC::DataCopyPad(yGm[r * cols + off], dst,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
                qy.FreeTensor(dst);
            }
        }
    }
}

// ---------------------------------------------------------------- argmax v2 (last dim)
// * one ReduceMax(calIndex=true) per tile + GetReduceRepeatMaxMinSpr gives value AND index, so the old
//   Compare/Select/second-ReduceMax sequence (3 vector ops + 3 barriers per tile) is gone
// * V_S event instead of PIPE_ALL, wider tile (calc_argmax_tiling)
template <typename T>
__global__ __aicore__ void argmax2_kernel(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile,
                                          int32_t useSpr)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3, b4, b5;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qo, kDepth, 32);
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b1, ((((8 * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b2, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b3, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b4, ((((tile * sizeof(half)) + 31) / 32) * 32));
    pipe.InitBuffer(b5, ((((tile * sizeof(float)) + 31) / 32) * 32));
    auto f0 = b0.Get<float>();
    auto red = b1.Get<float>();
    auto work = b2.Get<float>();     // sharedTmpBuffer for ReduceMax / Compare-Select fallback
    auto ramp = b3.Get<float>();
    auto mh = b4.Get<half>();
    auto zeroF = b5.Get<float>();
    {
        for (int64_t i = 0; i < tile; ++i) {
            ramp.SetValue(static_cast<int32_t>(i), static_cast<float>(tile - i));
        }
        KAG_S_V();
        AscendC::Duplicate(zeroF, 0.0f, static_cast<int32_t>(tile));
    }

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int64_t> oGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    oGm.SetGlobalBuffer((__gm__ int64_t *)idx);
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        int64_t bidx = 0;
        if constexpr (sizeof(T) == 8) {
            T best;
            int64_t bi = 0;
            bool has = false;
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                for (int32_t i = 0; i < n; ++i) {
                    T v = src.GetValue(i);
                    const int32_t vh = static_cast<int32_t>(static_cast<uint64_t>(v) >> 32);
                    const int32_t bh = static_cast<int32_t>(static_cast<uint64_t>(best) >> 32);
                    const int32_t vl = static_cast<int32_t>(static_cast<uint64_t>(v) & 0xFFFFFFFFu);
                    const int32_t bl = static_cast<int32_t>(static_cast<uint64_t>(best) & 0xFFFFFFFFu);
                    const bool better = (!has) || (vh > bh) || (vh == bh && vl > bl);
                    if (better) {
                        best = v;
                        bi = off + i;
                        has = true;
                    }
                }
                qx.FreeTensor(src);
            }
            bidx = bi;
        } else {
            float best = -3.0e38f;
            bool has = false;
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, n);
                qx.FreeTensor(src);
                {
                    AscendC::ReduceMax<float>(red, f0, work, n);
                    KAG_V_S();
                    const float v = red.GetValue(0);
                    if (!has || v > best) {
                        best = v;
                        AscendC::Compares(mh.template ReinterpretCast<uint8_t>(), f0, v, AscendC::CMPMODE::EQ,
                                          static_cast<uint32_t>(n));
                        AscendC::Select(work, mh.template ReinterpretCast<uint8_t>(), ramp, zeroF,
                                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(n));
                        AscendC::ReduceMax<float>(red, work, f0, n);
                        KAG_V_S();
                        const int32_t firstLocal = static_cast<int32_t>(tile) - static_cast<int32_t>(red.GetValue(0));
                        if (firstLocal >= 0 && firstLocal < n) {
                            bidx = off + firstLocal;
                        } else {
                            bidx = off;
                        }
                        has = true;
                    }
                }
            }
        }
        auto o = qo.AllocTensor<int64_t>();
        o.SetValue(0, bidx);
        KAG_S_V();
        qo.EnQue(o);
        o = qo.DeQue<int64_t>();
        AscendC::DataCopyPad(oGm[r], o, AscendC::DataCopyExtParams{1, 8u, 0, 0, 0});
        qo.FreeTensor(o);
    }
}

// ---------------------------------------------------------------- argmax (last dim)
template <typename T>
__global__ __aicore__ void argmax_kernel(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile,
                                         int64_t nd, int64_t dim, int64_t r0, int64_t r1, int64_t r2,
                                         int64_t r3, int64_t r4, int64_t r5, int64_t r6, int64_t r7,
                                         int64_t o0, int64_t o1, int64_t o2, int64_t o3, int64_t o4,
                                         int64_t o5, int64_t o6, int64_t o7)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3, b4, b5;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qo, kDepth, 32);
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b1, ((((8 * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b2, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b3, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b4, ((((tile * sizeof(half)) + 31) / 32) * 32));
    pipe.InitBuffer(b5, ((((tile * sizeof(float)) + 31) / 32) * 32));
    auto f0 = b0.Get<float>();
    auto red = b1.Get<float>();
    auto maskF = b2.Get<float>();
    auto ramp = b3.Get<float>();
    auto mh = b4.Get<half>();
    auto zeroF = b5.Get<float>();
    for (int64_t i = 0; i < tile; ++i) {
        ramp.SetValue(static_cast<int32_t>(i), static_cast<float>(tile - i));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Duplicate(zeroF, 0.0f, static_cast<int32_t>(tile));
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int64_t> oGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    oGm.SetGlobalBuffer((__gm__ int64_t *)idx);
    int64_t rsz[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
    int64_t ost[8] = {o0, o1, o2, o3, o4, o5, o6, o7};
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        // rows follow src.sizes()[:-1]; with a moved dim that order differs from
        // the (dim-removed) output order, so remap the destination slot here
        int64_t orow = r;
        if (dim != nd - 1) {
            int64_t rem = r;
            orow = 0;
            for (int k = static_cast<int>(nd) - 2; k >= 0; --k) {
                const int64_t c = rem % rsz[k];
                rem /= rsz[k];
                orow += c * ost[k];
            }
        }
        int64_t bidx = 0;
        if constexpr (sizeof(T) == 8) {
            T best;
            int64_t bi = 0;
            bool has = false;
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                for (int32_t i = 0; i < n; ++i) {
                    T v = src.GetValue(i);
                    const int32_t vh = static_cast<int32_t>(static_cast<uint64_t>(v) >> 32);
                    const int32_t bh = static_cast<int32_t>(static_cast<uint64_t>(best) >> 32);
                    const int32_t vl = static_cast<int32_t>(static_cast<uint64_t>(v) & 0xFFFFFFFFu);
                    const int32_t bl = static_cast<int32_t>(static_cast<uint64_t>(best) & 0xFFFFFFFFu);
                    const bool better = (!has) || (vh > bh) || (vh == bh && vl > bl);
                    if (better) {
                        best = v;
                        bi = off + i;
                        has = true;
                    }
                }
                qx.FreeTensor(src);
            }
            bidx = bi;
        } else {
            float best = -3.0e38f;
            bool has = false;
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, n);
                qx.FreeTensor(src);
                AscendC::ReduceMax<float>(red, f0, maskF, n);
                AscendC::PipeBarrier<PIPE_ALL>();
                const float v = red.GetValue(0);
                if (!has || v > best) {
                    best = v;
                    // first position equal to the max: mask ? (tile-index) : 0 -> max gives tile-first
                    AscendC::Compares(mh.template ReinterpretCast<uint8_t>(), f0, v, AscendC::CMPMODE::EQ,
                                      static_cast<uint32_t>(n));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Select(maskF, mh.template ReinterpretCast<uint8_t>(), ramp, zeroF,
                                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(n));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::ReduceMax<float>(red, maskF, f0, n);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    const int32_t firstLocal = static_cast<int32_t>(tile) - static_cast<int32_t>(red.GetValue(0));
                    if (firstLocal >= 0 && firstLocal < n) {
                        bidx = off + firstLocal;
                    } else {
                        bidx = off;
                    }
                    has = true;
                }
            }
        }
        auto o = qo.AllocTensor<int64_t>();
        o.SetValue(0, bidx);
        qo.EnQue(o);
        o = qo.DeQue<int64_t>();
        AscendC::DataCopyPad(oGm[orow], o, AscendC::DataCopyExtParams{1, 8u, 0, 0, 0});
        qo.FreeTensor(o);
    }
}


// argmax along an arbitrary axis: x is viewed as [O, K, S] with the reduction along K, so
// element (o,k,s) sits at ((o*K + k)*S + s) and each DMA is a contiguous run along S.
// Removes the torch transpose+contiguous copy that fails on the platform runner.
// dbg != 0 skips the data path and just records k, which isolates launch/writeback bugs.
template <typename T>
__global__ __aicore__ void argmax_axis_kernel(GM_ADDR x, GM_ADDR idx, int64_t O, int64_t K, int64_t S,
                                              int64_t tile, int64_t dbg)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3, b4, b5;
    const int64_t tbytes = ((((tile * (int64_t)sizeof(T)) + 31) / 32) * 32);
    const int64_t fbytes = ((((tile * (int64_t)sizeof(float)) + 31) / 32) * 32);
    pipe.InitBuffer(qx, kDepth, tbytes);
    pipe.InitBuffer(qo, kDepth, ((((tile * (int64_t)sizeof(int64_t)) + 31) / 32) * 32));
    pipe.InitBuffer(b0, fbytes);                                       // running max
    pipe.InitBuffer(b1, fbytes);                                       // current value
    pipe.InitBuffer(b2, fbytes);                                       // k broadcast
    pipe.InitBuffer(b3, fbytes);                                       // index accumulator
    pipe.InitBuffer(b4, ((((tile * sizeof(half)) + 31) / 32) * 32));    // compare mask
    pipe.InitBuffer(b5, fbytes);                                       // scratch
    auto fmax = b0.Get<float>();
    auto fval = b1.Get<float>();
    auto fk = b2.Get<float>();
    auto fidx = b3.Get<float>();
    auto mh = b4.Get<half>();
    auto ftmp = b5.Get<float>();

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int64_t> oGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    oGm.SetGlobalBuffer((__gm__ int64_t *)idx);

    const int64_t nchunk = (S + tile - 1) / tile;
    const int64_t items = O * nchunk;
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t it = AscendC::GetBlockIdx(); it < items; it += nb) {
        const int64_t o = it / nchunk;
        const int64_t s0 = (it % nchunk) * tile;
        const int64_t cnt = ((S - s0) < tile) ? (S - s0) : tile;
        const int32_t n = static_cast<int32_t>(cnt);
        const uint32_t nbytes = static_cast<uint32_t>(cnt * (int64_t)sizeof(T));
        const int64_t base = o * K * S + s0;
        if (dbg == 3) {
            // vector-only smoke test: one Duplicate, then write it back
            AscendC::Duplicate(fidx, 7.0f, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            auto ot3 = qo.AllocTensor<int64_t>();
            for (int32_t i = 0; i < n; ++i) ot3.SetValue(i, static_cast<int64_t>(fidx.GetValue(i)));
            AscendC::PipeBarrier<PIPE_ALL>();
            qo.EnQue(ot3);
            ot3 = qo.DeQue<int64_t>();
            AscendC::DataCopyPad(oGm[o * S + s0], ot3,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * 8), 0, 0, 0});
            qo.FreeTensor(ot3);
            continue;
        }
        if (dbg == 2) {
            // write-only smoke test: no UB math, no DMA in -- isolates launch/params/writeback
            auto ot0 = qo.AllocTensor<int64_t>();
            for (int32_t i = 0; i < n; ++i) ot0.SetValue(i, 42);
            AscendC::PipeBarrier<PIPE_ALL>();
            qo.EnQue(ot0);
            ot0 = qo.DeQue<int64_t>();
            AscendC::DataCopyPad(oGm[o * S + s0], ot0,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * 8), 0, 0, 0});
            qo.FreeTensor(ot0);
            continue;
        }
        AscendC::Duplicate(fmax, -3.0e38f, n);
        AscendC::Duplicate(fidx, 0.0f, n);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t k = 0; k < K; ++k) {
            if (dbg != 0) {
                // write-only probe: record k without touching GM
                AscendC::Duplicate(fk, static_cast<float>(k), n);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Adds(fidx, fk, 0.0f, n);
                AscendC::PipeBarrier<PIPE_ALL>();
                continue;
            }
            auto raw = qx.AllocTensor<T>();
            AscendC::DataCopyPad(raw, xGm[base + k * S], AscendC::DataCopyExtParams{1, nbytes, 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            qx.EnQue(raw);
            raw = qx.DeQue<T>();
            ToF32(fval, raw, n);
            qx.FreeTensor(raw);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (k == 0) {
                AscendC::Adds(fmax, fval, 0.0f, n);
                AscendC::PipeBarrier<PIPE_ALL>();
                continue;
            }
            // strictly greater replaces the running max, so the first occurrence wins.
            // Compare(fval, fmax) is unreliable here, so diff first and compare against 0.
            AscendC::Sub(ftmp, fval, fmax, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Compares(mh.template ReinterpretCast<uint8_t>(), ftmp, 0.0f, AscendC::CMPMODE::GT, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate(fk, static_cast<float>(k), n);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Select(ftmp, mh.template ReinterpretCast<uint8_t>(), fk, fidx,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(n));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Adds(fidx, ftmp, 0.0f, n);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Max(fmax, fmax, fval, n);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        auto ot = qo.AllocTensor<int64_t>();
        for (int32_t i = 0; i < n; ++i) {
            ot.SetValue(i, static_cast<int64_t>(fidx.GetValue(i)));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        qo.EnQue(ot);
        ot = qo.DeQue<int64_t>();
        AscendC::DataCopyPad(oGm[o * S + s0], ot,
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * 8), 0, 0, 0});
        qo.FreeTensor(ot);
    }
}

// ---------------------------------------------------------------- cummin (last dim)
template <typename T>
__global__ __aicore__ void cummin_kernel(GM_ADDR x, GM_ADDR vals, GM_ADDR idx, int64_t rows, int64_t cols,
                                         int64_t tile)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qv, qi;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qv, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qi, kDepth, ((((tile * sizeof(int64_t)) + 31) / 32) * 32));
    AscendC::GlobalTensor<T> xGm, vGm;
    AscendC::GlobalTensor<int64_t> iGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    vGm.SetGlobalBuffer((__gm__ T *)vals);
    iGm.SetGlobalBuffer((__gm__ int64_t *)idx);
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t r = AscendC::GetBlockIdx(); r < rows; r += nb) {
        T cur;
        float curF = 0.0f;
        int64_t cidx = 0;
        bool has = false;
        for (int64_t off = 0; off < cols; off += tile) {
            int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
            auto src = qx.AllocTensor<T>();
            AscendC::DataCopyPad(src, xGm[r * cols + off],
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            qx.EnQue(src);
            src = qx.DeQue<T>();
            auto vo = qv.AllocTensor<T>();
            auto io = qi.AllocTensor<int64_t>();
            for (int32_t i = 0; i < n; ++i) {
                T v = src.GetValue(i);
                bool take;
                float vf = 0.0f;
                if constexpr (std::is_same<T, bfloat16_t>::value) {
                    vf = Bf16ToFloat(v);
                    take = (!has) || (vf <= curF);
                    if (take) curF = vf;
                } else if constexpr (sizeof(T) <= 2) {
                    vf = static_cast<float>(v);
                    take = (!has) || (vf <= curF);
                    if (take) curF = vf;
                } else {
                    take = (!has) || (v <= cur);
                    if (take) cur = v;
                }
                if (take) {
                    cidx = off + i;
                    has = true;
                }
                if constexpr (std::is_same<T, bfloat16_t>::value) {
                    vo.SetValue(i, FloatToBf16(curF));
                } else if constexpr (sizeof(T) <= 2) {
                    vo.SetValue(i, static_cast<T>(curF));
                } else {
                    vo.SetValue(i, cur);
                }
                io.SetValue(i, cidx);
            }
            qx.FreeTensor(src);
            qv.EnQue(vo);
            qi.EnQue(io);
            vo = qv.DeQue<T>();
            io = qi.DeQue<int64_t>();
            AscendC::DataCopyPad(vGm[r * cols + off], vo,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0});
            AscendC::DataCopyPad(iGm[r * cols + off], io,
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * 8), 0, 0, 0});
            qv.FreeTensor(vo);
            qi.FreeTensor(io);
        }
    }
}

// ---------------------------------------------------------------- cross entropy
template <typename T, typename TI>
__global__ __aicore__ void ce_kernel(GM_ADDR x, GM_ADDR target, GM_ADDR lossF32, int64_t rows, int64_t cols,
                                     int64_t tile, int32_t ignoreIndex)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qt;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> ql;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3;
    constexpr int64_t kRowBlock = 64;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qt, 2, ((((kRowBlock * sizeof(int64_t)) + 31) / 32) * 32));
    pipe.InitBuffer(ql, 2, ((((kRowBlock * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b1, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b2, ((((8 * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b3, ((((tile * sizeof(float)) + 31) / 32) * 32));
    auto f0 = b0.Get<float>();
    auto f1 = b1.Get<float>();
    auto red = b2.Get<float>();
    auto f3 = b3.Get<float>();

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<TI> tGm;
    AscendC::GlobalTensor<float> lGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    tGm.SetGlobalBuffer((__gm__ TI *)target);
    lGm.SetGlobalBuffer((__gm__ float *)lossF32);

    const int64_t nb = AscendC::GetBlockNum();
    const int64_t nBlocks = (rows + kRowBlock - 1) / kRowBlock;
    for (int64_t blk = AscendC::GetBlockIdx(); blk < nBlocks; blk += nb) {
        const int64_t r0 = blk * kRowBlock;
        const int64_t r1 = (r0 + kRowBlock < rows) ? (r0 + kRowBlock) : rows;
        const int32_t tn = static_cast<int32_t>(r1 - r0);
        auto lt = qt.AllocTensor<int64_t>();
        // TI is int64 for all wired instantiations; copy the block as int64.
        AscendC::DataCopyPad(lt, tGm[r0],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(tn * 8), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        qt.EnQue(lt);
        lt = qt.DeQue<int64_t>();
        auto lo = ql.AllocTensor<float>();
        for (int64_t r = r0; r < r1; ++r) {
            int64_t t = lt.GetValue(static_cast<int32_t>(r - r0));
            if (t < 0 || t == static_cast<int64_t>(ignoreIndex)) {
                lo.SetValue(static_cast<int32_t>(r - r0), 0.0f);
                continue;
            }
            float mx = -3.0e38f;
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, n);
                qx.FreeTensor(src);
                AscendC::ReduceMax<float>(red, f0, f3, n);
                AscendC::PipeBarrier<PIPE_ALL>();
                float v = red.GetValue(0);
                if (v > mx) mx = v;
            }
            float sum = 0.0f;
            float xTarget = 0.0f;
            for (int64_t off = 0; off < cols; off += tile) {
                int32_t n = static_cast<int32_t>((cols - off) < tile ? (cols - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[r * cols + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, n);
                qx.FreeTensor(src);
                if (t >= off && t < off + n) {
                    xTarget = f0.GetValue(static_cast<int32_t>(t - off));
                }
                AscendC::Adds(f0, f0, -mx, n);
                AscendC::Exp(f0, f0, n);
                AscendC::ReduceSum<float>(red, f0, f3, n);
                AscendC::PipeBarrier<PIPE_ALL>();
                sum += red.GetValue(0);
            }
            float loss = mx + VecLn1(f3, sum) - xTarget;
            lo.SetValue(static_cast<int32_t>(r - r0), loss);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        ql.EnQue(lo);
        lo = ql.DeQue<float>();
        AscendC::DataCopyPad(lGm[r0], lo, AscendC::DataCopyExtParams{1,
                             static_cast<uint32_t>(tn * (int64_t)sizeof(float)), 0, 0, 0});
        ql.FreeTensor(lo);
        qt.FreeTensor(lt);
    }
}

__global__ __aicore__ void ce_reduce_kernel(GM_ADDR lossF32, GM_ADDR target, GM_ADDR out, int64_t rows,
                                            int32_t ignoreIndex, int32_t reduceMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qt, ql;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bRed, bWk;
    constexpr int64_t kRB = 64;
    pipe.InitBuffer(qt, 2, ((((kRB * sizeof(int64_t)) + 31) / 32) * 32));
    pipe.InitBuffer(ql, 2, ((((kRB * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(qo, 2, 32);
    pipe.InitBuffer(bRed, 32);
    pipe.InitBuffer(bWk, ((((kRB * sizeof(float)) + 31) / 32) * 32));
    AscendC::GlobalTensor<float> lGm;
    AscendC::GlobalTensor<int64_t> tGm;
    AscendC::GlobalTensor<float> oGm;
    lGm.SetGlobalBuffer((__gm__ float *)lossF32);
    tGm.SetGlobalBuffer((__gm__ int64_t *)target);
    oGm.SetGlobalBuffer((__gm__ float *)out);
    auto red = bRed.Get<float>();
    auto fwk = bWk.Get<float>();
    float sum = 0.0f;
    int64_t cnt = 0;
    for (int64_t r0 = 0; r0 < rows; r0 += kRB) {
        const int64_t r1 = (r0 + kRB < rows) ? (r0 + kRB) : rows;
        const int32_t tn = static_cast<int32_t>(r1 - r0);
        auto lt = qt.AllocTensor<int64_t>();
        AscendC::DataCopyPad(lt, tGm[r0], AscendC::DataCopyExtParams{1, static_cast<uint32_t>(tn * 8), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        qt.EnQue(lt);
        lt = qt.DeQue<int64_t>();
        auto lb = ql.AllocTensor<float>();
        AscendC::DataCopyPad(lb, lGm[r0], AscendC::DataCopyExtParams{1,
                             static_cast<uint32_t>(tn * (int64_t)sizeof(float)), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        ql.EnQue(lb);
        lb = ql.DeQue<float>();
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < tn; ++i) {
            const int64_t t = lt.GetValue(i);
            if (t < 0 || t == static_cast<int64_t>(ignoreIndex)) {
                lb.SetValue(i, 0.0f);          // ignored rows contribute nothing
            } else {
                cnt += 1;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::ReduceSum<float>(red, lb, fwk, tn);
        AscendC::PipeBarrier<PIPE_ALL>();
        sum += red.GetValue(0);
        ql.FreeTensor(lb);
        qt.FreeTensor(lt);
    }
    const float v = (reduceMode == 1) ? (cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f) : sum;
    auto ob = qo.AllocTensor<float>();
    ob.SetValue(0, v);
    AscendC::PipeBarrier<PIPE_ALL>();
    qo.EnQue(ob);
    ob = qo.DeQue<float>();
    AscendC::DataCopyPad(oGm[0], ob, AscendC::DataCopyExtParams{1, 4u, 0, 0, 0});
    qo.FreeTensor(ob);
}


// ---------------------------------------------------------------- group norm
template <typename T>
__global__ __aicore__ void group_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, int64_t batch,
                                             int64_t channels, int64_t groups, int64_t spatial, int64_t tile,
                                             float eps)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kDepth> qx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kDepth> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2, b3;
    pipe.InitBuffer(qx, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(qy, kDepth, ((((tile * sizeof(T)) + 31) / 32) * 32));
    pipe.InitBuffer(b0, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b1, ((((8 * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b2, ((((tile * sizeof(float)) + 31) / 32) * 32));
    pipe.InitBuffer(b3, ((((8 * sizeof(float)) + 31) / 32) * 32));
    auto f0 = b0.Get<float>();
    auto red = b1.Get<float>();
    auto tmpf = b2.Get<float>();
    auto tmp2 = b3.Get<float>();

    AscendC::GlobalTensor<T> xGm, yGm;
    AscendC::GlobalTensor<float> gGm, bGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    gGm.SetGlobalBuffer((__gm__ float *)gamma);
    bGm.SetGlobalBuffer((__gm__ float *)beta);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    const int64_t cg = channels / groups;
    const int64_t totalGroups = batch * groups;
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t g = AscendC::GetBlockIdx(); g < totalGroups; g += nb) {
        const int64_t n = g / groups;
        const int64_t gi = g % groups;
        const int64_t c0 = gi * cg;
        float s1 = 0.0f;
        for (int64_t c = c0; c < c0 + cg; ++c) {
            const int64_t rowBase = (n * channels + c) * spatial;
            for (int64_t off = 0; off < spatial; off += tile) {
                int32_t nn = static_cast<int32_t>((spatial - off) < tile ? (spatial - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[rowBase + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(nn * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, nn);
                qx.FreeTensor(src);
                AscendC::ReduceSum<float>(red, f0, tmp2, nn);
                AscendC::PipeBarrier<PIPE_ALL>();
                s1 += red.GetValue(0);
            }
        }
        const float cnt = static_cast<float>(cg * spatial);
        const float mean = s1 / cnt;
        float s2 = 0.0f;
        for (int64_t c = c0; c < c0 + cg; ++c) {
            const int64_t rowBase = (n * channels + c) * spatial;
            for (int64_t off = 0; off < spatial; off += tile) {
                int32_t nn = static_cast<int32_t>((spatial - off) < tile ? (spatial - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[rowBase + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(nn * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, nn);
                qx.FreeTensor(src);
                AscendC::Adds(f0, f0, -mean, nn);
                AscendC::Mul(tmpf, f0, f0, nn);
                AscendC::ReduceSum<float>(red, tmpf, tmp2, nn);
                AscendC::PipeBarrier<PIPE_ALL>();
                s2 += red.GetValue(0);
            }
        }
        const float var = s2 / cnt;
        const float rstd = 1.0f / VecSqrt1(tmp2, var + eps);
        for (int64_t c = c0; c < c0 + cg; ++c) {
            const float ga = static_cast<float>(gGm.GetValue(c));
            const float be = static_cast<float>(bGm.GetValue(c));
            const int64_t rowBase = (n * channels + c) * spatial;
            for (int64_t off = 0; off < spatial; off += tile) {
                int32_t nn = static_cast<int32_t>((spatial - off) < tile ? (spatial - off) : tile);
                auto src = qx.AllocTensor<T>();
                AscendC::DataCopyPad(src, xGm[rowBase + off],
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(nn * (int64_t)sizeof(T)), 0, 0, 0},
                                     AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
                qx.EnQue(src);
                src = qx.DeQue<T>();
                ToF32(f0, src, nn);
                qx.FreeTensor(src);
                AscendC::Adds(f0, f0, -mean, nn);
                AscendC::Muls(f0, f0, rstd * ga, nn);
                AscendC::Adds(f0, f0, be, nn);
                auto dst = qy.AllocTensor<T>();
                FromF32(dst, f0, nn);
                qy.EnQue(dst);
                dst = qy.DeQue<T>();
                AscendC::DataCopyPad(yGm[rowBase + off], dst,
                                     AscendC::DataCopyExtParams{1, static_cast<uint32_t>(nn * (int64_t)sizeof(T)), 0, 0, 0});
                qy.FreeTensor(dst);
            }
        }
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

std::tuple<int64_t, int64_t, int64_t> calc_row_tiling2(int64_t cols, int64_t elemBytes)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub * 9 / 10;
    // qx(2) + qg(2) + qy(2) x elemBytes  +  4 float scratch arrays (row, gamma, x*x, reduce work) + 64B
    int64_t perElem = 6 * elemBytes + 16;
    int64_t tile = static_cast<int64_t>(ub) / perElem;
    tile = (tile / 8) * 8;
    if (tile > cols) tile = (cols + 7) / 8 * 8;
    if (tile < 64) tile = 64;
    return std::make_tuple(tile, tile, tile);
}

// Multi-row budget for rms_norm3: unlike calc_row_tiling2 this is NOT clamped to one row.
// The kernel turns it into R = tile / cols rows per UB tile.
std::tuple<int64_t, int64_t, int64_t> calc_row_tiling3(int64_t cols, int64_t elemBytes)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub * 9 / 10;
    // v3 memory per element: qx+qg+qy (1 deep) = 3*es, row+sq = 8, reduce temp
    // R*64 floats = 256/cols per element, plus one gamma/qg row amortised over R>=2.
    int64_t redPerElem = (cols >= 64) ? 4 : (256 + cols - 1) / cols;
    int64_t perElem = 3 * elemBytes + 8 + redPerElem + (elemBytes + 4) / 2;
    int64_t tile = static_cast<int64_t>(ub) / perElem;
    tile = (tile / 8) * 8;
    if (tile < cols) tile = (cols + 7) / 8 * 8;
    if (tile < 64) tile = 64;
    return std::make_tuple(tile, tile, tile);
}

// rms_norm4 keeps one full row resident: qx + qg + qy (1 deep) + row/sq/gamma fp32.
bool rms_full_row_fits(int64_t cols, int64_t elemBytes)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    int64_t need = cols * (3 * elemBytes + 12) + 128;
    return need <= static_cast<int64_t>(ub * 9 / 10);
}

std::tuple<int64_t, int64_t, int64_t> calc_argmax_tiling(int64_t cols, int64_t elemBytes)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub * 9 / 10;
    // qx x2 deep + 4 float arrays (f0, reduce work, ramp, zeros) + half mask
    int64_t perElem = 2 * elemBytes + 4 * 4 + 2;
    int64_t tile = static_cast<int64_t>(ub) / perElem;
    tile = (tile / 8) * 8;
    if (tile > cols) tile = (cols + 7) / 8 * 8;
    if (tile < 256) tile = 256;
    return std::make_tuple(tile, tile, tile);
}

// dynamic_quant: unlike calc_row_tiling this budgets the fp32 row cache (cols*4) so that a whole
// row can stay resident in UB and be quantised without a second GM read.
// per element: qx(2)*es + qy(2)*1 + f0(4) + f1(4) + reduce work(4) + i32(4) + half(2)
std::tuple<int64_t, int64_t, int64_t> calc_dq_tiling(int64_t cols, int64_t elemBytes)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    int64_t perElem = 2 * elemBytes + 2 + 4 * 4 + 2;
    int64_t budget = static_cast<int64_t>(ub) - static_cast<int64_t>(ub) / 8;
    if (budget < 4096) budget = static_cast<int64_t>(ub) * 3 / 4;
    int64_t tile = budget / perElem;
    tile = (tile / 64) * 64;
    if (tile > cols) tile = (cols + 63) / 64 * 64;
    if (tile < 64) tile = 64;
    return std::make_tuple(tile, tile, tile);
}

std::tuple<int64_t, int64_t, int64_t> calc_row_tiling(int64_t cols, int64_t elemBytes)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub * 3 / 5;
    int64_t perElem = 8 * elemBytes + 32;
    int64_t tile = static_cast<int64_t>(ub) / perElem;
    tile = (tile / 64) * 64;
    if (tile < 256) tile = 256;
    if (tile > cols) tile = (cols + 63) / 64 * 64;
    if (tile < 64) tile = 64;
    return std::make_tuple(tile, tile, tile);
}

extern "C" {
#define ROW_LAUNCH_DEF(NAME, TN)                                                                          \
    void launch_##NAME##_##TN(GM_ADDR x, GM_ADDR y, GM_ADDR s, int64_t rows, int64_t cols, int64_t tile,   \
                              void *stream)
#define DQ_LAUNCH_DEF(SUF, TN)                                                                             \
    void launch_dynamic_quant_##SUF(GM_ADDR x, GM_ADDR y, GM_ADDR s, int64_t rows, int64_t cols,             \
                                    int64_t tile, int32_t flags, void *stream)                               \
    {                                                                                                        \
        dynamic_quant_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, y, s, rows, cols, tile, flags);     \
    }
DQ_LAUNCH_DEF(bf16, bfloat16_t)
DQ_LAUNCH_DEF(float, float)
DQ_LAUNCH_DEF(half, half)
#undef DQ_LAUNCH_DEF
#undef ROW_LAUNCH_DEF

#define RMS_LAUNCH_DEF(SUF, TN)                                                                              \
    void launch_rms_norm_##SUF(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile,       \
                              float eps, void *stream)                                                        \
    {                                                                                                         \
        rms_norm_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, g, y, rows, cols, tile, eps);                          \
    }
RMS_LAUNCH_DEF(bf16, bfloat16_t)
RMS_LAUNCH_DEF(float, float)
RMS_LAUNCH_DEF(half, half)
#undef RMS_LAUNCH_DEF

#define RMS2_LAUNCH_DEF(SUF, TN)                                                                             \
    void launch_rms_norm2_##SUF(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile,      \
                                float eps, void *stream)                                                       \
    {                                                                                                         \
        rms_norm2_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, g, y, rows, cols, tile, eps);                          \
    }
RMS2_LAUNCH_DEF(bf16, bfloat16_t)
RMS2_LAUNCH_DEF(float, float)
RMS2_LAUNCH_DEF(half, half)
#undef RMS2_LAUNCH_DEF

#define RMS3_LAUNCH_DEF(SUF, TN)                                                                             \
    void launch_rms_norm3_##SUF(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile,      \
                                float eps, int64_t useBcast, void *stream)                                     \
    {                                                                                                         \
        rms_norm3_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, g, y, rows, cols, tile, eps, useBcast);               \
    }
RMS3_LAUNCH_DEF(bf16, bfloat16_t)
RMS3_LAUNCH_DEF(float, float)
RMS3_LAUNCH_DEF(half, half)
#undef RMS3_LAUNCH_DEF

#define RMS4_LAUNCH_DEF(SUF, TN)                                                                             \
    void launch_rms_norm4_##SUF(GM_ADDR x, GM_ADDR g, GM_ADDR y, int64_t rows, int64_t cols, int64_t tile,      \
                                float eps, void *stream)                                                       \
    {                                                                                                         \
        rms_norm4_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, g, y, rows, cols, tile, eps);                          \
    }
RMS4_LAUNCH_DEF(bf16, bfloat16_t)
RMS4_LAUNCH_DEF(float, float)
RMS4_LAUNCH_DEF(half, half)
#undef RMS4_LAUNCH_DEF

#define AM2_LAUNCH_DEF(SUF, TN)                                                                              \
    void launch_argmax2_##SUF(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile, int32_t useSpr, \
                              void *stream)                                                                  \
    {                                                                                                        \
        argmax2_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, idx, rows, cols, tile, useSpr);                        \
    }
AM2_LAUNCH_DEF(bf16, bfloat16_t)
AM2_LAUNCH_DEF(float, float)
AM2_LAUNCH_DEF(half, half)
AM2_LAUNCH_DEF(i32, int32_t)
AM2_LAUNCH_DEF(i64, int64_t)
#undef AM2_LAUNCH_DEF

#define AMAX_AXIS_DEF(SUF, TN)                                                                                                    \
    void launch_argmax_axis_##SUF(GM_ADDR x, GM_ADDR idx, int64_t O, int64_t K, int64_t S, int64_t tile,\
                                  int64_t dbg, void *stream)\
    {                                                                                                    \
        argmax_axis_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, idx, O, K, S, tile, dbg);\
    }
AMAX_AXIS_DEF(bf16, bfloat16_t)
AMAX_AXIS_DEF(float, float)
AMAX_AXIS_DEF(half, half)
AMAX_AXIS_DEF(i32, int32_t)
#undef AMAX_AXIS_DEF

#define AM_LAUNCH_DEF(SUF, TN)                                                       \
    void launch_argmax_##SUF(GM_ADDR x, GM_ADDR idx, int64_t rows, int64_t cols, int64_t tile,\
                             int64_t nd, int64_t dim, int64_t r0, int64_t r1, int64_t r2, int64_t r3,\
                             int64_t r4, int64_t r5, int64_t r6, int64_t r7, int64_t o0, int64_t o1,\
                             int64_t o2, int64_t o3, int64_t o4, int64_t o5, int64_t o6, int64_t o7,\
                             void *stream)\
    {                                                                                                    \
        argmax_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, idx, rows, cols, tile, nd, dim,\
                                                                r0, r1, r2, r3, r4, r5, r6, r7,\
                                                                o0, o1, o2, o3, o4, o5, o6, o7);\
    }
AM_LAUNCH_DEF(bf16, bfloat16_t)
AM_LAUNCH_DEF(float, float)
AM_LAUNCH_DEF(half, half)
AM_LAUNCH_DEF(i32, int32_t)
AM_LAUNCH_DEF(i64, int64_t)
#undef AM_LAUNCH_DEF

#define CM_LAUNCH_DEF(SUF, TN)                                                                               \
    void launch_cummin_##SUF(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t cols, int64_t tile,        \
                              void *stream)                                                                    \
    {                                                                                                         \
        cummin_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, v, i, rows, cols, tile);                                 \
    }
CM_LAUNCH_DEF(bf16, bfloat16_t)
CM_LAUNCH_DEF(float, float)
CM_LAUNCH_DEF(half, half)
CM_LAUNCH_DEF(i32, int32_t)
#undef CM_LAUNCH_DEF

#define CE_LAUNCH_DEF(SUF, TT, TI)                                                                           \
    void launch_ce_##SUF##_int64_t(GM_ADDR x, GM_ADDR t, GM_ADDR l, int64_t rows, int64_t cols, int64_t tile, \
                                     int32_t ignoreIndex, void *stream)                                        \
    {                                                                                                         \
        ce_kernel<TT, TI><<<kag_block_num(), nullptr, stream>>>(x, t, l, rows, cols, tile, ignoreIndex);                    \
    }
CE_LAUNCH_DEF(bf16, bfloat16_t, int64_t)
CE_LAUNCH_DEF(float, float, int64_t)
CE_LAUNCH_DEF(half, half, int64_t)
#undef CE_LAUNCH_DEF

void launch_ce_reduce_f32(GM_ADDR loss, GM_ADDR target, GM_ADDR out, int64_t rows, int32_t ignoreIndex,
                          int32_t reduceMode, void *stream)
{
    ce_reduce_kernel<<<1, nullptr, stream>>>(loss, target, out, rows, ignoreIndex, reduceMode);
}

#define GN_LAUNCH_DEF(SUF, TN)                                                                               \
    void launch_group_norm_##SUF(GM_ADDR x, GM_ADDR g, GM_ADDR b, GM_ADDR y, int64_t batch, int64_t channels,      \
                              int64_t groups, int64_t spatial, int64_t tile, float eps, void *stream)           \
    {                                                                                                         \
        group_norm_kernel<TN><<<kag_block_num(), nullptr, stream>>>(x, g, b, y, batch, channels, groups, spatial, tile, eps); \
    }
GN_LAUNCH_DEF(bf16, bfloat16_t)
GN_LAUNCH_DEF(float, float)
GN_LAUNCH_DEF(half, half)
#undef GN_LAUNCH_DEF
}

// ---------------------------------------------------------------- f32 -> T cast (CE loss output)
// torch's ``.to(dtype)`` routes to an aclnn Cast/Copy kernel that is missing on the 950PR runner
// (copy_d2d_baseformat_opapi 561103), so CE casts its fp32 loss with our own DMA loop.
template <typename T>
__global__ __aicore__ void row_cast_kernel(GM_ADDR src, GM_ADDR dst, int64_t n)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bO;
    constexpr int64_t kTile = 4096;
    pipe.InitBuffer(qs, 2, kTile * 4);
    pipe.InitBuffer(bO, kTile * sizeof(T));
    auto ob = bO.Get<T>();
    AscendC::GlobalTensor<float> sg;
    AscendC::GlobalTensor<T> dg;
    sg.SetGlobalBuffer((__gm__ float *)src);
    dg.SetGlobalBuffer((__gm__ T *)dst);
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t off = AscendC::GetBlockIdx() * kTile; off < n; off += nb * kTile) {
        int64_t cnt = n - off;
        if (cnt > kTile) cnt = kTile;
        auto fb = qs.AllocTensor<float>();
        AscendC::DataCopyPad(fb, sg[off],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * 4), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        qs.EnQue(fb);
        fb = qs.DeQue<float>();
        FromF32<T>(ob, fb, static_cast<int32_t>(cnt));
        AscendC::PipeBarrier<PIPE_ALL>();
        qs.FreeTensor(fb);
        AscendC::DataCopyPad(dg[off], ob,
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * (int64_t)sizeof(T)), 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

extern "C" {
void launch_row_cast_f32f32(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream)
{
    row_cast_kernel<float><<<kag_block_num(), nullptr, stream>>>(src, dst, n);
}
void launch_row_cast_f32f16(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream)
{
    row_cast_kernel<half><<<kag_block_num(), nullptr, stream>>>(src, dst, n);
}
void launch_row_cast_f32bf16(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream)
{
    row_cast_kernel<bfloat16_t><<<kag_block_num(), nullptr, stream>>>(src, dst, n);
}
}

// f32 -> int32/int64 (USS output cast; sums are integral so RINT is exact)
template <typename TI>
__global__ __aicore__ void row_cast_i_kernel(GM_ADDR src, GM_ADDR dst, int64_t n)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bO;
    constexpr int64_t kTile = 2048;
    pipe.InitBuffer(qs, 2, kTile * 4);
    pipe.InitBuffer(bO, kTile * sizeof(TI));
    auto ob = bO.Get<TI>();
    AscendC::GlobalTensor<float> sg;
    AscendC::GlobalTensor<TI> dg;
    sg.SetGlobalBuffer((__gm__ float *)src);
    dg.SetGlobalBuffer((__gm__ TI *)dst);
    const int64_t nb = AscendC::GetBlockNum();
    for (int64_t off = AscendC::GetBlockIdx() * kTile; off < n; off += nb * kTile) {
        int64_t cnt = n - off;
        if (cnt > kTile) cnt = kTile;
        auto fb = qs.AllocTensor<float>();
        AscendC::DataCopyPad(fb, sg[off], AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * 4), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        qs.EnQue(fb);
        fb = qs.DeQue<float>();
        for (int64_t i = 0; i < cnt; ++i) {
            const float v = fb.GetValue(static_cast<int32_t>(i));
            ob.SetValue(static_cast<int32_t>(i), static_cast<TI>(v));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        qs.FreeTensor(fb);
        AscendC::DataCopyPad(dg[off], ob,
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cnt * (int64_t)sizeof(TI)), 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

extern "C" {
void launch_row_cast_f32i32(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream)
{
    row_cast_i_kernel<int32_t><<<kag_block_num(), nullptr, stream>>>(src, dst, n);
}
void launch_row_cast_f32i64(GM_ADDR src, GM_ADDR dst, int64_t n, void *stream)
{
    row_cast_i_kernel<int64_t><<<kag_block_num(), nullptr, stream>>>(src, dst, n);
}
}
