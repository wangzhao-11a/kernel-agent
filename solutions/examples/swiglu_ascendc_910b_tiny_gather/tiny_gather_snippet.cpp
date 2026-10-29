// Verified 910B tiny-row pattern (CANN 9.1.0, dav_c220).
// Use when half*sizeof(T) < 32B, outer is large, and the normal strided tile path
// degenerates to one element per DataCopyPad. 12/12 official cases, 2.96x aggregate;
// (1000003,2) bf16 0.011x -> 2.58x.
//
// Invariants:
// - Gather offsets are BYTE offsets: element_offset * sizeof(T).
// - chunk(2, dim) splits a row into [first half | second half], not even/odd.
// - Gather dst/src must share element type T; gather into T, then Cast to fp32.
// - inLocal, x0Local, x1Local must be distinct; in-place Gather corrupts the source.
// - Sync V->MTE2 before reusing the input buffer, MTE3->V before reusing the output.

template <typename T>
__global__ __aicore__ void swiglu_tiny_kernel(GM_ADDR x, GM_ADDR y, int64_t outer, int64_t half,
                                              int64_t tilesPerBlock, uint32_t tileBytes)
{
    AscendC::TPipe pipe;
    int64_t elemsPerTile = static_cast<int64_t>(tileBytes) / static_cast<int64_t>(sizeof(T));
    int64_t rowsPerTile = elemsPerTile / (2 * half);
    int64_t capRows = 4096 / half;
    if (capRows < 1) capRows = 1;
    if (capRows > 512) capRows = 512;
    if (rowsPerTile > capRows) rowsPerTile = capRows;
    if (rowsPerTile < 1) rowsPerTile = 1;

    int64_t outElems = rowsPerTile * half;
    uint32_t inBytes = static_cast<uint32_t>(rowsPerTile * 2 * half * sizeof(T));
    uint32_t outBytes = static_cast<uint32_t>(outElems * sizeof(T));
    uint32_t scratchBytes = static_cast<uint32_t>(outElems * sizeof(float));
    uint32_t offBytes = static_cast<uint32_t>(outElems * sizeof(uint32_t));

    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inX;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 2> xQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outY;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 2> tmpQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offBuf;
    pipe.InitBuffer(inX, 2, inBytes);
    pipe.InitBuffer(xQ, 2, outBytes);
    pipe.InitBuffer(outY, 2, outBytes);
    pipe.InitBuffer(tmpQ, 2, scratchBytes);
    pipe.InitBuffer(offBuf, 2 * offBytes);

    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalTiles = (outer + rowsPerTile - 1) / rowsPerTile;
    int64_t tStart = blockIdx * tilesPerBlock;
    int64_t tEnd = tStart + tilesPerBlock;
    if (tEnd > totalTiles) tEnd = totalTiles;
    if (tStart >= tEnd) return;

    AscendC::GlobalTensor<T> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    auto offT = offBuf.Get<uint32_t>();
    int64_t maxN = rowsPerTile * half;
    for (int64_t i = 0; i < maxN; ++i) {
        int64_t r = i / half;
        int64_t c = i - r * half;
        // byte offsets, first half / second half
        offT.SetValue(static_cast<uint32_t>(i),
                      static_cast<uint32_t>((r * 2 * half + c) * sizeof(T)));
        offT.SetValue(static_cast<uint32_t>(maxN + i),
                      static_cast<uint32_t>((r * 2 * half + half + c) * sizeof(T)));
    }
    auto off0 = offT[0];
    auto off1 = offT[maxN];

    for (int64_t t = tStart; t < tEnd; ++t) {
        int64_t r0 = t * rowsPerTile;
        int64_t rows = (rowsPerTile < outer - r0) ? rowsPerTile : (outer - r0);
        int64_t n = rows * half;
        uint32_t cnt = static_cast<uint32_t>(n);
        uint32_t nInBytes = static_cast<uint32_t>(rows * 2 * half * sizeof(T));
        uint32_t nOutBytes = static_cast<uint32_t>(n * sizeof(T));

        auto xLocal = inX.AllocTensor<T>();
        AscendC::DataCopyExtParams inParams{1, nInBytes, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[r0 * 2 * half], inParams, padParams);
        inX.EnQue(xLocal);
        xLocal = inX.DeQue<T>();

        auto x0 = xQ.AllocTensor<T>();
        auto x1 = xQ.AllocTensor<T>();
        AscendC::Gather(x0, xLocal, off0, 0, cnt);
        AscendC::Gather(x1, xLocal, off1, 0, cnt);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        inX.FreeTensor(xLocal);

        auto yLocal = outY.AllocTensor<T>();
        auto s0 = tmpQ.AllocTensor<float>();
        auto s1 = tmpQ.AllocTensor<float>();
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Muls(s1, x0, -1.0f, cnt);
            AscendC::Exp(s0, s1, cnt);
            AscendC::Adds(s1, s0, 1.0f, cnt);
            AscendC::Div(s0, x0, s1, cnt);
            AscendC::Mul(yLocal, s0, x1, cnt);
        } else {
            AscendC::Cast(s0, x0, AscendC::RoundMode::CAST_NONE, cnt);
            AscendC::Muls(s1, s0, -1.0f, cnt);
            AscendC::Exp(s1, s1, cnt);
            AscendC::Adds(s1, s1, 1.0f, cnt);
            AscendC::Div(s1, s0, s1, cnt);
            AscendC::Cast(s0, x1, AscendC::RoundMode::CAST_NONE, cnt);
            AscendC::Mul(s1, s1, s0, cnt);
            AscendC::Cast(yLocal, s1, AscendC::RoundMode::CAST_RINT, cnt);
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

// Host tiling: cap rows by 4096/half and by the UB budget.
std::tuple<int64_t, int64_t, uint32_t> calc_swiglu_tiny_tiling_params(
    int64_t outer, int64_t half, int64_t elemBytes)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = platform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    // 5*half*elemBytes: 2 in + 1 out + 2 gather(T); 16*half: 2 fp32 scratch + 2 offsets
    int64_t perRowBytes = 5 * half * elemBytes + 16 * half;
    int64_t rowsPerTile = static_cast<int64_t>(ubSize * 9 / 10) / perRowBytes;
    int64_t capRows = 4096 / half;
    if (capRows < 1) capRows = 1;
    if (capRows > 512) capRows = 512;
    if (rowsPerTile > capRows) rowsPerTile = capRows;
    if (rowsPerTile < 1) rowsPerTile = 1;
    int64_t totalTiles = (outer + rowsPerTile - 1) / rowsPerTile;
    int64_t numBlocks = std::min<int64_t>(coreNum, std::max<int64_t>(1, totalTiles));
    int64_t tilesPerBlock = (totalTiles + numBlocks - 1) / numBlocks;
    uint32_t tileBytes = static_cast<uint32_t>(rowsPerTile * 2 * half * elemBytes);
    return std::make_tuple(numBlocks, tilesPerBlock, tileBytes);
}

// Plugin dispatch (inside swi_glu_npu, before the tile path):
//   bool tinyPath = (!aligned) && (!simtPath || !SWIGLU_HAS_SIMT) && (rowBytes < 32) && (outer >= 4096);
//   if (tinyPath) { tie(tinyBlocks, tinyRowsPerBlock, tinyTileBytes) =
//       calc_swiglu_tiny_tiling_params(outer, half, x.element_size()); }
//   if (tinyPath) { launch_swiglu_tiny_<dtype>(x_ptr, y_ptr, outer, half,
//       tinyRowsPerBlock, tinyTileBytes, tinyBlocks, stream); return 0; }