# SwiGlu AscendC 910B: tiny-row Gather path

Verified on m18 (`deepinfer-real`, CANN 9.1.0, Ascend910B4, NPU 4) on 2026-10-23.

## Result

- Official cases: **12/12 PASS** (`tests/test_swiglu_official.py`)
- Aligned fp16/fp32 cases use `AscendC::SwiGLU`; bf16 uses a fused
  `Cast -> Silu<float> -> Mul -> Cast` path (no bf16 SwiGLU overload exists).
  A 3-run sample after this change had a **3.55x median**
  (3.546 / 3.599 / 2.798); best observed 3.71x. Baseline noise is significant.
- bf16 `(4096,8192)` profile after Silu fusion: duration 172.7us,
  **MTE2 0.756 / MTE3 0.716**, VEC 0.358, scalar 0.117 -> now memory-bound,
  not compute-bound.
- Tiny-row path after increasing `rowsPerTile` to 2048 and generating the Gather
  byte-offsets with vector ops (plus 128/64-element repeat alignment):
  - `(1000003, 2) dim=1 bf16`: kernel duration **28.4 -> ~16.4–16.7 us**
  - scalar ratio **0.874 -> ~0.39–0.42**; VEC ratio ~0.82, MTE2 ~0.13, MTE3 ~0.13
  - candidate ~0.028 ms, about 2.2–2.5x vs the PyTorch baseline
- Aggregate benchmark is still noisy; use the 3-run median and per-case/profile
  numbers rather than a single high run.

## Why a separate path

For `half * sizeof(T) < 32B`, the normal strided `DataCopyPad` tile path degenerates:
`R=1, W=1`, so it issues one DMA per output element. 910B (dav_c220) has no SIMT
(`SWIGLU_HAS_SIMT=0`), so the SIMT fallback is compiled out.

The tiny path instead:

1. copies `rowsPerTile` (up to 2048) rows as one contiguous input block (`blockCount=1`);
2. splits each row into its first/second `half` with `AscendC::Gather`;
3. for `half==1`, generates the byte offsets with `CreateVecIndex + Muls + Adds`
   instead of a scalar `SetValue` loop;
4. keeps `rowsPerTile * half` on the 128-element (16-bit) / 64-element (32-bit)
   Gather/vector repeat boundary — otherwise a 4095-element tail can raise
   "UB address accessed by VEC instruction is not aligned";
5. computes `silu(x0) * x1` in fp32 and casts on store;
6. writes the contiguous output block.

## Key API details (CANN 9.1.0)

- `Gather` offsets are **byte offsets**, not element indices. Multiply the element
  offset by `sizeof(T)`.
- `chunk(2, dim)` splits a row into `[first half | second half]`; it is **not**
  even/odd interleaving. For row `r`, column `c`:
  - x0 offset: `(r * 2 * half + c) * sizeof(T)`
  - x1 offset: `(r * 2 * half + half + c) * sizeof(T)`
- `Gather` requires `dst` and `src` to have the same element type `T`. Gather into
  `LocalTensor<T>` buffers, then cast to fp32; never gather directly into fp32.
- Use distinct `inLocal`, `x0Local`, `x1Local`; in-place Gather corrupts the source.
- `DeInterleave` is not implemented for dav_c220; `Gather` is.
- `AscendC::SwiGLU(dst, src0, src1, beta, count)` computes `swish(src1) * src0`.
  For `silu(x0) * x1`, pass **`src0=x1, src1=x0`**. The API is correct on
  32B-aligned slices on dav-2201, but not for bf16; keep the manual fp32 chain
  for bf16/non-aligned/tiny rows.
- `SWIGLU_HAS_ADV_API` is enabled only for `ascend910b` in CMake; 950 keeps the
  hand-written path until independently verified.

## Files

- `swiglu_kernel.cpp` — tile path + `swiglu_tiny_kernel` + tiling helpers
- `swiglu_launch.h` — launch declarations
- `swiglu_plugin.cpp` — host dispatch (`tinyPath` when `rowBytes < 32 && outer >= 4096`)