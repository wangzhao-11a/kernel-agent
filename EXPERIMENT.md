# P5 experiment — RSI on/off on held-out tasks

> This run is a **pilot**: model `deepseek-chat` via `openai`, device `npu`, temperature 0.2, max_iters 3, train budget 8 iters, seeds 2.

- started: 2026-11-05T08:29:41Z
- holdout tasks: softmax, matmul, mla_attention, moe_permute
- train tasks: vector_add, rmsnorm, decode_attention, grouped_matmul, moe_finalize_routing, rope, sdpa_attention, grouped_gemm_swiglu
- groups: kb,no_rsi
- holdout tasks were never in the training tasks root.

## Results

| group | n | first-pass iters (median, CI) | best speedup (median, CI) | tokens/gain | passes | cheat interceptions |
| --- | --- | --- | --- | --- | --- | --- |
| kb | 8 | 1.0 [1.0, 4.0] | 0.9534 [0.0, 11.9455] | 633200.0 | 5 | 0 |
| no_rsi | 8 | 1.0 [1.0, 1.0] | 0.9215 [0.7309, 12.1445] | 548250.0 | 8 | 0 |

## Paired comparisons (kb - no_rsi, per task x seed)

- pairs: 8
- first-pass improvement (no_rsi - kb; positive = RSI needs fewer iterations): 0.0 CI [-3.0, 0.0]
- best-speedup improvement (kb - no_rsi): -0.4448 CI [-0.8451, -0.046]
- tokens/gain difference: 44300.0 CI [35.3145, 204800.0]

## Pass lines

| line | required | observed | verdict |
| --- | --- | --- | --- |
| first-pass or best | at least one CI lower bound > 0 | first=False best=False | False |
| cost per gain | not worse than control | False | False |
| negative transfer | <= 1/4 of holdout tasks | ['matmul', 'mla_attention', 'moe_permute', 'softmax'] (rate 1.0) | False |

## Negative transfer by task

| task | kb best | no_rsi best | delta |
| --- | --- | --- | --- |
| matmul | 0.9534499999999999 | 1.0077500000000001 | -0.0543 |
| mla_attention | 0.0 | 0.7723 | -0.7723 |
| moe_permute | 11.95 | 12.520700000000001 | -0.5707 |
| softmax | 0.5005 | 0.7422500000000001 | -0.2418 |

## Limits

- The seed-only control is a seed pool, not token-length-matched; it is only run when requested.
- Bootstrap CIs resample task x seed pairs; a pilot has few pairs and wide intervals.
- This is a mechanism/scale-limited run unless the full protocol args are used.

## Conclusion (pilot)

This pilot **does not meet any P5 pass line**:

- best-speedup paired improvement `kb - no_rsi = -0.4448`, 95% CI `[-0.8451, -0.046]` — the RSI group was significantly worse;
- first-pass improvement `0.0`, 95% CI `[-3.0, 0.0]` — no iteration saving (kb was worse on softmax/MLA until failures);
- tokens per gain: kb worse (`+44,300` tokens/gain median difference);
- negative transfer: **4/4** holdout tasks (matmul, mla_attention, moe_permute, softmax);
- cheat interceptions: 0.

Cause and limits: the pilot trained for only **8 iterations** and produced **3 training runs** (`grouped_gemm_swiglu` ×2, `grouped_matmul` ×1); `kb@T` is correspondingly tiny. The holdout evaluation used 2 seeds and 3 max iterations, and the seed-only length-matched control was not run. This is a mechanism check, not a final verdict on RSI.

The full protocol (12 tasks / 4 holdout, 24+ training iterations, 5 seeds, 8 max iterations, seed-only length control, multi-stage learning curve) has **not** been run. The first thing to test in a larger run is whether a bigger `kb@T` removes the negative transfer seen here.

Raw data: [`data/p5/pilot1-results.json`](data/p5/pilot1-results.json). The API key is not stored in the repository or in these artifacts.
