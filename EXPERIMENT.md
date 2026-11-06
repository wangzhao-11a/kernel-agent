# P5 experiment — RSI on/off on held-out tasks

Two **real runs** on m18 / Ascend910B4-1 (NPU), model `deepseek-chat` via the DeepSeek OpenAI-compatible endpoint, temperature 0.2. Both use the same 4 holdout tasks (`softmax`, `matmul`, `mla_attention`, `moe_permute`), the same 8 training tasks, 2 evaluation seeds and `max_iters=3`; the holdout tasks were never in the training root.

| run | training budget | training coverage | groups |
| --- | --- | --- | --- |
| pilot | 8 iters (3 runs) | 2 tasks | kb, no_rsi |
| follow-up | 24 iters (8 runs) | all 8 tasks, 1 seed each | kb, no_rsi, seed |

## Run 1 — budget 8 (pilot)

Results:

| group | n | first-pass iters (median, CI) | best speedup (median, CI) | tokens/gain | passes | cheat interceptions |
| --- | --- | --- | --- | --- | --- | --- |
| kb | 8 | 1.0 [1.0, 4.0] | 0.9534 [0.0, 11.9455] | 633,200 | 5/8 | 0 |
| no_rsi | 8 | 1.0 [1.0, 1.0] | 0.9215 [0.7309, 12.1445] | 548,250 | 8/8 | 0 |

Paired (task × seed): best-speedup `kb - no_rsi = -0.4448`, 95% CI `[-0.8451, -0.046]` (significantly worse); first-pass improvement `0.0`, CI `[-3.0, 0.0]`; tokens/gain +44,300 (worse). Negative transfer 4/4.

Negative transfer by task:

| task | kb best | no_rsi best | delta |
| --- | --- | --- | --- |
| matmul | 0.9534 | 1.0078 | -0.0543 |
| mla_attention | 0.0 | 0.7723 | -0.7723 |
| moe_permute | 11.95 | 12.5207 | -0.5707 |
| softmax | 0.5005 | 0.7423 | -0.2418 |

## Run 2 — budget 24 (coverage-first follow-up)

Training: 24 iterations over all 8 train tasks (`grouped_gemm_swiglu`, `grouped_matmul`, `vector_add`, `moe_finalize_routing`, `rmsnorm`, `rope`, `decode_attention`, `sdpa_attention`), 1 seed each, 3 iterations per run; 7/8 runs passed; `sdpa_attention` failed correctness.

Results:

| group | n | first-pass iters (median, CI) | best speedup (median, CI) | tokens/gain | passes | cheat interceptions |
| --- | --- | --- | --- | --- | --- | --- |
| kb | 8 | 1.0 [1.0, 2.0] | 0.9816 [0.7194, 12.1037] | 722,500 | 7/8 | 0 |
| no_rsi | 8 | 1.0 [1.0, 2.0] | 0.9989 [0.7268, 11.165] | 515,750 | 7/8 | 0 |
| seed | 8 | 1.0 [1.0, 1.0] | 1.0044 [0.6349, 12.8494] | 640,200 | 7/8 | 0 |

Paired (task × seed): best-speedup `kb - no_rsi = +0.0055`, 95% CI `[-0.0397, 0.7194]` (not significant); first-pass improvement `0.0`, CI `[-1.0, 1.0]`; tokens/gain +144,250, CI `[28.7, 206900]` (worse).

Negative transfer (threshold 0.03): **0/4** — matmul -0.0206, MLA -0.0037, moe_permute +0.7366, softmax +0.0048.

## Hard-operator end-to-end (post-integration, MLA attention)

After merging P0–P4 into `main` (`17a5e66`), a real-model end-to-end run on the hard `mla_attention` task (m18 Ascend910B4, `deepseek-chat`, temperature 0.7, 16 iterations, fresh knowledge root, P2 ablation budget 3):

- **Stable best: 1.2678x**, ABAB 95% CI `[1.2648, 1.2956]`, stability 0.61%. An unstable 1.3428x result appeared earlier; the stable pass is selected (fix #9) so evidence-producing paths use the stable solution.
- **P2 ablation gate:** removing the two declared strategies produced **1.3509x** and **1.3661x** — both *faster* than the full 1.2678x solution. The declarations were therefore not causal; the success evidence was correctly demoted to neutral.
- Consequences: 0 verified strategies, 0 combos, 0 auto-compiled skills; 1 exemplar recorded. 0 cheat interceptions; provenance valid.
- Raw data: [`data/e2e/mla-20261106-runB1.json`](data/e2e/mla-20261106-runB1.json) (attempts, ablations, distill summary, final solution).

This validates the integrated pipeline on a hard operator and shows the P2/P4 gates doing their job: a real 1.27x win with non-causal declarations is **not** turned into strategy evidence or a skill.

## Combined learning curve (training budget → holdout outcome)

| budget | group | best speedup median | paired best delta vs no_rsi (95% CI) | paired first-pass delta (95% CI) | negative transfer |
| --- | --- | --- | --- | --- | --- |
| 8 | kb | 0.9534 | -0.4448 [-0.8451, -0.046] | 0.0 [-3.0, 0.0] | 4/4 |
| 8 | no_rsi | 0.9215 | — | — | — |
| 24 | kb | 0.9816 | +0.0055 [-0.0397, 0.7194] | 0.0 [-1.0, 1.0] | 0/4 |
| 24 | no_rsi | 0.9989 | — | — | — |
| 24 | seed | 1.0044 | — | — | — |

The 8-iteration pilot and the 24-iteration follow-up use the same holdout/task/model/eval settings; the second point is a comparable learning-curve checkpoint, not a separate protocol.

## P5 pass lines

| line | budget-8 pilot | budget-24 follow-up |
| --- | --- | --- |
| first-pass iteration lower, CI not crossing 0 | ❌ (0.0, [-3,0]) | ❌ (0.0, [-1,1]) |
| best speedup higher, CI not crossing 0 | ❌ (-0.4448, significant negative) | ❌ (+0.0055, CI crosses 0) |
| cost per gain not worse | ❌ | ❌ (+144,250, CI positive) |
| negative transfer ≤ 1/4 | ❌ (4/4) | ✅ (0/4) |
| cheat interceptions | 0 (report only) | 0 (report only) |

## Conclusion

**P5 is not passed.** A larger `kb@T` (24 iterations over all 8 training tasks) removes the significant negative transfer seen with the 8-iteration pilot (4/4 → 0/4) and moves the best-speedup paired delta from significantly negative (-0.4448) to approximately zero (+0.0055), but it does **not** produce a positive, significant gain: the 95% CI crosses 0, first-pass iterations do not improve, and cost per gain is worse. The seed-only control (not token-length-matched) scores 1.0044, similar to both groups.

Under the tested model and hardware, the current evidence does **not** support “RSI enabled gets faster/better than RSI disabled”; it supports “a tiny knowledge base is harmful; a modest one is neutral on the measured holdout tasks”.

## Limits

- 2 evaluation seeds and 3 max iterations per run; the plan's full protocol (5 seeds, 8 max iterations, 12-task split) has not been run.
- The seed-only control is a seed pool, not token-length-matched; the “longer prompt” confound is not fully excluded.
- One checkpoint per budget (no per-stage evaluation); the combined curve uses the 8- and 24-iteration runs.
- The 4 holdout tasks produce 8 pairs per group; bootstrap CIs over task × seed are wide.
- The model/solution quality varies per seed (e.g., MLA seed 1 vs 2); failures are included as `max_iters+1` / speedup 0 in the metrics.

## Raw data

- [`data/p5/pilot1-results.json`](data/p5/pilot1-results.json) — budget 8.
- [`data/p5/budget24-results.json`](data/p5/budget24-results.json) — budget 24.

The API key is not stored in the repository or in these artifacts.
