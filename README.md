# kernel-agent

**A closed loop for kernel optimization — and a memory that makes the next loop cheaper.**

```
write implementation → build/run → check correctness → measure → structured feedback → revise
```

On top of that loop sits a small recursive-self-improvement (RSI) layer:

```
run trajectory → distill (practice × measurement × expectation gap)
    → strategy pool + combination rules → condition-matched retrieval
    → priors injected into the next run → reuse across tasks
    → conditions narrow as counter-examples arrive → …
```

The same code runs on CPU (for development), NVIDIA GPUs (`--device cuda`) and
Ascend NPUs (`--device npu`). Four representative task families ship with the
repo: SparseFlashAttention, MLA, MoEFinalizeRouting and GroupedMatmul, plus
twelve smaller tasks at two levels.

Zero runtime dependencies beyond NumPy for the core loop. Torch is optional and
only needed for the shipped torch examples and real benchmarking.

---

## Why

An agent that can edit a kernel is easy to build. An agent that *improves* over
time is not, because three things are usually missing:

1. **A trustworthy measurement loop** — isolation, timeouts, warmup, repetitions,
   input-mutation checks, and a rule that a CPU run is never recorded as hardware
   evidence.
2. **Structured feedback** — the agent must see *why* an attempt failed
   (`build_error` / `fail_correctness` / `crash` / `timeout` / `bench_error`),
   not just a score.
3. **Memory with scope** — "this tile size was faster" is useless without the
   SOC, dtype, shape family and compiler version it was measured on.

This repository is an attempt at all three, in a form small enough to read.

---

## Quick start (no GPU needed)

```bash
pip install numpy

# 22 checks: parsers, pipeline, correctness gates, timeout isolation,
# replay agent, RSI store/retrieval/refinement, memory and forms.
python scripts/self_test.py

# run one task with a reference solution on CPU
python -m harness.runner --task tasks/level2/moe_finalize_routing.py \
    --solution solutions/examples/moe_finalize_routing_torch.py --device cpu

# summarize results
python -m harness.report runs/results.jsonl

# offline agent loop (replay provider, no API key)
python -m agent.loop --task tasks/level1/vector_add.py \
    --provider replay --replay-file configs/replay_vector_add.example.json \
    --device cpu --max-iters 6
```

With torch installed, add the torch examples and the profiling checks:

```bash
pip install torch
python scripts/self_test.py
```

---

## Layout

```
harness/     evaluation loop: task loading, correctness, timing, isolated
             execution, result archiving, profilers (ncu / msprof / torch)
agent/       minimal agent loop: prior injection, declared strategy, failure
             hints, profiling feedback, automatic distillation
rsi/         RSI layer: features, practice library, strategy store, trajectory,
             distill, retrieve, evolve, memory (facts/failures/skills)
tasks/       level1 (4 general tasks) + level2 (8 attention/GEMM/MoE/elementwise)
solutions/   reference solutions and templates
scripts/     self-tests, profiling helpers, dashboard builders
configs/     agent configs and offline replay fixtures
rsi/knowledge/   seed strategy pool + operator memory (JSONL)
```

---

## The evaluation contract

A task is a Python module with a `TASK` dict and three functions:

```python
TASK = {
    "name": "my_op",
    "level": 1,
    "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    "num_trials": 3,
    "warmup": 5,
    "rep": 50,
    "require": [],          # e.g. ["triton"] or ["ascendc"]
}

def make_inputs(device, trial): ...   # return a tuple of tensors
def reference(*inputs): ...           # the ground truth
def run(*inputs): ...                 # the candidate implementation
```

A solution is any file defining `run(...)`. The harness:

- runs each trial in a subprocess with a hard timeout;
- checks output correctness against `reference`;
- rejects solutions that mutate their inputs;
- measures with warmup + repetitions and reports median/p50/p90;
- classifies every failure into a stage (`build_error`, `fail_correctness`,
  `crash`, `timeout`, `bench_error`);
- records provenance (`cpu` / `cuda` / `npu`) so mechanics runs cannot be
  mistaken for hardware results.

### Evaluation policy (strict tasks)

Legacy PyTorch tasks carry an explicit exemption:

```python
"schema_version": 1,
"policy": {
    "forbid_native_compute": False,
    "require_kernel": False,
    "torch_fused_exempt": True,
    "reason": "legacy torch reference / replay fixture exemption",
},
```

A kernel-only task opts in with `forbid_native_compute: true` and
`require_kernel: true`.  While the candidate runs, a `TorchDispatchMode`
allow-list permits allocation/view/copy/metadata ATen ops and rejects native
compute.  AST checks are auxiliary (reference imports, dynamic task loading,
native compute calls, kernel evidence).  A candidate that calls the reference,
uses native compute, caches across trials, launches on a side stream, overfits a
fixed shape, or mishandles NaN/Inf is rejected by `harness/policy.py`; see
`scripts/policy_selftest.py`.

Tasks can declare a weighted, held-out-aware shape set:

```python
"shape_set": [
    {"name": "train", "shape": [2, 2048], "dtype": "float16", "weight": 0.7,
     "heldout": false, "input_specs": [{"kind": "randn"}]},
    {"name": "heldout_odd", "shape": [17, 257], "dtype": "float16", "weight": 0.0,
     "heldout": true, "input_specs": [{"kind": "randn"}]},
]
```

Phase 1 fixes the schema, held-out split, per-case seeds and NaN/Inf/empty/
extreme input kinds; weighted aggregation and the worst-shape gate are Phase 4.
`scripts/migrate_jsonl_schema.py` adds `schema_version` to old result/attempt
JSONL files with a `.bak` copy.

---

## AscendC operator loop (CannBench)

Besides Python tasks, the harness can drive a compiled AscendC operator package:

- `harness/cannbench.py` loads a JSON spec (`tasks/cannbench_ascendc/*.json`)
  with an allow-listed edit set and build/test/bench/profile commands, copies
  the package into an isolated workspace, and parses per-case PASS/FAIL plus
  `CANNBENCH_METRICS` / `CANNBENCH_PROFILE`.
- `agent/cannbench_loop.py` runs a plan-gated loop: `plan` -> `read_file` /
  `patch_files` -> `kb_search` / `source_grep` -> `run_eval` -> `reflect` /
  `profile`. Edits are cumulative, a no-op write is rejected, and a failing
  candidate rolls back to the best passing snapshot.
- `agent/review.py` blocks known AscendC mistakes before the build (in-place
  `Gather`, wrong `GatherMask`, header-only edits, plan/API mismatch, arch35
  leakage, destructive rewrites).
- `rsi/cann_ops.py` + `data/cann_ops_catalog.json` expose an offline source map
  of the installed CANN operators; `rsi/cards.py` adds AscendC-aware retrieval
  and free-text `kb_search`.

`tasks/cannbench_ascendc/` ships SwiGlu and RMSNorm specs for 910B (plus SwiGlu
950 and add). `work/main_tree` is the trimmed operator package they edit;
`solutions/examples/swiglu_ascendc_910b_tiny_gather` is a reference kernel.

Run the offline contract test (no NPU, compiler or API key):

```bash
python scripts/cannbench_selftest.py   # 20 checks
```

On a device, drive a spec through the normal command contract:

```bash
python -m agent.cannbench_loop \
    --spec tasks/cannbench_ascendc/swi_glu_910b.json \
    --max-iters 6 --api-key-env KERNEL_AGENT_API_KEY
```

## Search and meta-eval (Phase 3)

`rsi/search.py` implements search policies over a synthetic operator terrain
(`rsi/terrain.py`): the original single chain, best-first beam, UCB and Thompson
bandits over tile/split/mode moves, and staged cheap screening.  `scripts/meta_eval.py`
runs them under a fixed evaluation and token budget and compares multi-seed
success/best distributions.

The tagged baseline `baseline-single-chain-v1` (5 seeds, 40 evaluations,
6000 tokens, CPU synthetic terrain) is in `data/meta_eval/`.  The Phase-3
comparison (`data/meta_eval/phase3-search-compare.json`) is:

| policy | pass@1 | pass@2 | pass@3 | best mean | best min | best max | tokens |
|---|---|---|---|---|---|---|---|
| single-chain | 0.60 | 0.90 | 1.00 | 2.328 | 1.930 | 2.753 | 4800 |
| beam | 0.20 | 0.40 | 0.60 | 2.049 | 1.864 | 2.595 | 4800 |
| UCB | 1.00 | 1.00 | 1.00 | 3.080 | 2.558 | 3.442 | 4800 |
| Thompson | 1.00 | 1.00 | 1.00 | 3.283 | 2.479 | 3.662 | 4800 |
| staged | 1.00 | 1.00 | 1.00 | 2.811 | 2.476 | 3.225 | 2010 |

Bandit allocation and staged screening beat the single chain on this terrain;
the current best-first beam does not.  These are CPU synthetic mechanism results,
not hardware/operator performance claims.  `scripts/search_selftest.py` pins the
budget accounting and the seeded ranking.

`scripts/ablate.py` adds component ablations (`data/meta_eval/phase4-ablation.json`):
strict memory has 0 cross-SoC recall, soft scope raises it to 1.0 with every hit
flagged unverified, and leave-one-family-out produces 0 recall and 0
contaminating records.  Its screening proxy reuses the staged-vs-single-chain
comparison and is explicitly **not** a profiler ablation (replay runs cannot
measure profiler efficacy).

## Observation tools (Phase 4.2)

- **Roofline.** A TASK may declare `"workload": {"bytes": ..., "flops": ...}` and
  optional `"peak_bw_gbps"` / `"peak_tflops"` (or set
  `KERNEL_AGENT_PEAK_BW_GBPS` / `KERNEL_AGENT_PEAK_TFLOPS`).  The worker then adds
  `timing.roofline` with achieved bandwidth/throughput, utilization and a
  memory/compute bottleneck label; `format_summary` prints it.  This is an
  estimate from declared traffic, not a hardware counter.
- **On-demand memory.** The agent can call
  `{"action": "memory", "query": "<free text>"}`.  `MemoryStore.search_text`
  returns facts/failures/skills/strategies by token overlap (CJK bigrams included)
  across scopes, explicitly marked **not scope-matched** so a text hit cannot be
  mistaken for a condition-matched prior.
- `scripts/roofline_selftest.py` covers both on CPU.

### Soft scope and failure similarity (Phase 4.3)

Strict retrieval is unchanged: a fact/failure/skill whose scope contradicts the
current run is never presented as applicable.  The agent prompt additionally
receives a low-weight **analogy** layer (`soft_scope_match`, 20% weight, at most
two per kind) for records that share the op/op_family/form but differ in
SoC/dtype/shape.  Every such record is prefixed `UNVERIFIED ANALOGY` and carries
"verify with one experiment"; `--no-soft-scope` disables the layer.  Failed
attempts also query past failures by text similarity
(`MemoryStore.similar_failures`) and surface their workaround, marked as
not scope-matched.  `scripts/soft_scope_selftest.py` pins the separation.

### Long trajectories: compression, resume, per-turn accounting (Phase 4.4)

When the conversation exceeds `--max-context-chars` (default 60000), the loop
keeps the task prompt and the last `--keep-recent` messages and folds everything
else into a structured summary (evaluation counts, best result, strategies tried,
recent errors).  The agent can call `read_solution` to re-read the current code
after compression.  Every turn writes `checkpoint.json` (messages, best, current
path, usage, call count) atomically; `--resume <run_dir>` continues from it, and
replay runs skip already-consumed responses.  `meta.json` now carries
`token_usage_by_iteration`, and each assistant transcript row carries `usage` and
`context_chars`.  `scripts/context_selftest.py` covers compression and a real
checkpoint/resume round trip on CPU.

### Weighted shapes and the worst-shape gate (Phase 4.5)

A task with `shape_set` is timed on **every training case**.  The reported
`timing.speedup` is the weight-normalized mean (`weighted_speedup`), all per-case
rows are kept under `timing.shape_cases`, and `timing.worst_case` names the
slowest shape.  When more than one training case exists, a candidate fails with
`fail_shape_gate` if its worst case is below `shape_gate_min_speedup` (default
0.9, set 0 to disable).  Held-out cases remain correctness-only.  The bootstrap
interval uses within-case resampling (`bootstrap_weighted_ratio`).
`scripts/shape_gate_selftest.py` covers a passing candidate and a candidate that
regresses on one shape.

---

## The RSI layer

**Trajectory.** Every attempt is archived as JSONL: the code, the declared
strategy, the measured speedup, the expected speedup, the profiler summary and
the failure signature.

**Distillation.** After a run, the trajectory is reduced to:
- which practices were claimed, and which are supported by the diff;
- measured vs expected speedup (calibration);
- failure signatures with a minimal repro.

**Strategy pool.** Strategies are stored with applicability conditions
(`op_class`, hardware, dtypes, shape features). Retrieval is deterministic and
explainable; there are no embeddings.

**Condition narrowing.** If a strategy succeeds on one shape family and fails on
another, the condition set narrows instead of the claim being kept globally.

**Memory.** Three additional stores:
- `facts.jsonl` — authoritative constraints (API rules, alignment, launch rules);
- `failures.jsonl` — error signature + stage + minimal repro + workaround;
- `skills.jsonl` — executable templates with preconditions and validation evidence.

Scope filtering is hard: a fact measured on one SOC, dtype or form is not
injected into a run where it does not match. When in doubt, the agent gets no
prior rather than a wrong one.

**Audit and curriculum.**

```bash
python -m rsi.evolve audit   # capability and evidence audit
python -m rsi.evolve plan    # next practice curriculum
```

The audit deliberately flags "mechanics-only" runs (CPU/replay) that claim
hardware provenance. The plan scores tasks by family coverage, transfer and
evidence gaps, so the next round practices what is actually missing.

---

## Profiling

Three backends, all optional:

```bash
python -m harness.profile_runner --task tasks/level1/vector_add.py \
    --solution solutions/examples/vector_add_good.py --device cpu --backend torch
```

- `torch` — `torch.profiler`, works on CPU/CUDA/NPU;
- `ncu` — NVIDIA Nsight Compute CSV export;
- `msprof` — Ascend msprof export.

The profile summary is fed back into the agent's next prompt as a bottleneck
block (e.g. "host sync dominates", "allocation overhead dominates").

---

## What this is not

- Not a kernel library. It does not ship production kernels.
- Not a benchmark leaderboard. Tasks are for development and regression.
- Not a model-training framework.
- Not a replacement for a profiler: it tells the agent *what* changed and
  whether it helped; the profiler explains *why*.

---

## Status and limits

- The core loop is tested on CPU with NumPy and torch. CUDA and NPU paths need
  the corresponding hardware and a matching torch/torch_npu install.
- The shipped reference solutions are correctness oracles, not optimized kernels.
- Three repetitions are a screening signal, not statistical proof. For
  statistically gated service-level tuning, see the sister project
  [`infer-rsi`](https://github.com/wangzhao-11a/infer-rsi).

---

## Related

- [infer-rsi](https://github.com/wangzhao-11a/infer-rsi) — experiment ledger and statistical gate for service-level inference tuning.
- [agent-harness](https://github.com/wangzhao-11a/agent-harness) — zero-dependency harness runtime for LLM agents.

## License

Apache-2.0 — see `LICENSE`.