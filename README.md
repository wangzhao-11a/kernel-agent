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