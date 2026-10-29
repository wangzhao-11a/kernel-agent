# CannBench AscendC specs

These JSON files are tasks for `agent.cannbench_loop` and `harness.cannbench`.
They describe a **package + the commands that build/test it**, instead of a
single Python `run()` file.

```json
{
  "id": "swiglu_ascendc_910b",
  "package": "../../work/main_tree",
  "edit_files": ["csrc/ops/swiglu/op_kernel/swiglu_kernel.cpp"],
  "build_command": ["bash", "build.sh", "--soc=ascend910b", "--ops=swiglu", "--install"],
  "test_command": ["python3", "tests/test_swiglu_official.py"],
  "bench_command": ["python3", "tests/bench_swiglu.py"],
  "timeout_s": 900,
  "soc": "ascend910b",
  "hardware": "ascend-910b",
  "expected_speedup": 3.0,
  "metadata": {"op_class": "elementwise", "dtypes": ["float16", "float32", "bfloat16"]}
}
```

## Contract

- `package` is resolved relative to the spec file. The backend copies it to a
  fresh workspace per evaluation, so the source tree is never mutated.
- `edit_files` is the **allow-list**. The agent may only write these paths; any
  other path is rejected before a build starts.
- `build_command` must produce an installed `cann_bench` (or otherwise make the
  test command runnable). `test_command` must print one line per official case
  in the standard format and a final `*TEST PASS/FAIL`:
  ```
  (1024, 2048)           float16   dim=-1  MERE=0.000e+00 MARE=0.000e+00 PASS
  ...
  SWIGLU_TEST PASS
  ```
- `bench_command` is optional. If present it should print:
  ```
  CANNBENCH_METRICS {"speedup": 1.23, "metric_key": "speedup"}
  ```
- `hardware` is the **claimed** hardware. The agent loop records the actual
  `device` separately; `rsi.evolve audit` flags CPU runs labelled as NPU.
- The AscendC loop also reads the CANN KB (`--kb-root`) and the installed source-map
  catalog (`--ops-catalog`). The agent can use `{"action":"knowledge","id":"..."}`
  for a card, `{"action":"source","path":"..."}` for an official kernel/tiling file,
  `{"action":"kb_search","query":"..."}` for free-text KB search, and
  `{"action":"source_grep","pattern":"..."}` for source grep.
- `{"action":"reflect","lesson":"...","next":"..."}` records the lesson and the one
  next change in `reflections.jsonl` / `meta.reflections`.
- If the spec has `profile_command` (printing `CANNBENCH_PROFILE {...}`), the agent can
  call `{"action":"profile","shape":"...","dtype":"...","dim":...}` and get
  VEC/scalar/MTE2/MTE3 ratios plus the bottleneck.

## Add a new operator

1. Start from a package that already builds on the target SOC
   (`work/main_tree` is the working SwiGlu scaffold).
2. Add a `tasks/cannbench_ascendc/<op>_<soc>.json` with the package, the files
   the agent may edit, and the build/test commands.
3. Run the mock self-test to validate the wiring:
   ```bash
   python scripts/cannbench_selftest.py
   ```
4. On the NPU machine, run one correctness pass:
   ```bash
   python -m agent.cannbench_loop --spec tasks/cannbench_ascendc/swi_glu_910b.json \
     --provider openai --model <model> --base-url <url> \
     --max-iters 8 --target-speedup 1.0
   ```
5. Only after all official cases pass should you add a `bench_command` and optimize
   for speed. `--ops=<name>` keeps the wheel build scoped to the target operator.
   Do not auto-submit to the platform; submission costs credits.

## Status meanings

| status | meaning |
|---|---|
| `pass` | build ok, all parsed cases passed, optional bench ok |
| `fail_correctness` | at least one case FAIL, or the final `*TEST FAIL` |
| `build_error` | build command returned non-zero |
| `crash` | test command failed without a usable case table/verdict |
| `timeout` | build/test exceeded `timeout_s` |
| `bench_error` | correctness passed but the benchmark command failed |

The result JSON also contains `workspace`, `files`, `build/test/bench` logs and
`case_summary`. `rsi.evolve` reads the agent loop's `meta.json`/`attempts.jsonl`
and treats a run as valid performance evidence only when `device` matches the
claimed `hardware`.