"""Self-test: profiler parsers, harness pipeline, example solutions, agent loop.

Runs anywhere — the core checks use a numpy task, so no torch/GPU is required.
When torch is installed, the shipped torch examples are exercised as well.

    python scripts/self_test.py
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NP_TASK_SOURCE = textwrap.dedent('''
    """Minimal numpy task used by the self-test (no torch needed)."""
    import numpy as np

    TASK = {
        "name": "np_vector_add",
        "level": 0,
        "description": "Elementwise add of two float32 vectors (65536,), numpy backend.",
        "tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "num_trials": 2,
        "warmup": 3,
        "rep": 20,
    }

    def make_inputs(device, trial):
        rng = np.random.default_rng(100 + trial)
        return (rng.standard_normal(65536).astype(np.float32),
                rng.standard_normal(65536).astype(np.float32))

    def reference(x, y):
        return x + y
''')

GOOD_SOLUTION = "def run(x, y):\n    return x + y\n"
WRONG_SOLUTION = "def run(x, y):\n    return x - y\n"
MUTATING_SOLUTION = "def run(x, y):\n    x += y\n    return x\n"
HANGING_SOLUTION = "def run(x, y):\n    while True:\n        pass\n"

NCU_CSV = textwrap.dedent('''\
    ==PROF== Connected to process 1234
    "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
    "0","1234","python","host","my_kernel","2026-10-12","1","7","256","1024","0","9.0","SpeedOfLight","Duration","ns","12.5"
    "1","1234","python","host","my_kernel","2026-10-12","1","7","256","1024","0","9.0","SpeedOfLight","Compute (SM) Throughput","%","45.2"
    "2","1234","python","host","my_kernel","2026-10-12","1","7","256","1024","0","9.0","SpeedOfLight","Memory Throughput","%","88.1"
    "3","1234","python","host","other_kernel","2026-10-12","1","7","256","1024","0","9.0","SpeedOfLight","Duration","ns","3.0"
''')

MSPROF_CSV = textwrap.dedent('''\
    Op Name,Task Duration(us),aicore_time(us)
    FlashAttentionScore,120.5,110.0
    MatMulV2,80.0,75.0
    FlashAttentionScore,119.5,109.0
''')

NP_SOFTMAX_TASK = textwrap.dedent('''
    """Minimal numpy softmax task used by the RSI self-test."""
    import numpy as np

    TASK = {
        "name": "np_softmax",
        "level": 0,
        "description": "Row-wise softmax over the last dim, numpy backend (512, 512).",
        "tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "num_trials": 2,
        "warmup": 3,
        "rep": 20,
        "tags": ["softmax", "reduction", "memory-bound"],
    }

    def make_inputs(device, trial):
        rng = np.random.default_rng(7 + trial)
        return (rng.standard_normal((512, 512)).astype(np.float32) * 2.0,)

    def reference(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
''')

NP_SOFTMAX_GOOD = (
    "import numpy as np\n\n\n"
    "def run(x):\n"
    "    m = x.max(axis=-1, keepdims=True)\n"
    "    e = np.exp(x - m)\n"
    "    return e / e.sum(axis=-1, keepdims=True)\n"
)
NP_SOFTMAX_WRONG = "import numpy as np\n\n\ndef run(x):\n    return x * 2.0\n"

REQ_TASK_SOURCE = textwrap.dedent('''
    """numpy task with a harness-enforced requirement (self-test only)."""
    import numpy as np

    TASK = {
        "name": "req_task",
        "level": 0,
        "require": ["triton"],
        "tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "num_trials": 1,
        "warmup": 1,
        "rep": 5,
    }

    def make_inputs(device, trial):
        rng = np.random.default_rng(11 + trial)
        return (rng.standard_normal(2048).astype(np.float32),)

    def reference(x):
        return x + 1.0
''')

OPBASE_TEMPLATE = textwrap.dedent('''
    """numpy task with a task-provided 'op' baseline (self-test only)."""
    import numpy as np

    TASK = {{
        "name": "opbase_task",
        "level": 0,
        "baseline": "op",
        "tolerance": {{"atol": 1e-6, "rtol": 1e-6}},
        "num_trials": 1,
        "warmup": 1,
        "rep": 5,
    }}

    def make_inputs(device, trial):
        rng = np.random.default_rng(21 + trial)
        return (rng.standard_normal(2048).astype(np.float32),)

    def reference(x):
        return x + 1.0

    def baseline(x):
        {body}
''')

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))
        traceback.print_exc()
        return
    RESULTS["pass"].append(name)
    print("PASS  %s" % name)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def make_fixture_files(tmp):
    return {
        "task": write(os.path.join(tmp, "tasks", "np_vector_add.py"), NP_TASK_SOURCE),
        "good": write(os.path.join(tmp, "solutions", "good.py"), GOOD_SOLUTION),
        "wrong": write(os.path.join(tmp, "solutions", "wrong.py"), WRONG_SOLUTION),
        "mutating": write(os.path.join(tmp, "solutions", "mutating.py"), MUTATING_SOLUTION),
        "hanging": write(os.path.join(tmp, "solutions", "hanging.py"), HANGING_SOLUTION),
    }


def run_task(tmp, files, solution, timeout=180):
    from harness.runner import run_evaluation

    return run_evaluation(
        task=files["task"], solution=solution, device="cpu", timeout=timeout,
        out_dir=os.path.join(tmp, "out"), results_file=os.path.join(tmp, "results.jsonl"),
        quiet=True,
    )


def check_ncu(tmp):
    from harness.profilers import ncu

    summary = ncu.summarize_csv(write(os.path.join(tmp, "ncu_export.csv"), NCU_CSV))
    assert summary["n_kernels"] == 2, summary
    assert summary["primary"]["name"] == "my_kernel", summary["primary"]
    duration = summary["primary"]["metrics"]["Duration"]["value"]
    assert abs(duration - 12.5) < 1e-9, duration
    assert "Duration" in ncu.format_summary(summary)


def check_msprof(tmp):
    from harness.profilers import msprof

    path = write(os.path.join(tmp, "msprof_export", "op_summary.csv"), MSPROF_CSV)
    summary = msprof.summarize_dir(os.path.dirname(path))
    assert summary["top_ops"], summary
    top = summary["top_ops"][0]
    assert top["name"] == "FlashAttentionScore", top
    assert abs(top["total_us"] - 240.0) < 1e-6, top
    assert abs(summary["total_us"] - 320.0) < 1e-6, summary["total_us"]
    assert "FlashAttentionScore" in msprof.format_summary(summary)


def check_pipeline_pass(tmp, files):
    res = run_task(tmp, files, files["good"])
    assert res["status"] == "pass", res.get("error")
    assert res["correctness"]["ok"] is True
    assert res["timing"]["speedup"] > 0, res["timing"]


def check_pipeline_wrong(tmp, files):
    res = run_task(tmp, files, files["wrong"])
    assert res["status"] == "fail_correctness", res.get("status")
    assert "tolerance" in (res.get("error") or ""), res.get("error")


def check_pipeline_mutation(tmp, files):
    res = run_task(tmp, files, files["mutating"])
    assert res["status"] == "fail_correctness", res.get("status")
    assert "mutated" in (res.get("error") or ""), res.get("error")


def check_pipeline_timeout(tmp, files):
    res = run_task(tmp, files, files["hanging"], timeout=8)
    assert res["status"] == "timeout", res.get("status")


def check_report(tmp):
    from harness import report

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = report.main([os.path.join(tmp, "results.jsonl")])
    out = buf.getvalue()
    assert rc == 0, rc
    assert "pass@1" in out, out


def check_agent_replay(tmp, files):
    replay = write(os.path.join(tmp, "replay.json"), json.dumps({
        "responses": [
            json.dumps({"action": "write_solution", "content": WRONG_SOLUTION}),
            json.dumps({"action": "write_solution", "content": GOOD_SOLUTION}),
            json.dumps({"action": "finish", "summary": "self-test finish"}),
        ]
    }, ensure_ascii=False))
    out_dir = os.path.join(tmp, "agent_runs")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", files["task"], "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--out-dir", out_dir,
        "--knowledge-root", os.path.join(tmp, "agent_knowledge"),
        "--results-file", os.path.join(tmp, "agent_results.jsonl"),
        "--max-iters", "6", "--target-speedup", "0.5", "--timeout", "180",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["best"]["status"] == "pass", meta
    final = os.path.join(run_dir, "final_solution.py")
    assert os.path.isfile(final), run_dir
    with open(final, encoding="utf-8") as fh:
        assert "x + y" in fh.read()
    for expected in ("transcript.jsonl", "transcript.md"):
        assert os.path.isfile(os.path.join(run_dir, expected)), expected


def check_torch_examples(tmp):
    from harness.runner import run_evaluation

    task = os.path.join(ROOT, "tasks", "level1", "vector_add.py")
    common = dict(device="cpu", timeout=600, out_dir=os.path.join(tmp, "torch_out"),
                  results_file=os.path.join(tmp, "torch_results.jsonl"), quiet=True)
    good = run_evaluation(task=task, solution=os.path.join(ROOT, "solutions", "examples", "vector_add_good.py"), **common)
    assert good["status"] == "pass", good.get("error")
    wrong = run_evaluation(task=task, solution=os.path.join(ROOT, "solutions", "examples", "vector_add_wrong.py"), **common)
    assert wrong["status"] == "fail_correctness", wrong.get("status")


def check_torch_softmax(tmp):
    from harness.runner import run_evaluation

    res = run_evaluation(
        task=os.path.join(ROOT, "tasks", "level1", "softmax.py"),
        solution=os.path.join(ROOT, "solutions", "examples", "softmax_manual.py"),
        device="cpu", timeout=600, out_dir=os.path.join(tmp, "torch_out"),
        results_file=os.path.join(tmp, "torch_results.jsonl"), quiet=True,
    )
    assert res["status"] == "pass", res.get("error")


def check_rsi_store_and_retrieve(tmp):
    from rsi.retrieve import priors_block, retrieve
    from rsi.store import StrategyStore

    root = os.path.join(tmp, "retrieve_knowledge")
    store = StrategyStore(root)
    assert len(store.strategies) >= 10, len(store.strategies)
    again = StrategyStore(root)
    assert len(again.strategies) == len(store.strategies)

    from rsi.features import task_features

    feats = task_features({
        "name": "sparse_flash_attention",
        "description": "top-k sparse attention over fp32 tensors",
        "tags": ["attention", "sparse", "fp32"],
        "op_class": "attention",
        "dtypes": ["float32"],
    })
    matched = retrieve(store, feats, "ascend-910b", top_k=5)
    assert matched, matched
    keys = [s.get("key") for s, _sc, _m in matched]
    assert "online_softmax" in keys or "sparse_skip_blocks" in keys, keys
    block = priors_block(feats, "ascend-910b", matched, [])
    assert "Tuning priors" in block, block
    assert matched[0][0]["id"] in block, block


def check_rsi_condition_refinement(tmp):
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "refine_knowledge"))
    st = store.find_by_key("block_tiling")
    store.record_evidence(st["id"], "runs/a", "task_a", 1.4, "success", hardware="ascend-910b", dtype="fp16")
    assert store.refine_conditions(st["id"]) is None, "single-sided evidence must not narrow conditions"
    store.record_evidence(st["id"], "runs/b", "task_b", 0.0, "failure", hardware="nvidia-gpu", dtype="fp16")
    note = store.refine_conditions(st["id"])
    assert note and "ascend-910b" in note, note
    assert st["conditions"]["hardware"] == ["ascend-910b"], st["conditions"]
    assert st["confidence"] in ("seed", "verified"), st["confidence"]


def check_rsi_loop_distill(tmp):
    task_path = write(os.path.join(tmp, "rsi_tasks", "np_softmax.py"), NP_SOFTMAX_TASK)
    replay = write(os.path.join(tmp, "rsi_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": NP_SOFTMAX_WRONG,
             "strategies": ["online softmax"], "expected_speedup": 1.3},
            {"action": "write_solution", "content": NP_SOFTMAX_GOOD,
             "strategies": ["online softmax", "multi-core split"], "expected_speedup": 1.0},
            {"action": "finish", "summary": "rsi self-test"},
        ]
    }, ensure_ascii=False))
    knowledge_root = os.path.join(tmp, "rsi_knowledge")
    out_dir = os.path.join(tmp, "rsi_runs")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--hardware", "ascend-910b",
        "--knowledge-root", knowledge_root, "--out-dir", out_dir,
        "--results-file", os.path.join(tmp, "rsi_results.jsonl"),
        "--max-iters", "6", "--target-speedup", "0.5", "--timeout", "180",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])

    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["best"]["status"] == "pass", meta["best"]

    with open(os.path.join(run_dir, "transcript.md"), encoding="utf-8") as fh:
        transcript = fh.read()
    assert "Tuning priors" in transcript, "priors block missing from the prompt"
    assert meta["rsi"]["priors_matched"], meta["rsi"]
    assert meta["rsi"]["attempted"], meta["rsi"]
    assert meta["rsi"]["distill"] and meta["rsi"]["distill"]["evidence"] >= 2, meta["rsi"]

    with open(os.path.join(run_dir, "trajectory.json"), encoding="utf-8") as fh:
        traj = json.load(fh)
    assert len(traj["attempts"]) == 2, traj["attempts"]
    assert traj["task_file"] and os.path.isfile(traj["task_file"]), traj["task_file"]

    from rsi.trajectory import replay_trajectory

    replay = replay_trajectory(run_dir, device="cpu")
    assert replay["n"] == 2 and replay["matches"] == replay["n"], replay

    from rsi.distill import distill_run
    from rsi.store import StrategyStore

    store = StrategyStore(knowledge_root)
    st = store.find_by_key("online_softmax")
    assert st is not None and st["stats"]["applied"] >= 1, st
    assert st["stats"]["calibration_mae"] is not None, st["stats"]
    assert any(str(e["trajectory"]).endswith("trajectory.json") for e in st["evidence"]), st["evidence"]
    assert store.combos, "expected a combination rule from the 2-practice attempt"
    combo = list(store.combos.values())[0]
    assert len(combo["members"]) >= 2, combo

    before = {k: v["stats"]["applied"] for k, v in store.strategies.items()}
    report = distill_run(run_dir, store=store)
    assert report["skipped"], report
    assert all(v["stats"]["applied"] == before[k] for k, v in store.strategies.items()), "double-counted evidence"


def check_rsi_attribution_precision(tmp):
    from rsi.distill import attribute_from_source, resolve_declared
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "attr_knowledge"))
    # natural-language declarations resolve to the seed practices
    assert resolve_declared("block tiling", store) == store.find_by_key("block_tiling")["id"]
    assert resolve_declared("multi-core split", store) == store.find_by_key("multi_core_split")["id"]
    assert resolve_declared("MLA absorption", store) == store.find_by_key("mla_absorption")["id"]
    assert resolve_declared("completely unknown practice xyz", store) is None

    feats = {"tags": ["attention", "mla", "compute-bound", "fp32"], "op_class": "attention",
             "dtypes": ["float32"]}
    # prose and identifier mentions must NOT count as usage
    source = ('"""solution that still materializes k_nope/v; absorption is left to the agent."""\n'
              "import torch\n\n\ndef run(x, w_uk, c_kv):\n    return x + 0.0\n")
    hits = attribute_from_source(source, store, feats)
    assert store.find_by_key("mla_absorption")["id"] not in hits, hits
    # real usage in code still matches
    source2 = "import torch\n\n\ndef run(x):\n    absorbed = x * 2\n    return absorbed\n"
    hits2 = attribute_from_source(source2, store, feats)
    assert store.find_by_key("mla_absorption")["id"] in hits2, hits2

    # combo ids expand to their member strategies instead of becoming junk entries
    from rsi.distill import expand_declared

    members = [store.find_by_key("permute_once")["id"], store.find_by_key("reduce_launch_overhead")["id"]]
    combo = store.upsert_combo(members, measured_speedup=12.0, task="t", trajectory="x")
    expanded = expand_declared([combo["id"], "unresolvable practice"], store)
    assert members[0] in expanded and members[1] in expanded, expanded
    assert "unresolvable practice" in expanded, expanded


def check_knowledge_blocks(tmp):
    from rsi import bottlenecks
    from rsi.exemplars import record_exemplar, render_exemplars, retrieve_exemplars
    from rsi.retrieve import priors_block, retrieve
    from rsi.store import StrategyStore

    root = os.path.join(tmp, "kb")
    row = record_exemplar(root, "decode_attention", "def run(q, k, v):\n    return q\n", 1.4,
                          hardware="ascend-910b", dtypes=["float32"], op_class="attention",
                          tags=["attention", "decode", "fp32"], strategies=["stg-0001"],
                          profile_top=["aten::bmm"])
    feats = {"name": "decode_attention", "tags": ["attention", "decode", "fp32"],
             "op_class": "attention", "dtypes": ["float32"]}
    got = retrieve_exemplars(root, feats, "ascend-910b", top_k=1)
    assert got and got[0]["id"] == row["id"], got
    block = render_exemplars(got)
    assert "decode_attention" in block and "1.40x" in block, block
    dup = record_exemplar(root, "decode_attention", row["source"], 1.6, hardware="ascend-910b")
    assert dup["id"] == row["id"] and dup["speedup"] >= 1.6, dup
    record_exemplar(root, "weak_task", "def run():\n    return 0\n", 0.8, hardware="ascend-910b")
    weak = retrieve_exemplars(root, {"name": "weak_task", "tags": [], "op_class": "", "dtypes": ["float32"]},
                              "ascend-910b", top_k=5)
    assert all(r["task"] != "weak_task" for r in weak), weak

    advice = bottlenecks.lookup({"top_ops": [{"op": "aten::_local_scalar_dense", "pct": 77.0}]})
    assert advice and "Host sync" in advice[0], advice
    advice2 = bottlenecks.lookup({"top_ops": [{"op": "aten::scatter_add_", "pct": 44.0}]})
    assert any("scatter_add" in item for item in advice2), advice2
    assert "playbook" in bottlenecks.render(advice)

    store = StrategyStore(root)
    strategy = store.find_by_key("permute_once")
    store.record_evidence(strategy["id"], "t", "moe_permute", 12.0, "success",
                          expected_speedup=4.0, hardware="ascend-910b", dtype="float32")
    feats2 = {"name": "moe_permute", "tags": ["moe", "permute", "memory-bound", "fp32"],
              "op_class": "moe", "dtypes": ["float32"]}
    matched = retrieve(store, feats2, "ascend-910b")
    block2 = priors_block(feats2, "ascend-910b", matched, [])
    assert "Calibration note" in block2, block2


def check_constraint_and_baseline(tmp):
    from harness.runner import run_evaluation

    req_task = write(os.path.join(tmp, "constraints", "req_task.py"), REQ_TASK_SOURCE)
    plain = write(os.path.join(tmp, "constraints", "plain.py"), "def run(x):\n    return x + 1.0\n")
    with_triton = write(os.path.join(tmp, "constraints", "triton_ish.py"),
                        "try:\n    import triton\n    import triton.language as tl\n"
                        "except ImportError:\n    triton = None\n    tl = None\n\n\n"
                        "def run(x):\n    return x + 1.0\n")
    common = dict(device="cpu", timeout=120, out_dir=os.path.join(tmp, "constraints_out"),
                  results_file=os.path.join(tmp, "constraint_results.jsonl"), quiet=True)

    blocked = run_evaluation(task=req_task, solution=plain, **common)
    assert blocked["status"] == "fail_constraint", blocked.get("status")
    assert "triton" in (blocked.get("error") or ""), blocked.get("error")

    # run-level requirement (--require triton) is merged in even when the task has none
    no_task_req = write(os.path.join(tmp, "constraints", "no_task_req.py"),
                        REQ_TASK_SOURCE.replace('        "require": ["triton"],\n', ""))
    cli_blocked = run_evaluation(task=no_task_req, solution=plain, require="triton", **common)
    assert cli_blocked["status"] == "fail_constraint", cli_blocked.get("status")
    assert cli_blocked["require"] == ["triton"], cli_blocked.get("require")

    allowed = run_evaluation(task=req_task, solution=with_triton, **common)
    assert allowed["status"] == "pass", (allowed.get("status"), allowed.get("error"))
    assert allowed["solution_features"]["uses_triton"] is True, allowed.get("solution_features")

    wrong_op = write(os.path.join(tmp, "constraints", "opbase_wrong.py"),
                     OPBASE_TEMPLATE.format(body="return x * 3.0"))
    ok_op = write(os.path.join(tmp, "constraints", "opbase_ok.py"),
                  OPBASE_TEMPLATE.format(body="return x + 1.0"))
    good_solution = write(os.path.join(tmp, "constraints", "good.py"), "def run(x):\n    return x + 1.0\n")

    res_wrong = run_evaluation(task=wrong_op, solution=good_solution, **common)
    assert res_wrong["status"] == "pass", res_wrong.get("error")
    check = res_wrong.get("baseline_check") or {}
    assert check.get("used") == "eager" and "rejected" in (check.get("fallback") or ""), check

    res_ok = run_evaluation(task=ok_op, solution=good_solution, **common)
    assert res_ok["status"] == "pass", res_ok.get("error")
    assert (res_ok.get("baseline_check") or {}).get("used") == "op", res_ok.get("baseline_check")


CARD_VEC = textwrap.dedent('''\
    ---
    type: Runbook
    title: Vector add 优化
    description: elementwise add 的内存访问优化:向量化加载、避免额外 pass,适用于 vector add / elementwise 场景
    tags: [topic.elementwise, topic.memory]
    status: stable
    platforms: [a2]
    updated_at: '2026-09-08T09:05:00Z'
    ---

    # Vector add 优化

    ## 优化 1:向量化加载
    vectorized load/store, avoid extra passes
    ''')

CARD_UNRELATED = textwrap.dedent('''\
    ---
    type: Doc
    title: 无关卡片
    description: unrelated card about a different subsystem
    tags: [topic.foo]
    status: draft
    platforms: [a2]
    ---

    body B
    ''')

CARD_TRITON_ATTN = textwrap.dedent('''\
    ---
    type: Runbook
    title: Attention softmax 融合与 tiling
    description: attention softmax 在 triton 下的 tiling 与 online softmax 优化,适用于 sparse attention 场景
    tags: [topic.compute.reduction, topic.tiling]
    status: stable
    platforms: [a2]
    updated_at: '2026-09-08T09:05:00Z'
    ---

    # Attention softmax 融合

    triton kernel recipe: BLOCK_M/BLOCK_N tiling, online softmax
    ''')


def check_cards_module(tmp):
    from rsi import cards as cards_mod

    kb = os.path.join(tmp, "kb")
    write(os.path.join(kb, "knowledge", "ops", "vec_add.md"), CARD_VEC)
    write(os.path.join(kb, "knowledge", "ops", "unrelated.md"), CARD_UNRELATED)
    write(os.path.join(kb, "knowledge", "model", "architecture_note.md"), CARD_UNRELATED)
    write(os.path.join(kb, "knowledge", "ops", "triton", "runbooks", "attn.md"), CARD_TRITON_ATTN)
    feats = {"name": "np_vector_add", "tags": ["elementwise", "memory-bound"],
             "op_class": "elementwise", "dtypes": ["float32"]}
    got = cards_mod.retrieve_cards(kb, feats, top_k=2)
    assert got and got[0]["id"].endswith("vec_add.md"), got
    assert got[0]["platforms"] == ["a2"], got[0]
    block = cards_mod.render_candidates(got)
    assert got[0]["id"] in block and "read on demand" in block, block
    body = cards_mod.read_card(kb, got[0]["id"])
    assert "向量化加载" in body, body[:200]
    assert cards_mod.read_card(kb, "../../etc/passwd") == ""
    assert cards_mod.read_card(kb, "knowledge/ops/missing.md") == ""

    # form-aware weighting: kernel-form tasks prefer /ops/triton material
    attn_feats = {"name": "sparse_flash_attention", "tags": ["attention", "sparse"],
                  "op_class": "attention", "dtypes": ["float32"]}
    ranked = cards_mod.retrieve_cards(kb, attn_feats, top_k=3, required_form="triton_kernel")
    assert ranked and "/ops/triton/" in ranked[0]["id"], ranked


def check_agent_kb_action(tmp):
    task_path = write(os.path.join(tmp, "kb_tasks", "np_vector_add.py"), NP_TASK_SOURCE)
    kb = os.path.join(tmp, "kb_loop")
    write(os.path.join(kb, "knowledge", "ops", "vec_add.md"), CARD_VEC)
    replay = write(os.path.join(tmp, "kb_replay.json"), json.dumps({
        "responses": [
            {"action": "knowledge", "id": "knowledge/ops/vec_add.md"},
            {"action": "write_solution", "content": GOOD_SOLUTION},
            {"action": "finish", "summary": "kb self-test"},
        ]
    }, ensure_ascii=False))
    out_dir = os.path.join(tmp, "kb_runs")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--hardware", "ascend-910b",
        "--kb-root", kb, "--kb-cards", "3",
        "--knowledge-root", os.path.join(tmp, "kb_knowledge"), "--no-distill",
        "--out-dir", out_dir, "--results-file", os.path.join(tmp, "kb_results.jsonl"),
        "--max-iters", "4", "--target-speedup", "0.5", "--timeout", "180",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:])
    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["rsi"]["kb_reads"] == ["knowledge/ops/vec_add.md"], meta["rsi"]["kb_reads"]
    assert any(str(cid).endswith("vec_add.md") for cid in meta["rsi"]["kb_candidates"]), meta["rsi"]
    with open(os.path.join(run_dir, "transcript.md"), encoding="utf-8") as fh:
        transcript = fh.read()
    assert "CANN knowledge cards" in transcript, "candidates block missing"
    assert "Knowledge card knowledge/ops/vec_add.md" in transcript, "card read block missing"
    assert "向量化加载" in transcript, "card body missing from the transcript"


def check_form_and_nudge(tmp):
    from rsi.forms import form_for, form_hint

    assert form_for({"op_class": "attention"}) == "triton_kernel"
    assert form_for({"op_class": "moe"}) == "torch_fused_first"
    assert form_for({"op_class": "unknownclass"}) == ""
    assert "22.3x" in form_hint({"op_class": "attention"}), form_hint({"op_class": "attention"})
    assert "12.3x" in form_hint({"op_class": "moe"}), form_hint({"op_class": "moe"})

    # the loop must not accept "finish" while no solution has passed and budget remains
    task_path = write(os.path.join(tmp, "form_tasks", "np_vector_add.py"), NP_TASK_SOURCE)
    replay = write(os.path.join(tmp, "form_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": WRONG_SOLUTION},
            {"action": "finish", "summary": "giving up"},
            {"action": "write_solution", "content": GOOD_SOLUTION},
            {"action": "finish", "summary": "done"},
        ]
    }, ensure_ascii=False))
    out_dir = os.path.join(tmp, "form_runs")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--hardware", "ascend-910b",
        "--knowledge-root", os.path.join(tmp, "form_knowledge"), "--no-distill",
        "--kb-cards", "0",
        "--out-dir", out_dir, "--results-file", os.path.join(tmp, "form_results.jsonl"),
        "--max-iters", "4", "--target-speedup", "0.5", "--timeout", "180",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, (proc.stdout[-1200:] + "\n" + proc.stderr[-1200:])
    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["best"]["status"] == "pass", meta["best"]
    assert meta["rsi"]["nudges"] == 1, meta["rsi"]
    assert meta["rsi"]["form"] == "torch_fused_first", meta["rsi"]["form"]
    with open(os.path.join(run_dir, "transcript.md"), encoding="utf-8") as fh:
        transcript = fh.read()
    assert "do not stop" in transcript, "nudge message missing"
    assert "Implementation form prior" in transcript, "form hint block missing"


DEMO_SOLUTION = textwrap.dedent('''\
    import torch

    try:
        import triton
        import triton.language as tl
        HAVE_TRITON = True
    except Exception:
        HAVE_TRITON = False


    def run(x, y):
        if not HAVE_TRITON or not x.is_cuda:
            return torch.matmul(x, y)   # banned + must be stripped for submission
        out = torch.empty_like(x)
        return out
    ''')

DEMO_SCHEMA = "demo_add(Tensor x, Tensor y) -> Tensor z"


def check_profile_backend(tmp):
    from harness.profile_runner import profile

    res = profile(
        task=os.path.join(ROOT, "tasks", "level1", "vector_add.py"),
        solution=os.path.join(ROOT, "solutions", "examples", "vector_add_good.py"),
        device="cpu", backend="torch", out_dir=os.path.join(tmp, "profiling"), timeout=300, iters=2,
    )
    assert res["status"] == "ok", (res.get("status"), res.get("error"))
    assert res["summary"]["top_ops"], res["summary"]
    assert "top ops" in res["text"], res["text"]


def check_agent_profiling(tmp):
    replay = write(os.path.join(tmp, "prof_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": "def run(x, y):\n    return x + y\n"},
            {"action": "finish", "summary": "profiling self-test"},
        ]
    }, ensure_ascii=False))
    out_dir = os.path.join(tmp, "prof_runs")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", os.path.join(ROOT, "tasks", "level1", "vector_add.py"),
        "--provider", "replay", "--replay-file", replay, "--device", "cpu", "--no-rsi",
        "--out-dir", out_dir, "--results-file", os.path.join(tmp, "prof_results.jsonl"),
        "--max-iters", "3", "--target-speedup", "0.5", "--timeout", "300",
        "--profiler", "torch", "--profile-every", "1",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:])
    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(os.path.join(run_dir, "transcript.md"), encoding="utf-8") as fh:
        transcript = fh.read()
    assert "Profiling" in transcript, "profiling block missing from the feedback"
    with open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") as fh:
        attempts = [json.loads(line) for line in fh if line.strip()]
    assert attempts and attempts[0].get("profile"), attempts
    assert attempts[0]["profile"].get("top_ops"), attempts[0]["profile"]


def check_ascend_tasks(tmp):
    from harness.runner import run_evaluation

    cases = [
        ("tasks/level2/sparse_flash_attention.py", "solutions/examples/sparse_flash_attention_torch.py"),
        ("tasks/level2/mla_attention.py", "solutions/examples/mla_attention_torch.py"),
        ("tasks/level2/moe_finalize_routing.py", "solutions/examples/moe_finalize_routing_torch.py"),
        ("tasks/level2/grouped_matmul.py", "solutions/examples/grouped_matmul_torch.py"),
        ("tasks/level2/moe_permute.py", "solutions/examples/moe_permute_torch.py"),
        ("tasks/level2/grouped_gemm_swiglu.py", "solutions/examples/grouped_gemm_swiglu_torch.py"),
        ("tasks/level2/rope.py", "solutions/examples/rope_torch.py"),
        ("tasks/level2/decode_attention.py", "solutions/examples/decode_attention_torch.py"),
        ("tasks/level2/rmsnorm_rope_fused.py", "solutions/examples/rmsnorm_rope_fused_torch.py"),
        ("tasks/level2/quantized_matmul_w8a8.py", "solutions/examples/quantized_matmul_w8a8_torch.py"),
    ]
    failures = []
    for task_rel, solution_rel in cases:
        res = run_evaluation(
            task=os.path.join(ROOT, task_rel), solution=os.path.join(ROOT, solution_rel),
            device="cpu", timeout=600, out_dir=os.path.join(tmp, "ascend_tasks"),
            results_file=os.path.join(tmp, "ascend_results.jsonl"), quiet=True,
        )
        if res.get("status") != "pass":
            failures.append((task_rel, res.get("status"), res.get("error")))
    assert not failures, failures


def main():
    tmp = tempfile.mkdtemp(prefix="kernel_agent_selftest_")
    print("temp dir: %s" % tmp)
    files = make_fixture_files(tmp)

    check("profilers.ncu parses an ncu --csv export", lambda: check_ncu(tmp))
    check("profilers.msprof parses an msprof export", lambda: check_msprof(tmp))
    check("pipeline: correct solution passes", lambda: check_pipeline_pass(tmp, files))
    check("pipeline: wrong solution fails correctness", lambda: check_pipeline_wrong(tmp, files))
    check("pipeline: input mutation is rejected", lambda: check_pipeline_mutation(tmp, files))
    check("pipeline: hanging solution times out", lambda: check_pipeline_timeout(tmp, files))
    check("report: pass@1 table", lambda: check_report(tmp))
    check("agent loop: replay (wrong -> good -> finish)", lambda: check_agent_replay(tmp, files))
    check("rsi: strategy pool bootstrap + retrieval priors", lambda: check_rsi_store_and_retrieve(tmp))
    check("rsi: applicability conditions narrow with evidence", lambda: check_rsi_condition_refinement(tmp))
    check("rsi: loop records trajectory, distills, reuses priors", lambda: check_rsi_loop_distill(tmp))
    check("rsi: attribution precision (declarations resolve, prose ignored)", lambda: check_rsi_attribution_precision(tmp))
    check("rsi: knowledge blocks (exemplars, bottleneck playbook, calibration)", lambda: check_knowledge_blocks(tmp))
    check("harness: require-triton gate + op-baseline sanity check", lambda: check_constraint_and_baseline(tmp))
    check("cann knowledge: card index / retrieval / on-demand read", lambda: check_cards_module(tmp))
    check("cann knowledge: agent reads a card in the loop", lambda: check_agent_kb_action(tmp))
    check("forms: per-family form prior + early-finish nudge", lambda: check_form_and_nudge(tmp))

    if importlib.util.find_spec("torch") is not None:
        check("profiling: torch backend produces an op breakdown", lambda: check_profile_backend(tmp))
        check("profiling: agent feedback carries the breakdown", lambda: check_agent_profiling(tmp))
        check("torch: vector_add good/wrong examples", lambda: check_torch_examples(tmp))
        check("torch: softmax_manual example", lambda: check_torch_softmax(tmp))
        check("real-world tasks: example solutions pass on CPU",
          lambda: check_ascend_tasks(tmp))
    else:
        print("SKIP  torch-dependent checks (torch not installed)")

    print()
    print("=" * 64)
    print("passed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    for name, exc in RESULTS["fail"]:
        print("  FAILED: %s (%s)" % (name, exc))
    print("temp artifacts: %s" % tmp)
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())