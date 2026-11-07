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
# Matches both the online_softmax detector (max+exp) and block_tiling (TILE).
SOFTMAX_SOURCE = "\n".join([
    "import torch", "", "TILE = 1024", "",
    "def run(x):",
    "    m = x.max(dim=-1, keepdim=True).values",
    "    e = torch.exp(x - m)",
    "    return e / e.sum(dim=-1, keepdim=True)",
]) + "\n"

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

P2_TASK = textwrap.dedent('''
    """Deliberately slow reference so a torch.softmax solution is a real win."""
    import torch

    TASK = {
        "name": "p2_softmax_slow_ref",
        "level": 1,
        "description": "Row-wise softmax with a slow reference (P2 ablation self-test).",
        "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "num_trials": 2,
        "warmup": 5,
        "rep": 100,
        "input_specs": [{"kind": "randn", "shape": [2048, 4096], "dtype": "float32"}],
        "entrypoint": "run",
        "baseline": "eager",
        "tags": ["softmax", "reduction", "memory-bound"],
    }

    def reference(x):
        m = x.max(dim=-1, keepdim=True).values
        e = torch.exp(x - m)
        return e / e.sum(dim=-1, keepdim=True)
''')


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
    assert res["status"] in ("pass", "pass_unstable"), res.get("error")
    assert res["correctness"]["ok"] is True
    assert res["timing"]["speedup"] > 0, res["timing"]
    timing = res["timing"]
    assert timing["n_rounds"] == 5, timing
    assert len(timing["speedup_ci"]) == 2, timing
    assert len(timing["rounds"]) == 5, timing
    assert timing["speedup_ci"][0] <= timing["speedup"] <= timing["speedup_ci"][1], timing


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
    assert meta["best"]["status"] in ("pass", "pass_unstable"), meta
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
    assert good["status"] in ("pass", "pass_unstable"), good.get("error")
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
    assert res["status"] in ("pass", "pass_unstable"), res.get("error")


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
    # P1.4: one successful run is not a reproduction -- it stays a candidate.
    assert st["confidence"] == "candidate", st["confidence"]


def check_verified_two_runs(tmp):
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "verify_kb"))
    st = store.find_by_key("block_tiling")
    store.record_evidence(st["id"], "runs/a/trajectory.json", "t", 1.4, "success",
                          run_id="run_a", holdout_ok=True)
    assert st["confidence"] == "candidate", st["confidence"]
    store.record_evidence(st["id"], "runs/a/trajectory.json", "t", 1.4, "success",
                          run_id="run_a", holdout_ok=True)
    assert st["confidence"] == "candidate", "the same run must not verify a strategy"
    store.record_evidence(st["id"], "runs/b/trajectory.json", "t", 1.4, "success",
                          run_id="run_b", holdout_ok=True)
    assert st["confidence"] == "verified", st["confidence"]


def check_statistical_gate(tmp):
    from harness import noise, stats

    assert stats.percentile([0.0, 10.0], 50) == 5.0, stats.percentile([0.0, 10.0], 50)
    assert stats.percentile([], 50) is None
    lo, hi = stats.bootstrap_ci([1.0, 1.1, 0.9, 1.05, 1.0])
    assert lo <= 1.0 <= hi, (lo, hi)

    assert noise.classify([1.20, 1.30], 0.05) == "success"
    assert noise.classify([0.70, 0.80], 0.05) == "regression"
    assert noise.classify([0.98, 1.03], 0.05) == "neutral"
    assert noise.classify(None, 0.05) == "neutral"
    assert noise.classify([1.20, 1.30], None) == "neutral"
    assert noise.promotion_floor(0.05) == 1.1

    path = write(os.path.join(tmp, "noise.json"), json.dumps({
        "schema_version": 1, "devices": {"cpu": {"epsilon": 0.07}}}))
    assert noise.epsilon_for(device="cpu", path=path) == 0.07
    assert noise.epsilon_for(device="npu", path=path) is None
    # the actual measurement device wins over a claimed hardware label
    assert noise.device_key(device="cpu", hardware="ascend-910b") == "cpu"
    assert noise.device_key(device="npu", hardware="cpu") == "ascend-910b"


def check_noise_config(tmp):
    from harness import noise

    data = noise.load_noise()
    devices = data.get("devices") or {}
    for key in ("cpu", "ascend-910b"):
        entry = devices.get(key) or {}
        assert entry.get("epsilon") is not None and float(entry["epsilon"]) > 0, (key, entry)
        assert int(entry.get("n_runs") or 0) >= 20, (key, entry)
        validation = entry.get("validation") or {}
        assert int(validation.get("success_count", 99)) <= 1, (key, validation)


def check_distill_gate(tmp):
    """P1.3 + P2.2: a CI success counts only with a supporting ablation."""
    from rsi.distill import distill_run
    from rsi.store import StrategyStore

    run_dir = os.path.join(tmp, "gate_run")
    source = "\n".join([
        "import torch",
        "",
        "",
        "def run(x):",
        "    m = x.max(dim=-1, keepdim=True).values",
        "    e = torch.exp(x - m)",
        "    return e / e.sum(dim=-1, keepdim=True)",
    ]) + "\n"
    fast_path = write(os.path.join(run_dir, "fast.py"), source)
    neutral_path = write(os.path.join(run_dir, "neutral.py"), source)
    meta = {
        "task": "np_softmax", "device": "cpu", "hardware": "cpu",
        "features": {"tags": ["softmax", "reduction"], "op_class": "reduction",
                     "dtypes": ["float32"]},
        "best": {"status": "pass", "speedup": 1.5, "path": fast_path},
    }
    write(os.path.join(run_dir, "meta.json"), json.dumps(meta))
    base = {"strategies_declared": ["online softmax"], "solution_source": source,
            "stability_delta": 0.01, "distilled": False}
    traj = {
        "task": "np_softmax", "task_file": "", "hardware": "cpu", "device": "cpu",
        "features": meta["features"], "best": meta["best"], "distilled": False,
        "attempts": [
            dict(base, iter=1, status="pass", speedup=1.5, speedup_ci=[1.45, 1.55],
                 solution=fast_path),
            dict(base, iter=2, status="pass", speedup=1.0, speedup_ci=[0.98, 1.02],
                 solution=neutral_path),
            {"iter": "ablation:stg-0001", "kind": "ablation", "ablation_of": "stg-0001",
             "status": "pass", "speedup": 0.55, "speedup_ci": [0.50, 0.60],
             "solution_source": source, "distilled": False},
        ],
    }
    write(os.path.join(run_dir, "trajectory.json"), json.dumps(traj))
    store = StrategyStore(os.path.join(tmp, "gate_kb"))
    report = distill_run(run_dir, store=store)
    outcomes = [row["outcome"] for row in report["evidence"]]
    assert "success" in outcomes, report["evidence"]
    assert "neutral" in outcomes, report["evidence"]
    assert "failure" not in outcomes, report["evidence"]
    success = next(row for row in report["evidence"] if row["outcome"] == "success")
    assert success["ablation"].get("without_speedup_ci") == [0.5, 0.6], success
    st = store.find_by_key("online_softmax")
    assert st["confidence"] == "candidate", st["confidence"]

def check_unstable_kept(tmp):
    """P1 fix: instability widens the CI, it does not discard a real gain."""
    from rsi.distill import distill_run
    from rsi.store import StrategyStore

    run_dir = os.path.join(tmp, "unstable_run")
    best_path = write(os.path.join(run_dir, "best.py"), SOFTMAX_SOURCE)
    weak_path = write(os.path.join(run_dir, "weak.py"), SOFTMAX_SOURCE)
    meta = {
        "task": "np_softmax", "device": "cpu", "hardware": "cpu",
        "features": {"tags": ["softmax", "reduction"], "op_class": "reduction",
                     "dtypes": ["float32"]},
        "best": {"status": "pass_unstable", "speedup": 4.0, "path": best_path},
    }
    write(os.path.join(run_dir, "meta.json"), json.dumps(meta))
    traj = {
        "task": "np_softmax", "task_file": "", "hardware": "cpu", "device": "cpu",
        "features": meta["features"], "best": meta["best"], "distilled": False,
        "attempts": [
            # unstable, but the (already widened) interval clears 1 + epsilon
            {"iter": 1, "status": "pass_unstable", "speedup": 4.0,
             "speedup_ci": [3.79, 4.21], "speedup_ci_raw": [3.9, 4.1],
             "stability_delta": 0.046, "epsilon": 0.043, "solution": best_path,
             "strategies_declared": ["online softmax"],
             "solution_source": SOFTMAX_SOURCE, "distilled": False},
            # unstable and genuinely inconclusive: must stay neutral
            {"iter": 2, "status": "pass_unstable", "speedup": 1.01,
             "speedup_ci": [0.80, 1.02], "stability_delta": 0.09, "epsilon": 0.043,
             "solution": weak_path,
             "strategies_declared": ["block tiling"],
             "solution_source": SOFTMAX_SOURCE, "distilled": False},
            # ablation evidence: removing online softmax hurts a lot ...
            {"iter": "ablation:stg-0001", "kind": "ablation", "ablation_of": "stg-0001",
             "status": "pass", "speedup": 0.5, "speedup_ci": [0.45, 0.55],
             "solution_source": SOFTMAX_SOURCE, "distilled": False},
            # ... but removing block tiling does not (it was already neutral)
            {"iter": "ablation:stg-0002", "kind": "ablation", "ablation_of": "stg-0002",
             "status": "pass", "speedup": 4.1, "speedup_ci": [4.0, 4.2],
             "solution_source": SOFTMAX_SOURCE, "distilled": False},
        ],
    }
    write(os.path.join(run_dir, "trajectory.json"), json.dumps(traj))
    store = StrategyStore(os.path.join(tmp, "unstable_kb"))
    report = distill_run(run_dir, store=store)
    assert not report["skipped"], report["skipped"]
    assert len(report.get("unstable_kept") or []) == 2, report
    outcomes = {row["strategy"]: row["outcome"] for row in report["evidence"]}
    online = store.find_by_key("online_softmax")["id"]
    tiling = store.find_by_key("block_tiling")["id"]
    assert outcomes.get(online) == "success", report["evidence"]
    assert outcomes.get(tiling) == "neutral", report["evidence"]
    assert store.find_by_key("online_softmax")["stats"]["succeeded"] == 1
def check_unstable_candidates_holdout_none(tmp):
    """holdout=None (unconfigured task) may reach candidate, never verified."""
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "holdout_none_kb"))
    st = store.find_by_key("block_tiling")
    for index in range(3):
        store.record_evidence(st["id"], "run-%d" % index, "t", 1.4, "success",
                              run_id="run-%d" % index, holdout_ok=None)
    assert st["confidence"] == "candidate", st["confidence"]
def check_rsi_loop_distill(tmp):
    # torch softmax is used here because a ~0.05 ms numpy kernel is genuinely
    # unstable on a CPU (the P1 stability gate would skip it); the softmax
    # task is long enough for the gate to be meaningful.
    task_path = os.path.join(ROOT, "tasks", "level1", "softmax.py")
    wrong = "\n".join([
        "import torch",
        "",
        "",
        "def run(x):",
        "    m = torch.max(x, dim=-1, keepdim=True).values",
        "    return torch.exp(x - m)",
    ]) + "\n"
    good = "\n".join([
        "import torch",
        "",
        "",
        "def run(x):",
        "    m = x.max(dim=-1, keepdim=True).values",
        "    e = torch.exp(x - m)",
        "    return e / e.sum(dim=-1, keepdim=True)",
    ]) + "\n"
    replay = write(os.path.join(tmp, "rsi_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": wrong,
             "strategies": ["online softmax"], "expected_speedup": 1.3},
            {"action": "write_solution", "content": good,
             "strategies": ["online softmax"], "expected_speedup": 1.0},
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
        "--profiler", "off",
        # mechanics test: this run only exercises distillation, not performance
        # evidence, so it must opt in explicitly (strict provenance is the default).
        "--allow-mechanics-evidence",
        "--no-ablate",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])

    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["best"]["status"] in ("pass", "pass_unstable"), meta["best"]

    with open(os.path.join(run_dir, "transcript.md"), encoding="utf-8") as fh:
        transcript = fh.read()
    assert "Tuning priors" in transcript, "priors block missing from the prompt"
    assert meta["rsi"]["priors_matched"], meta["rsi"]
    assert meta["rsi"]["attempted"], meta["rsi"]

    with open(os.path.join(run_dir, "trajectory.json"), encoding="utf-8") as fh:
        traj = json.load(fh)
    assert len(traj["attempts"]) == 2, traj["attempts"]
    assert traj["task_file"] and os.path.isfile(traj["task_file"]), traj["task_file"]
    unstable = any(attempt.get("status") == "pass_unstable" for attempt in traj["attempts"])

    from rsi.trajectory import replay_trajectory

    replay = replay_trajectory(run_dir, device="cpu")
    assert replay["n"] == 2 and replay["matches"] == replay["n"], replay

    from rsi.distill import distill_run
    from rsi.store import StrategyStore

    store = StrategyStore(knowledge_root)
    st = store.find_by_key("online_softmax")
    if unstable:
        # P1.5: a noisy attempt is not distilled; the failing attempt still
        # records its failure evidence.
        distill = meta["rsi"]["distill"]
        assert distill["skipped"] == 1, meta["rsi"]
        assert distill["evidence"] == 1, meta["rsi"]
        with open(os.path.join(run_dir, "distill_report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        assert any("unstable" in row["reason"] for row in report["skipped"]), report
        assert st is not None, store.strategies
        assert not any(e.get("outcome") == "success" for e in (st.get("evidence") or [])), st
    else:
        assert meta["rsi"]["distill"] and meta["rsi"]["distill"]["evidence"] >= 2, meta["rsi"]
        assert st is not None and st["stats"]["applied"] >= 1, st
        assert st["stats"]["calibration_mae"] is not None, st["stats"]
        assert any(str(e["trajectory"]).endswith("trajectory.json") for e in st["evidence"]), st["evidence"]
        # P2.4: a single attributed practice cannot form a combo, and a
        # combo without ablation evidence is not created at all.
        assert not store.combos, "combo created without ablation evidence"

    before = {k: v["stats"]["applied"] for k, v in store.strategies.items()}
    report = distill_run(run_dir, store=store, strict_provenance=False)
    assert report["skipped"], report
    assert all(v["stats"]["applied"] == before[k] for k, v in store.strategies.items()), "double-counted evidence"


def check_strict_provenance(tmp):
    """A CPU mechanics run labelled ascend-910b must not write performance evidence."""
    task_path = os.path.join(ROOT, "tasks", "level1", "softmax.py")
    good = "\n".join([
        "import torch",
        "",
        "",
        "def run(x):",
        "    m = x.max(dim=-1, keepdim=True).values",
        "    e = torch.exp(x - m)",
        "    return e / e.sum(dim=-1, keepdim=True)",
    ]) + "\n"
    replay = write(os.path.join(tmp, "prov_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": good,
             "strategies": ["online softmax"], "expected_speedup": 1.0},
            {"action": "finish", "summary": "strict provenance self-test"},
        ]
    }, ensure_ascii=False))

    def run_loop(name, extra):
        out_dir = os.path.join(tmp, name + "_runs")
        root = os.path.join(tmp, name + "_knowledge")
        cmd = [
            sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
            "--replay-file", replay, "--device", "cpu", "--hardware", "ascend-910b",
            "--knowledge-root", root, "--out-dir", out_dir,
            "--results-file", os.path.join(tmp, name + "_results.jsonl"),
            "--max-iters", "4", "--target-speedup", "0.5", "--timeout", "180",
            "--profiler", "off", "--no-ablate",
        ] + list(extra)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
        assert proc.returncode == 0, (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:])
        metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
        assert len(metas) == 1, metas
        with open(metas[0], encoding="utf-8") as fh:
            meta = json.load(fh)
        path = os.path.join(root, "strategies.jsonl")
        rows = []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        return meta, rows, root

    meta, rows, root = run_loop("prov_strict", [])
    distill = meta["rsi"]["distill"] or {}
    assert distill.get("provenance_valid") is False, distill
    assert distill.get("evidence") == 0, distill
    assert all(not row.get("evidence") for row in rows), "mechanics evidence leaked into the pool"
    assert not os.path.isfile(os.path.join(root, "exemplars.jsonl")), "mechanics exemplar recorded"

    meta2, _rows2, _root2 = run_loop("prov_allow", ["--allow-mechanics-evidence"])
    assert meta2["rsi"]["strict_provenance"] is False, meta2["rsi"]

    # The end-to-end allow run can legitimately be pass_unstable on a noisy CPU
    # (and is then skipped regardless of provenance), so the provenance switch
    # itself is verified deterministically at the distill layer.
    def synth_run(name):
        source = "\n".join([
            "import torch", "", "",
            "def run(x):",
            "    m = x.max(dim=-1, keepdim=True).values",
            "    e = torch.exp(x - m)",
            "    return e / e.sum(dim=-1, keepdim=True)",
        ]) + "\n"
        run_dir = os.path.join(tmp, name)
        meta = {"task": "softmax", "device": "cpu", "hardware": "ascend-910b",
                "features": {"tags": ["softmax"], "op_class": "reduction",
                             "dtypes": ["float32"]},
                "best": {"status": "pass", "speedup": 1.0}}
        write(os.path.join(run_dir, "meta.json"), json.dumps(meta))
        traj = {"task": "softmax", "task_file": "", "hardware": "ascend-910b",
                "device": "cpu", "features": meta["features"], "distilled": False,
                "attempts": [{
                    "iter": 1, "status": "pass", "speedup": 1.0,
                    "speedup_ci": [0.99, 1.01], "stability_delta": 0.01,
                    "strategies_declared": ["online softmax"],
                    "solution_source": source, "distilled": False,
                }]}
        write(os.path.join(run_dir, "trajectory.json"), json.dumps(traj))
        return run_dir

    from rsi.distill import distill_run
    from rsi.store import StrategyStore

    blocked_store = StrategyStore(os.path.join(tmp, "prov_synth_strict_kb"))
    blocked = distill_run(synth_run("prov_synth_strict"), store=blocked_store,
                          strict_provenance=True)
    assert not blocked["evidence"], blocked
    assert any("provenance" in row["reason"] for row in blocked["skipped"]), blocked

    allow_store = StrategyStore(os.path.join(tmp, "prov_synth_allow_kb"))
    allowed = distill_run(synth_run("prov_synth_allow"), store=allow_store,
                          strict_provenance=False)
    assert allowed["evidence"], allowed
    evidence = [e for row in allow_store.strategies.values() for e in (row.get("evidence") or [])]
    assert any(e.get("hardware") == "ascend-910b" for e in evidence), evidence


def check_p3_verified_holdout(tmp):
    """P3.1: a train-only success cannot verify a strategy."""
    from rsi.store import StrategyStore

    for holdout_ok, expected in ((False, "candidate"), (True, "verified")):
        store = StrategyStore(os.path.join(tmp, "p3_holdout_%s" % holdout_ok))
        st = store.find_by_key("block_tiling")
        store.record_evidence(st["id"], "a", "t", 1.4, "success",
                              run_id="r1", holdout_ok=holdout_ok)
        store.record_evidence(st["id"], "b", "t", 1.4, "success",
                              run_id="r2", holdout_ok=holdout_ok)
        assert st["confidence"] == expected, (holdout_ok, st["confidence"])


def check_p3_holdout_measurement(tmp):
    """P3.1: a solution that is only faster on the training shape fails the holdout gate."""
    from harness.runner import run_evaluation
    from harness.tasks import load_task

    root = os.path.join(tmp, "p3_holdout_task")
    task_path = write(os.path.join(root, "task.py"), textwrap.dedent('''
        import torch
        TASK = {"name": "p3_slow_ref", "level": 1,
                "tolerance": {"atol": 1e-5, "rtol": 1e-5},
                "num_trials": 2, "warmup": 5, "rep": 100,
                "input_specs": [{"kind": "randn", "shape": [2048, 4096], "dtype": "float32"}],
                "entrypoint": "run", "baseline": "eager", "tags": ["softmax", "reduction"]}
        def reference(x):
            m = x.max(dim=-1, keepdim=True).values
            e = torch.exp(x - m)
            return e / e.sum(dim=-1, keepdim=True)
    '''))
    solution = write(os.path.join(root, "solution.py"), textwrap.dedent('''
        import torch
        def run(x):
            if tuple(x.shape) == (2048, 4096):
                return torch.softmax(x, dim=-1)
            m = x.max(dim=-1, keepdim=True).values
            e = torch.exp(x - m)
            return e / e.sum(dim=-1, keepdim=True)
    '''))
    config = write(os.path.join(root, "holdout.json"), json.dumps({
        "schema_version": 1,
        "tasks": {"p3_slow_ref": {"cases": [
            {"name": "holdout-1024x4096", "shapes": [[1024, 4096]]}]}},
    }))
    task = load_task(task_path, holdout_config=config)
    assert task.holdout_cases, task.holdout_cases
    shapes = [tuple(x.shape) for x in task.holdout_inputs(task.holdout_cases[0], "cpu", 0)]
    assert shapes == [(1024, 4096)], shapes
    res = run_evaluation(
        task=task_path, solution=solution, device="cpu", timeout=900,
        holdout_config=config, out_dir=os.path.join(root, "out"),
        results_file=os.path.join(root, "results.jsonl"), append=False, quiet=True)
    hold = (res.get("timing") or {}).get("holdout") or {}
    assert hold.get("ok") is True, hold
    assert hold.get("speedup_ci"), hold
    assert hold.get("success") is False, hold
    assert res["timing"]["speedup"] > 2.0, res["timing"]["speedup"]


def check_p3_shape_refinement(tmp):
    from rsi.memory import shape_signature
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "p3_shape_kb"))
    st = store.find_by_key("block_tiling")
    aligned = shape_signature((2048, 4096), -1, "float32")
    prime = shape_signature((4099, 4096), -1, "float32")
    for index in range(2):
        store.record_evidence(st["id"], "aligned-%d" % index, "t", 1.4, "success",
                              hardware="ascend-910b", dtype="float32",
                              run_id="aligned-%d" % index, shape_signature=aligned)
        store.record_evidence(st["id"], "prime-%d" % index, "t", 0.8, "failure",
                              hardware="ascend-910b", dtype="float32",
                              run_id="prime-%d" % index, shape_signature=prime)
    note = store.refine_conditions(st["id"])
    assert note and "shape->" in note, note
    shape = st["conditions"].get("shape") or {}
    assert shape.get("prime_outer") is False, shape
    assert shape.get("half_aligned_32b") is True, shape


def check_p3_retrieval_calibration(tmp):
    from rsi.retrieve import retrieve
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "p3_quality_kb"))
    alpha = store.upsert(statement="alpha strategy", key="alpha",
                         tags=["softmax", "reduction"], op_class=["reduction"],
                         keywords=["alpha"])
    beta = store.upsert(statement="beta strategy", key="beta",
                        tags=["softmax", "reduction"], op_class=["reduction"],
                        keywords=["beta"])
    for strategy in (alpha, beta):
        strategy["stats"].update({"applied": 10, "succeeded": 8, "ema_speedup": 1.2})
    alpha["stats"]["calibration_mae"] = 0.1
    beta["stats"]["calibration_mae"] = 1.0
    feats = {"tags": ["softmax", "reduction"], "op_class": "reduction", "dtypes": ["float32"]}
    ranked = retrieve(store, feats, "cpu", top_k=10, min_match=0)
    keys = [strategy.get("key") for strategy, _score, _match in ranked]
    assert "alpha" in keys and "beta" in keys, keys
    assert keys.index("alpha") < keys.index("beta"), keys


def check_p3_exemplar_archive(tmp):
    from rsi.exemplars import record_exemplar, retrieve_exemplars, update_descendant_gain

    root = os.path.join(tmp, "p3_exemplars")
    torch_src = "import torch\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n"
    triton_src = ("import triton\nimport triton.language as tl\n\n"
                  "@triton.jit\ndef k():\n    pass\n\n\ndef run(x):\n    return x\n")
    fast = record_exemplar(root, "softmax", torch_src, 1.6, hardware="cpu",
                           profile_top=["aten::softmax"], tags=["softmax"],
                           op_class="reduction")
    slow = record_exemplar(root, "softmax", triton_src, 1.2, hardware="cpu",
                           profile_top=["aten::softmax"], tags=["softmax"],
                           op_class="reduction")
    feats = {"name": "softmax", "tags": ["softmax"], "op_class": "reduction",
             "dtypes": ["float32"]}
    got = retrieve_exemplars(root, feats, "cpu", top_k=5)
    forms = {row.get("form") for row in got}
    assert {"torch_fused", "triton"} <= forms, got
    assert fast["descendant_gain"] == 0.0 and slow["descendant_gain"] == 0.0, (fast, slow)
    # the same cell keeps only the fastest version
    assert record_exemplar(root, "softmax", torch_src, 1.1, hardware="cpu",
                           profile_top=["aten::softmax"])["speedup"] == 1.6
    update_descendant_gain(root, [fast["id"]], 1.9)
    update_descendant_gain(root, [fast["id"]], 1.7)
    gain = next(row for row in retrieve_exemplars(root, feats, "cpu", top_k=5)
                if row["id"] == fast["id"])["descendant_gain"]
    assert gain == 0.3, gain


def check_p3_evolve_holdout(tmp):
    from rsi.evolve import holdout_label

    assert holdout_label("softmax") == "holdout-1024x4096", holdout_label("softmax")
    assert holdout_label("no_such_task") == "shape-x0.5"
def _p2_sources():
    full = """import torch


def run(x):
    y = x.contiguous()
    return torch.softmax(y, dim=-1)
"""
    # removes torch.softmax, keeps .contiguous()
    no_fused = """import torch


def run(x):
    y = x.contiguous()
    m = y.max(dim=-1, keepdim=True).values
    e = torch.exp(y - m)
    return e / e.sum(dim=-1, keepdim=True)
"""
    # removes .contiguous(), keeps torch.softmax() and does the work twice
    no_contig = """import torch


def run(x):
    y = torch.softmax(x, dim=-1)
    for _ in range(5):
        y = y.clone()   # stands in for the copies the fast path avoids
    return y
"""
    # invalid ablation: claims to remove contiguous_input but keeps the marker
    invalid = full
    return full, no_fused, no_contig, invalid


def _seed_p2_store(root):
    from rsi.store import StrategyStore

    store = StrategyStore(root)
    store.upsert(statement="use torch fused softmax", key="fused_softmax",
                 tags=["softmax", "reduction"], op_class=["reduction"],
                 keywords=["fused softmax"], expected=2.0,
                 detector={"kind": "regex", "any": [r"torch\.softmax\("]})
    store.upsert(statement="make the hot tensor contiguous first",
                 key="contiguous_input",
                 tags=["softmax", "reduction"], op_class=["reduction"],
                 keywords=["contiguous input"], expected=1.2,
                 detector={"kind": "regex", "any": [r"\.contiguous\("]})
    store.save()
    return store


def _run_p2_loop(tmp, name, task_path, replay_obj, extra=()):
    replay = write(os.path.join(tmp, name + "_replay.json"),
                   json.dumps(replay_obj, ensure_ascii=False))
    out_dir = os.path.join(tmp, name + "_runs")
    root = os.path.join(tmp, name + "_kb")
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu",
        "--knowledge-root", root, "--out-dir", out_dir,
        "--results-file", os.path.join(tmp, name + "_results.jsonl"),
        "--max-iters", "3", "--target-speedup", "0.5", "--timeout", "300",
        "--profiler", "off",
    ] + list(extra)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
    metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    report = {}
    path = os.path.join(run_dir, "distill_report.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    return run_dir, root, meta, report


def check_p2_claimed_only(tmp):
    """P2.1: declaring online_softmax on a direct torch.softmax gains no evidence."""
    from rsi.store import StrategyStore

    task_path = write(os.path.join(tmp, "p2_tasks", "slow.py"), P2_TASK)
    full = _p2_sources()[0]
    replay = {"responses": [
        {"action": "write_solution", "content": full,
         "strategies": ["online softmax"], "expected_speedup": 3.0},
        {"action": "finish", "summary": "claimed-only self-test"},
    ]}
    _run_dir, root, _meta, report = _run_p2_loop(
        tmp, "p2_claimed", task_path, replay, ["--no-ablate"])
    store = StrategyStore(root)
    sid = store.find_by_key("online_softmax")["id"]
    assert not report.get("evidence"), report
    claimed = [row for row in (report.get("claimed_only") or []) if sid in row.get("strategies", [])]
    assert claimed, report.get("claimed_only")
    assert not store.find_by_key("online_softmax")["evidence"], "claimed-only practice got evidence"


def check_p2_ablation_end_to_end(tmp):
    """P2.2 + P2.4 + P6: ablation evidence gates success, combos and validity."""
    from rsi.store import StrategyStore

    task_path = write(os.path.join(tmp, "p2_tasks", "slow_abl.py"), P2_TASK)
    root = os.path.join(tmp, "p2_abl_kb")
    store = _seed_p2_store(root)
    full, no_fused, no_contig, invalid = _p2_sources()
    replay = {"responses": [
        {"action": "write_solution", "content": full,
         "strategies": ["fused softmax", "contiguous input"], "expected_speedup": 3.0},
        # ablation of fused_softmax (removes torch.softmax, keeps .contiguous())
        {"action": "write_solution", "content": no_fused, "strategies": [],
         "expected_speedup": 1.0},
        # ablation of contiguous_input (keeps torch.softmax, removes .contiguous())
        {"action": "write_solution", "content": no_contig, "strategies": [],
         "expected_speedup": 1.0},
        {"action": "finish", "summary": "ablation self-test"},
    ]}
    _run_dir, root, meta, report = _run_p2_loop(
        tmp, "p2_abl", task_path, replay, ["--ablate-budget", "2"])
    ablations = meta["rsi"]["ablations"]
    assert len(ablations) == 2, ablations
    assert all(row.get("speedup_ci") for row in ablations), ablations
    assert all(row.get("valid") is True for row in ablations), ablations
    sids = [store.find_by_key("fused_softmax")["id"], store.find_by_key("contiguous_input")["id"]]
    rows = {row["strategy"]: row for row in report.get("evidence") or []}
    for sid in sids:
        row = rows.get(sid)
        assert row is not None, report
        assert row["outcome"] == "success", row
        assert row["ablation"]["performed"] is True, row
        assert row["ablation"]["without_speedup_ci"], row
        assert row["ablation"]["contributed"] is True, row
    combos = [json.loads(line) for line in open(os.path.join(root, "combos.jsonl"), encoding="utf-8")
              if line.strip()]
    assert len(combos) == 1, combos
    assert set(combos[0]["members"]) == set(sids), combos
    assert all(combos[0]["contributions"].get(sid) for sid in sids), combos

    # An "ablation" that does not actually remove the practice is invalid: its
    # success is demoted to neutral and no combo is formed from it.
    root2 = os.path.join(tmp, "p2_abl_invalid_kb")
    store2 = _seed_p2_store(root2)
    replay2 = {"responses": [
        {"action": "write_solution", "content": full,
         "strategies": ["fused softmax", "contiguous input"], "expected_speedup": 3.0},
        {"action": "write_solution", "content": invalid, "strategies": [],
         "expected_speedup": 1.0},
        {"action": "write_solution", "content": no_contig, "strategies": [],
         "expected_speedup": 1.0},
        {"action": "finish", "summary": "invalid ablation self-test"},
    ]}
    _run_dir2, root2, meta2, report2 = _run_p2_loop(
        tmp, "p2_abl_invalid", task_path, replay2, ["--ablate-budget", "2"])
    fused = store2.find_by_key("fused_softmax")["id"]
    rows2 = {row["strategy"]: row for row in report2.get("evidence") or []}
    assert rows2[fused]["outcome"] == "neutral", rows2[fused]
    assert rows2[fused]["ablation"]["valid"] is False, rows2[fused]
    assert "still present" in rows2[fused]["ablation"]["problem"], rows2[fused]
    combos2 = [line for line in open(os.path.join(root2, "combos.jsonl"), encoding="utf-8")
               if line.strip()]
    assert not combos2, combos2
def check_p2_stage_failure(tmp):
    """P2.3: a syntax error is failure memory, not a strategy failure."""
    from rsi.store import StrategyStore

    task_path = write(os.path.join(tmp, "p2_tasks", "slow_stage.py"), P2_TASK)
    good = _p2_sources()[1]
    bad = """import torch


def run(x):
    return (
"""
    replay = {"responses": [
        {"action": "write_solution", "content": bad,
         "strategies": ["online softmax"], "expected_speedup": 2.0},
        {"action": "write_solution", "content": good,
         "strategies": ["online softmax"], "expected_speedup": 1.0},
        {"action": "finish", "summary": "stage failure self-test"},
    ]}
    _run_dir, root, _meta, report = _run_p2_loop(
        tmp, "p2_stage", task_path, replay, ["--no-ablate"])
    store = StrategyStore(root)
    st = store.find_by_key("online_softmax")
    assert st["stats"]["failed"] == 0, st["stats"]
    assert any("stage failure (import_error)" in row.get("reason", "")
               for row in report.get("skipped") or []), report.get("skipped")


def check_p4_synthetic_cap(tmp):
    """P4.3: many cheap synthetic tasks cannot fill or inflate the plan."""
    from rsi.evolve import build_plan

    def make(name, synthetic):
        return {"name": name, "path": os.path.join(tmp, "tasks", name + ".py"),
                "kind": "python", "level": 1, "expected_speedup": 1.1,
                "features": {"name": name, "tags": ["softmax", "reduction"],
                             "op_class": "reduction", "dtypes": ["float32"]},
                "op_class": "reduction", "dtypes": ["float32"], "tags": ["softmax"],
                "synthetic": synthetic}

    tasks = [make("real_%d" % i, False) for i in range(3)]
    tasks += [make("synth_%d" % i, True) for i in range(12)]
    plan = build_plan({"runs": [], "tasks": tasks}, top=3)
    assert not [row for row in plan if row.get("synthetic")], plan
    only_synth = build_plan({"runs": [], "tasks": [make("synth_only", True)]}, top=3)
    assert len(only_synth) == 1 and only_synth[0]["synthetic"], only_synth
    real_row = build_plan({"runs": [], "tasks": [make("real_one", False)]}, top=1)[0]
    synth_row = build_plan({"runs": [], "tasks": [make("synth_one", True)]}, top=1)
    assert synth_row and synth_row[0]["score"] < real_row["score"], (real_row, synth_row)


def check_p4_version_rollback(tmp):
    """P4.5: knowledge snapshots + a real holdout re-evaluation drive rollback."""
    from rsi import version

    root = os.path.join(tmp, "p4_v_kb")
    os.makedirs(root)
    write(os.path.join(root, "strategies.jsonl"),
          json.dumps({"id": "stg-1", "statement": "before"}) + "\n")
    before = version.snapshot(root, os.path.join(tmp, "p4_v_snaps"))
    write(os.path.join(root, "strategies.jsonl"),
          json.dumps({"id": "stg-2", "statement": "after"}) + "\n")
    after = version.snapshot(root, os.path.join(tmp, "p4_v_snaps"))
    assert before["version"] != after["version"]

    task_paths = [os.path.join(ROOT, "tasks", "level1", "softmax.py")]

    def stub(speedup):
        def runner(task_path, knowledge_root, seed, run_dir, max_iters=4, timeout=900.0):
            os.makedirs(run_dir, exist_ok=True)
            return {"status": "pass", "speedup": speedup, "task": "softmax", "seed": seed}
        return runner

    score_before = version.evaluate_holdout(root, task_paths,
                                            os.path.join(tmp, "p4_v_eval_before"),
                                            runner=stub(1.2))
    score_after = version.evaluate_holdout(root, task_paths,
                                           os.path.join(tmp, "p4_v_eval_after"),
                                           runner=stub(1.0))
    assert abs(score_before["score"] - 0.1823) < 1e-3, score_before
    assert abs(score_after["score"] - 0.0) < 1e-6, score_after
    assert score_before["metric"].startswith("mean log(speedup)"), score_before
    result = version.rollback_if_worse(root, before, score_before, score_after, 0.01)
    assert result["rolled_back"] is True, result
    text = open(os.path.join(root, "strategies.jsonl"), encoding="utf-8").read()
    assert "stg-1" in text and "stg-2" not in text, text


def check_p4_holdout_eval_is_real(tmp):
    """P4.5: holdout_score is a fresh evaluation, not an average of old runs."""
    from rsi import version

    calls = []

    def runner(task_path, knowledge_root, seed, run_dir, max_iters=4, timeout=900.0):
        calls.append((task_path, knowledge_root, seed))
        os.makedirs(run_dir, exist_ok=True)
        return {"status": "pass", "speedup": 1.5, "task": "t", "seed": seed}

    tasks = [os.path.join(ROOT, "tasks", "level1", "softmax.py"),
             os.path.join(ROOT, "tasks", "level1", "matmul.py")]
    score = version.evaluate_holdout(os.path.join(tmp, "p4_real_kb"), tasks,
                                     os.path.join(tmp, "p4_real_eval"), runner=runner, seeds=2)
    assert len(calls) == 4, calls
    assert score["n_tasks"] == 2 and score["score"] is not None, score
    # per task log ratio: log(1.5) for each, mean stays log(1.5)
    assert abs(score["score"] - 0.4055) < 1e-3, score
    assert all(abs(row["log_ratio"] - 0.4055) < 1e-3 for row in score["tasks"].values()), score
def check_p4_auto_skill(tmp):
    """P4.4: a strategy that becomes verified auto-compiles a twice-validated skill."""
    from rsi.distill import distill_run
    from rsi.memory import MemoryStore
    from rsi.store import StrategyStore

    task_path = os.path.join(tmp, "p4_auto_task", "p4_auto.py")
    write(task_path, textwrap.dedent('''
        import torch
        TASK = {"name": "p4_auto", "level": 1,
                "tolerance": {"atol": 1e-5, "rtol": 1e-5},
                "num_trials": 2, "warmup": 5, "rep": 30,
                "input_specs": [{"kind": "randn", "shape": [2048, 4096], "dtype": "float32"}],
                "entrypoint": "run", "baseline": "eager", "tags": ["softmax", "reduction"]}
        def reference(x):
            m = x.max(dim=-1, keepdim=True).values
            e = torch.exp(x - m)
            return e / e.sum(dim=-1, keepdim=True)
    '''))
    root = os.path.join(tmp, "p4_auto_kb")
    store = StrategyStore(root)
    sid = store.find_by_key("block_tiling")["id"]
    store.record_evidence(sid, "prior", "p4_auto", 1.4, "success",
                          run_id="prior", hardware="cpu", dtype="float32",
                          holdout_ok=True)
    assert store.strategies[sid]["confidence"] == "candidate"

    run_dir = os.path.join(tmp, "p4_auto_run")
    solution = write(os.path.join(run_dir, "solution.py"),
                     "import torch\n\nTILE = 1\n\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n")
    source = open(solution, encoding="utf-8").read()
    features = {"name": "p4_auto", "tags": ["softmax", "reduction"],
                "op_class": "reduction", "dtypes": ["float32"]}
    meta = {"task": "p4_auto", "task_file": task_path, "device": "cpu",
            "hardware": "cpu", "features": features,
            "best": {"status": "pass", "speedup": 4.0, "path": solution}}
    write(os.path.join(run_dir, "meta.json"), json.dumps(meta))
    traj = {"task": "p4_auto", "task_file": task_path, "hardware": "cpu",
            "device": "cpu", "features": features, "best": meta["best"],
            "distilled": False,
            "attempts": [
                {"iter": 1, "status": "pass", "speedup": 4.0,
                 "speedup_ci": [3.9, 4.1], "solution": solution,
                 "solution_source": source, "epsilon": 0.043,
                 "holdout_ok": True,
                 "strategies_declared": ["block tiling"],
                 "strategies_used": [sid], "distilled": False},
                {"iter": "ablation:%s" % sid, "kind": "ablation", "ablation_of": sid,
                 "status": "pass", "speedup": 0.5, "speedup_ci": [0.45, 0.55],
                 "solution_source": source, "distilled": False},
            ]}
    write(os.path.join(run_dir, "trajectory.json"), json.dumps(traj))
    report = distill_run(run_dir, store=store)
    assert store.strategies[sid]["confidence"] == "verified", store.strategies[sid]
    skills = report.get("skills") or []
    assert skills and skills[0]["status"] == "compiled", report
    assert skills[0]["confidence"] == "verified" and skills[0]["passes"] == 2, skills
    memory = MemoryStore(root)
    auto = [row for row in memory.skills if row.get("auto_generated")]
    assert auto and auto[0]["confidence"] == "verified", memory.skills
    assert (auto[0].get("validation") or {}).get("passes") == 2, auto[0]


def check_p4_synth_and_schedule(tmp):
    """P4.1 + P4.2: a real failure becomes a task and the scheduler runs it."""
    root = ROOT
    fail_kb = os.path.join(tmp, "p4_fail_kb")
    fail_runs = os.path.join(tmp, "p4_fail_runs")
    wrong = "def run(x, y):\n    return x - y\n"
    replay_fail = write(os.path.join(tmp, "p4_fail_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": wrong, "strategies": [],
             "expected_speedup": 2.0},
            {"action": "finish", "summary": "failure generation"},
        ]}))
    proc = subprocess.run([
        sys.executable, "-m", "agent.loop", "--task", os.path.join(root, "tasks", "level1", "vector_add.py"),
        "--provider", "replay", "--replay-file", replay_fail, "--device", "cpu",
        "--knowledge-root", fail_kb, "--out-dir", fail_runs,
        "--max-iters", "1", "--profiler", "off", "--no-distill",
        "--timeout", "300", "--quiet",
    ], cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)
    failures = [json.loads(line) for line in
                open(os.path.join(fail_kb, "failures.jsonl"), encoding="utf-8") if line.strip()]
    assert failures and "x - y" in (failures[0]["minimal_repro"].get("code") or ""), failures

    synth_out = os.path.join(tmp, "p4_tasks", "synth")
    synth = subprocess.run([
        sys.executable, "-m", "rsi.synth", "--failures", os.path.join(fail_kb, "failures.jsonl"),
        "--tasks-root", os.path.join(root, "tasks"), "--out", synth_out,
        "--approve", "--device", "cpu", "--timeout", "600",
    ], cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=1200)
    assert synth.returncode == 0, (synth.stdout[-1500:] + "\n" + synth.stderr[-1500:])
    task_files = glob.glob(os.path.join(synth_out, "*.py"))
    assert task_files, synth_out
    sidecar = json.load(open(os.path.splitext(task_files[0])[0] + ".json", encoding="utf-8"))
    assert sidecar["approved"] is True and sidecar["validation"]["ok"] is True, sidecar

    runs_root = os.path.join(tmp, "p4_sched_runs")
    write(os.path.join(runs_root, "20260101-softmax", "meta.json"), json.dumps({
        "task": "softmax", "device": "cpu", "hardware": "cpu", "features": {},
        "iterations": 1, "best": {"status": "pass", "speedup": 1.2},
    }))
    good = "def run(x, y):\n    return x + y\n"
    replay_good = write(os.path.join(tmp, "p4_sched_replay.json"), json.dumps({
        "responses": [
            {"action": "write_solution", "content": good, "strategies": [],
             "expected_speedup": 1.0},
            {"action": "finish", "summary": "scheduled"},
        ]}))
    sched_out = os.path.join(tmp, "p4_schedule")
    sched = subprocess.run([
        sys.executable, "-m", "rsi.schedule", "--budget-iters", "3",
        "--approve", "auto", "--approve-synthetic", "--top", "1", "--seeds", "1",
        "--variants", "base", "--max-iters", "2", "--target-speedup", "0.5",
        "--provider", "replay", "--replay-file", replay_good, "--device", "cpu",
        "--knowledge-root", os.path.join(tmp, "p4_sched_kb"),
        "--runs-root", runs_root, "--tasks-root", os.path.join(tmp, "p4_tasks"),
        "--holdout-config", os.path.join(root, "configs", "holdout.json"),
        "--versions-root", os.path.join(tmp, "p4_versions"),
        "--out", sched_out, "--timeout", "300", "--skip-holdout-eval",
    ], cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=1200)
    assert sched.returncode == 0, (sched.stdout[-2000:] + "\n" + sched.stderr[-2000:])
    audits = glob.glob(os.path.join(sched_out, "*", "audit.json"))
    assert audits, sched_out
    audit = json.load(open(audits[0], encoding="utf-8"))
    assert audit["runs"], audit
    assert all(row["task"].startswith("synth_") for row in audit["runs"]), audit["runs"]
    assert audit["used_iters"] <= 3, audit
    assert audit["holdout_score"]["before"]["score"] is None, audit["holdout_score"]
    assert "holdout evaluation disabled" in (audit["holdout_score"]["before"].get("reason") or ""), audit["holdout_score"]
    assert audit["knowledge"]["before"]["id"] and audit["knowledge"]["after"]["id"], audit["knowledge"]
    assert audit["knowledge"]["final_version"], audit["knowledge"]


def check_adversarial_solutions(tmp):
    """Every cheat must be rejected; the honest control must still pass."""
    import statistics as stats_mod

    from harness import device as dev
    from harness.runner import run_evaluation

    task = os.path.join(ROOT, "tasks", "level1", "softmax.py")
    root = os.path.join(ROOT, "tests", "adversarial")
    common = dict(device="cpu", timeout=600, out_dir=os.path.join(tmp, "adv_out"),
                  results_file=os.path.join(tmp, "adv_results.jsonl"), quiet=True)

    def run(name, **extra):
        opts = dict(common)
        opts.update(extra)
        return run_evaluation(task=task, solution=os.path.join(root, name), **opts)

    cache = run("cache_hack.py", rep=50)
    assert cache["status"] == "fail_correctness", (cache["status"], cache.get("error"))
    assert "stale" in (cache.get("error") or "").lower(), cache.get("error")

    patch = run("patch_hack.py", rep=50)
    assert patch["status"] == "fail_suspicious", (patch["status"], patch.get("error"))
    assert "torch.softmax" in (patch.get("error") or ""), patch.get("error")

    view = run("view_hack.py", rep=50)
    assert view["status"] == "fail_correctness", (view["status"], view.get("error"))

    ref = run("ref_read_hack.py", rep=50)
    assert ref["status"] == "fail_suspicious", (ref["status"], ref.get("error"))

    timer = run("timer_patch_hack.py", rep=50)
    assert timer["status"] == "fail_suspicious", (timer["status"], timer.get("error"))
    assert "harness" in (timer.get("error") or ""), timer.get("error")

    accelerator = dev.is_accelerator(dev.resolve_device("auto"))
    if accelerator:
        side = run("side_stream_hack.py", rep=50, device="auto")
        speed = float((side.get("timing") or {}).get("speedup") or 0.0)
        assert side["status"] not in ("pass", "pass_unstable") or speed < 1.2, (side["status"], speed)
        print("  side_stream_hack: status=%s speedup=%.3fx" % (side["status"], speed))
    else:
        print("  SKIP side_stream_hack: needs a CUDA/NPU device (CPU has no streams)")

    speeds = []
    for _ in range(3):
        res = run("honest.py", warmup=25, rep=200)
        assert res["status"] in ("pass", "pass_unstable"), (res["status"], res.get("error"))
        speeds.append(float(res["timing"]["speedup"]))
    median = stats_mod.median(speeds)
    assert 0.9 <= median <= 1.1, (
        "honest median speedup %.3f outside [0.9, 1.1] (runs %s)" % (median, speeds))
    print("  honest speedups: %s (median %.3f)" % (
        ", ".join("%.3f" % s for s in speeds), median))


def check_rsi_attribution_precision(tmp):
    from rsi.distill import attribute_from_source, resolve_declared
    from rsi.store import StrategyStore

    store = StrategyStore(os.path.join(tmp, "attr_knowledge"))
    # natural-language declarations resolve to the seed practices
    assert resolve_declared("block tiling", store) == store.find_by_key("block_tiling")["id"]
    assert resolve_declared("multi-core split", store) == store.find_by_key("multi_core_split")["id"]
    assert resolve_declared("MLA absorption", store) == store.find_by_key("mla_absorption")["id"]
    assert resolve_declared("completely unknown practice xyz", store) is None
    # threshold is 30: a 0.6 token overlap (24) is below the bar now
    assert resolve_declared("fold scaling producer zzz yyy", store) is None

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

    # exemplars must state their real provenance, not claim hardware verification
    mech = record_exemplar(root, "mech_task", "def run():\n    return 1\n", 1.2,
                           hardware="ascend-910b", device_class="cpu", evidence_valid=False)
    mech_block = render_exemplars([mech])
    assert "mechanics-only" in mech_block and "measured on cpu" in mech_block, mech_block

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
    assert allowed["status"] in ("pass", "pass_unstable"), (allowed.get("status"), allowed.get("error"))
    assert allowed["solution_features"]["uses_triton"] is True, allowed.get("solution_features")

    wrong_op = write(os.path.join(tmp, "constraints", "opbase_wrong.py"),
                     OPBASE_TEMPLATE.format(body="return x * 3.0"))
    ok_op = write(os.path.join(tmp, "constraints", "opbase_ok.py"),
                  OPBASE_TEMPLATE.format(body="return x + 1.0"))
    good_solution = write(os.path.join(tmp, "constraints", "good.py"), "def run(x):\n    return x + 1.0\n")

    res_wrong = run_evaluation(task=wrong_op, solution=good_solution, **common)
    assert res_wrong["status"] in ("pass", "pass_unstable"), res_wrong.get("error")
    check = res_wrong.get("baseline_check") or {}
    assert check.get("used") == "eager" and "rejected" in (check.get("fallback") or ""), check

    res_ok = run_evaluation(task=ok_op, solution=good_solution, **common)
    assert res_ok["status"] in ("pass", "pass_unstable"), res_ok.get("error")
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
    assert meta["best"]["status"] in ("pass", "pass_unstable"), meta["best"]
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
        if res.get("status") not in ("pass", "pass_unstable"):
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
    check("harness: ABAB bootstrap + success classification", lambda: check_statistical_gate(tmp))
    check("harness: calibrated noise config (cpu + ascend-910b)", lambda: check_noise_config(tmp))
    check("report: pass@1 table", lambda: check_report(tmp))
    check("agent loop: replay (wrong -> good -> finish)", lambda: check_agent_replay(tmp, files))
    check("rsi: strategy pool bootstrap + retrieval priors", lambda: check_rsi_store_and_retrieve(tmp))
    check("rsi: applicability conditions narrow with evidence", lambda: check_rsi_condition_refinement(tmp))
    check("rsi: verification requires two independent runs", lambda: check_verified_two_runs(tmp))
    check("p3: verification requires holdout success", lambda: check_p3_verified_holdout(tmp))
    check("p3: shape features narrow applicability", lambda: check_p3_shape_refinement(tmp))
    check("p3: calibration error down-weights retrieval", lambda: check_p3_retrieval_calibration(tmp))
    check("p3: exemplar behavior archive + descendant gain", lambda: check_p3_exemplar_archive(tmp))
    check("p3: evolve plan uses configured holdout", lambda: check_p3_evolve_holdout(tmp))
    check("p4: synthetic tasks are capped in the plan", lambda: check_p4_synthetic_cap(tmp))
    check("p4: knowledge snapshot/rollback + holdout score", lambda: check_p4_version_rollback(tmp))
    check("p4: holdout score is a real re-evaluation", lambda: check_p4_holdout_eval_is_real(tmp))
    check("rsi: distill gate classifies neutral vs success", lambda: check_distill_gate(tmp))
    check("rsi: instability widens CI instead of discarding gains", lambda: check_unstable_kept(tmp))
    check("rsi: holdout=None caps confidence at candidate", lambda: check_unstable_candidates_holdout_none(tmp))
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
        check("rsi: loop records trajectory, distills, reuses priors", lambda: check_rsi_loop_distill(tmp))
        check("rsi: strict provenance blocks CPU evidence for ascend-910b", lambda: check_strict_provenance(tmp))
        check("p3: holdout measurement rejects train-only gains", lambda: check_p3_holdout_measurement(tmp))
        check("p4: verified strategy auto-compiles a skill", lambda: check_p4_auto_skill(tmp))
        check("p4: failure -> synthetic task -> scheduled cycle", lambda: check_p4_synth_and_schedule(tmp))
        check("p2: claimed-only declarations gain no evidence", lambda: check_p2_claimed_only(tmp))
        check("p2: ablation evidence gates success and combos", lambda: check_p2_ablation_end_to_end(tmp))
        check("p2: syntax errors do not count as strategy failures", lambda: check_p2_stage_failure(tmp))
        check("adversarial: cheat regression set (cache/patch/side-stream/view/reference)",
          lambda: check_adversarial_solutions(tmp))
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
