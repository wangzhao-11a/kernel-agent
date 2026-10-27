"""Offline self-test for the CannBench AscendC backend and agent loop.

Runs on Windows/CPU with a mock package: no AscendC compiler, no NPU, no API key.

    python scripts/cannbench_selftest.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cannbench_loop import main as loop_main  # noqa: E402
from agent.review import review_edit  # noqa: E402
from harness.cannbench import (  # noqa: E402
    CannBenchError, format_summary, load_spec, prepare_workspace, run_cannbench, run_profile,
)

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
        RESULTS["pass"].append(name)
        print("PASS  %s" % name)
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))


BUILD_PY = r'''
import os, sys
src = open("kernel.txt", encoding="utf-8").read()
if "BUILD_FAIL" in src:
    print("mock compiler: simulated error")
    sys.exit(1)
os.makedirs("dist", exist_ok=True)
open("dist/ok", "w", encoding="utf-8").write("ok")
print("mock build ok")
'''

TEST_PY = r'''
src = open("kernel.txt", encoding="utf-8").read()
cases = [
    ("(2, 4)", "float16", -1),
    ("(1022, 2047)", "bfloat16", 0),
    ("(11, 13, 16, 67)", "float32", 2),
]
bad = "BAD" in src
for i, (shape, dtype, dim) in enumerate(cases):
    status = "FAIL" if bad and i == 1 else "PASS"
    print("%-22s %-9s dim=%-3d MERE=%.3e MARE=%.3e %s" % (shape, dtype, dim, 0.0, 0.0, status))
print("SWIGLU_TEST", "FAIL" if bad else "PASS")
'''

BENCH_PY = r'''
print('CANNBENCH_METRICS {"speedup": 1.25, "metric_key": "speedup"}')
'''

PROFILE_PY = r'''
import json
print("CANNBENCH_PROFILE " + json.dumps({
    "shape": [2, 4], "dtype": "float16", "dim": -1, "kernel": "mock_swiglu",
    "count": 1, "duration_us": 1.0,
    "aiv_vec_ratio": 0.5, "aiv_scalar_ratio": 0.1,
    "aiv_mte2_ratio": 0.7, "aiv_mte3_ratio": 0.2, "bottleneck": "MTE2",
}))
print("SWIGLU_PROFILE PASS")
'''


def make_package(tmp):
    pkg = os.path.join(tmp, "pkg")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "kernel.txt"), "w", encoding="utf-8") as fh:
        fh.write("GOOD\n")
    with open(os.path.join(pkg, "extra.txt"), "w", encoding="utf-8") as fh:
        fh.write("EXTRA_BASE\n")
    with open(os.path.join(pkg, "large.txt"), "w", encoding="utf-8") as fh:
        for i in range(400):
            fh.write("line %03d\n" % i)
    for name, body in (("mock_build.py", BUILD_PY), ("mock_test.py", TEST_PY),
                       ("mock_bench.py", BENCH_PY), ("mock_profile.py", PROFILE_PY)):
        with open(os.path.join(pkg, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    spec = {
        "id": "mock_swiglu",
        "description": "mock package for the CannBench backend",
        "package": "pkg",
        "edit_files": ["kernel.txt", "extra.txt", "large.txt"],
        "build_command": [sys.executable, "mock_build.py"],
        "test_command": [sys.executable, "mock_test.py"],
        "bench_command": [sys.executable, "mock_bench.py"],
        "profile_command": [sys.executable, "mock_profile.py"],
        "timeout_s": 30,
        "soc": "ascend950",
        "hardware": "ascend-950pr",
        "baseline": "vendor",
        "expected_speedup": 1.0,
        "metadata": {"op": "swi_glu", "op_class": "elementwise", "dtypes": ["float16"]},
    }
    spec_path = os.path.join(tmp, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return spec_path


def main():
    tmp = tempfile.mkdtemp(prefix="cannbench-selftest-")
    try:
        spec_path = make_package(tmp)
        spec = load_spec(spec_path)

        check("spec loads", lambda: (_ for _ in ()).throw(AssertionError(spec.id))
              if spec.id != "mock_swiglu" else None)

        def review_rules():
            bad = review_edit({"a.cpp": "Gather(x0, x0, off, 0, n);"},
                              {"apis": ["Gather"]})
            assert not bad["ok"], bad
            assert any(f["code"] == "inplace_gather" for f in bad["findings"]), bad
            mask = review_edit({"a.cpp": "GatherMask(a, b, 1, false, 8, p, cnt);"},
                               {"apis": ["Gather"]})
            assert any(f["code"] == "gathermask" for f in mask["findings"]), mask
            header = review_edit({"a.h": "// header"}, {"apis": ["Gather"], "layout": "tiny-row"})
            assert any(f["code"] == "header_only" for f in header["findings"]), header
            mismatch = review_edit({"a.cpp": "DataCopyPad(a, b, p);"}, {"apis": ["Gather"]})
            assert any(f["code"] == "plan_api_missing" for f in mismatch["findings"]), mismatch
        check("static review: in-place Gather / GatherMask / header-only / API mismatch", review_rules)

        def good_case():
            result = run_cannbench(spec, files={"kernel.txt": "GOOD\n"},
                                   work_root=os.path.join(tmp, "run_good"))
            assert result["status"] == "pass", result
            assert result["case_summary"]["passed"] == 3, result["case_summary"]
            assert (result["timing"] or {}).get("speedup") == 1.25, result["timing"]
            assert "pass" in format_summary(result)
        check("build+test+bench pass with 3/3 cases", good_case)

        def profile_case():
            result = run_profile(spec, files={"kernel.txt": "GOOD_PROFILE\n"},
                                 work_root=os.path.join(tmp, "run_profile"))
            assert result["status"] == "pass", result
            assert result["metrics"]["bottleneck"] == "MTE2", result["metrics"]
            assert result["metrics"]["aiv_mte2_ratio"] == 0.7, result["metrics"]
        check("profile: build + CANNBENCH_PROFILE metrics", profile_case)

        def bad_case():
            result = run_cannbench(spec, files={"kernel.txt": "BAD\n"},
                                   work_root=os.path.join(tmp, "run_bad"))
            assert result["status"] == "fail_correctness", result["status"]
            assert result["case_summary"]["failed"] == 1, result["case_summary"]
        check("case failure -> fail_correctness", bad_case)

        def build_fail():
            result = run_cannbench(spec, files={"kernel.txt": "BUILD_FAIL\n"},
                                   work_root=os.path.join(tmp, "run_buildfail"))
            assert result["status"] == "build_error", result["status"]
        check("build failure -> build_error", build_fail)

        def reject_path():
            try:
                prepare_workspace(spec, os.path.join(tmp, "run_reject"),
                                  {"not_allowed.txt": "x"})
            except CannBenchError:
                return
            raise AssertionError("prepare_workspace accepted a disallowed file")
        check("workspace rejects files outside edit_files", reject_path)

        replay_path = os.path.join(tmp, "replay.json")
        with open(replay_path, "w", encoding="utf-8") as fh:
            json.dump({"responses": [
                {"action": "plan", "layout": "mock flat",
                 "tiling": {"rows_per_tile": 1, "cols_per_tile": 1, "ub_budget": "mock", "cores": 1},
                 "queues": ["VECIN x1"], "apis": ["DataCopyPad"], "sync": ["V_MTE2"],
                 "risks": ["mock risk"], "expected_bottleneck": "V"},
                {"action": "write_files", "files": {"kernel.txt": "GOOD_IMPL\n"},
                 "strategies": ["block_tiling"], "expected_speedup": 1.2},
                {"action": "run_eval"},
                {"action": "finish", "summary": "mock done"},
            ]}, fh)
        out_dir = os.path.join(tmp, "runs")

        def agent_loop():
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", replay_path,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "4",
                "--out-dir", out_dir, "--no-rsi",
            ])
            assert rc == 0, "agent loop returned %s" % rc
            runs = [os.path.join(out_dir, d) for d in os.listdir(out_dir)]
            assert runs, "no run directory"
            run_dir = runs[0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            assert meta["best"]["status"] == "pass", meta["best"]
            assert meta["plan"]["layout"] == "mock flat", meta.get("plan")
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert len(attempts) == 2, len(attempts)
            assert os.path.isfile(os.path.join(run_dir, "final_files", "kernel.txt"))
        check("agent loop: plan -> write_files -> run_eval -> finish", agent_loop)

        def plan_gate():
            pg_replay = os.path.join(tmp, "replay_plan_gate.json")
            with open(pg_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "write_files", "files": {"kernel.txt": "GOOD\n"}},
                    {"action": "plan", "layout": "gated layout",
                     "tiling": {"rows_per_tile": 1, "cores": 1},
                     "apis": ["DataCopyPad"], "risks": ["mock"]},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD_PG\n"}},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "plan gate done"},
                ]}, fh)
            out_pg = os.path.join(tmp, "runs_plan_gate")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", pg_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "5",
                "--out-dir", out_pg, "--no-rsi",
            ])
            assert rc == 0, "plan-gate loop returned %s" % rc
            run_dir = [os.path.join(out_pg, d) for d in os.listdir(out_pg)][0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            assert meta["plan"]["layout"] == "gated layout", meta.get("plan")
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert len(attempts) == 2, len(attempts)
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            assert any(r.get("role") == "user" and "A plan is required" in (r.get("content") or "")
                       for r in transcript), transcript[:3]
        check("plan gate: write_files rejected before plan", plan_gate)

        def noop_gate():
            noop_replay = os.path.join(tmp, "replay_noop.json")
            with open(noop_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "plan", "layout": "noop layout",
                     "tiling": {"rows_per_tile": 1, "cores": 1},
                     "apis": ["DataCopyPad"], "risks": ["mock"]},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD\n"}},
                    {"action": "finish", "summary": "noop done"},
                ]}, fh)
            out_noop = os.path.join(tmp, "runs_noop")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", noop_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "3",
                "--out-dir", out_noop, "--no-rsi",
            ])
            assert rc == 1, "no-op run should not report pass, got %s" % rc
            run_dir = [os.path.join(out_noop, d) for d in os.listdir(out_noop)][0]
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert not attempts, attempts
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            assert any("no content changes" in (r.get("content") or "") for r in transcript), transcript
        check("no-op write guard: identical content is not evaluated", noop_gate)

        def read_patch():
            rp_replay = os.path.join(tmp, "replay_read_patch.json")
            with open(rp_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "read_file", "path": "kernel.txt", "start": 1, "end": 5},
                    {"action": "read_file", "path": "extra.txt", "start": 1, "end": 5},
                    {"action": "patch_files", "patches": [
                        {"path": "kernel.txt", "find": "GOOD", "replace": "GOOD_PATCH"}]},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "read patch done"},
                ]}, fh)
            out_rp = os.path.join(tmp, "runs_read_patch")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", rp_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "3",
                "--out-dir", out_rp, "--no-rsi", "--no-require-plan",
            ])
            assert rc == 0, "read/patch loop returned %s" % rc
            run_dir = [os.path.join(out_rp, d) for d in os.listdir(out_rp)][0]
            final = open(os.path.join(run_dir, "final_files", "kernel.txt"), encoding="utf-8").read()
            assert final == "GOOD_PATCH\n", final
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            assert sum(1 for r in transcript if r.get("role") == "read_file") == 2, transcript
            assert any("GOOD_PATCH" in (r.get("content") or "") for r in transcript), transcript
        check("read_file + patch_files: range read + exact patch", read_patch)

        def review_gate():
            rg_replay = os.path.join(tmp, "replay_review.json")
            with open(rg_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "plan", "layout": "review layout",
                     "tiling": {"rows_per_tile": 1, "cores": 1},
                     "apis": ["Gather"], "risks": ["mock"]},
                    {"action": "write_files",
                     "files": {"kernel.txt": "Gather(x0, x0, off, 0, n);\n"}},
                    {"action": "finish", "summary": "review gate done"},
                ]}, fh)
            out_rg = os.path.join(tmp, "runs_review")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", rg_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "4",
                "--out-dir", out_rg, "--no-rsi",
            ])
            assert rc == 1, "review-blocked run should not pass, got %s" % rc
            run_dir = [os.path.join(out_rg, d) for d in os.listdir(out_rg)][0]
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert not attempts, attempts
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            assert any(r.get("role") == "review" for r in transcript), transcript
            assert any("inplace_gather" in (r.get("content") or "") for r in transcript), transcript
        check("static review blocks in-place Gather before build", review_gate)

        def massive_rewrite():
            mr_replay = os.path.join(tmp, "replay_massive.json")
            with open(mr_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "write_files", "files": {"large.txt": "PLACEHOLDER\n"}},
                    {"action": "finish", "summary": "massive rewrite"},
                ]}, fh)
            out_mr = os.path.join(tmp, "runs_massive")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", mr_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "2",
                "--out-dir", out_mr, "--no-rsi", "--no-require-plan",
            ])
            assert rc == 1, "massive rewrite should not pass, got %s" % rc
            run_dir = [os.path.join(out_mr, d) for d in os.listdir(out_mr)][0]
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert not attempts, attempts
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            assert any("massive_rewrite" in (r.get("content") or "") for r in transcript), transcript
        check("massive rewrite guard: 400-line file cannot become a placeholder", massive_rewrite)

        def cumulative_edits():
            cu_replay = os.path.join(tmp, "replay_cumulative.json")
            with open(cu_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "plan", "layout": "cumulative layout",
                     "tiling": {"rows_per_tile": 1, "cores": 1},
                     "apis": ["DataCopyPad"], "risks": ["mock"]},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD_CUM\n"}},
                    {"action": "write_files", "files": {"extra.txt": "EXTRA_NEW\n"}},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "cumulative done"},
                ]}, fh)
            out_cu = os.path.join(tmp, "runs_cumulative")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", cu_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "6",
                "--out-dir", out_cu, "--no-rsi",
            ])
            assert rc == 0, "cumulative loop returned %s" % rc
            run_dir = [os.path.join(out_cu, d) for d in os.listdir(out_cu)][0]
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert len(attempts) == 3, len(attempts)
            workspace = attempts[1]["workspace"]
            assert open(os.path.join(workspace, "kernel.txt"), encoding="utf-8").read() == "GOOD_CUM\n"
            assert open(os.path.join(workspace, "extra.txt"), encoding="utf-8").read() == "EXTRA_NEW\n"
        check("cumulative edits: second write keeps the first file", cumulative_edits)

        def rollback_best():
            rb_replay = os.path.join(tmp, "replay_rollback.json")
            with open(rb_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "plan", "layout": "rollback layout",
                     "tiling": {"rows_per_tile": 1, "cores": 1},
                     "apis": ["DataCopyPad"], "risks": ["mock"]},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD_BEST\n"}},
                    {"action": "write_files", "files": {"kernel.txt": "BAD\n"}},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "rollback done"},
                ]}, fh)
            out_rb = os.path.join(tmp, "runs_rollback")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", rb_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "6",
                "--out-dir", out_rb, "--no-rsi",
            ])
            assert rc == 0, "rollback loop returned %s" % rc
            run_dir = [os.path.join(out_rb, d) for d in os.listdir(out_rb)][0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            assert meta["rollbacks"], meta
            attempts = [json.loads(line) for line in
                        open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8") if line.strip()]
            assert [a["status"] for a in attempts] == ["pass", "fail_correctness", "pass"], attempts
            final = open(os.path.join(run_dir, "final_files", "kernel.txt"), encoding="utf-8").read()
            assert final == "GOOD_BEST\n", final
        check("rollback: failed candidate restores best passing files", rollback_best)

        def canned_kb():
            kb = os.path.join(tmp, "kb_cards")
            card_dir = os.path.join(kb, "operators", "mock")
            os.makedirs(card_dir, exist_ok=True)
            with open(os.path.join(card_dir, "card.md"), "w", encoding="utf-8") as fh:
                fh.write("---\n"
                         "title: Mock AscendC recipe\n"
                         "type: operator\n"
                         "description: mock SwiGlu AscendC tiling recipe\n"
                         "tags: [swiglu, ascendc, tiling]\n"
                         "status: stable\n"
                         "platforms: [a5]\n"
                         "---\n\n"
                         "MOCK_ASCENDC_RECIPE\n")
            kb_replay = os.path.join(tmp, "replay_kb.json")
            with open(kb_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "knowledge", "id": "operators/mock/card.md"},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD_KB\n"}},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "kb done"},
                ]}, fh)
            out_kb = os.path.join(tmp, "runs_kb")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", kb_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "4",
                "--out-dir", out_kb, "--no-rsi", "--no-require-plan", "--kb-root", kb, "--kb-cards", "2",
            ])
            assert rc == 0, "kb loop returned %s" % rc
            run_dir = [os.path.join(out_kb, d) for d in os.listdir(out_kb)][0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            assert "operators/mock/card.md" in meta["kb_candidates"], meta
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            first = next(r for r in transcript if r.get("role") == "user")["content"]
            assert "Mock AscendC recipe" in first, first[:500]
            assert "MOCK_ASCENDC_RECIPE" in first, first[:500]
            assert any(r.get("role") == "knowledge" for r in transcript), transcript
        check("CANN KB: prompt candidates + knowledge action", canned_kb)

        def source_map():
            mock_root = os.path.join(tmp, "mock_cann")
            src_dir = os.path.join(mock_root, "ops_math", "ascendc", "mock_swiglu", "op_kernel")
            os.makedirs(src_dir, exist_ok=True)
            with open(os.path.join(src_dir, "mock.cpp"), "w", encoding="utf-8") as fh:
                fh.write("MOCK_SOURCE\n")
            catalog = os.path.join(tmp, "cann_ops_catalog.json")
            with open(catalog, "w", encoding="utf-8") as fh:
                json.dump({"impl_root": mock_root, "ops": [{
                    "name": "mock_swiglu", "key": "swiglu", "category": "ops_math",
                    "path": os.path.join(mock_root, "ops_math", "ascendc", "mock_swiglu"),
                    "files": 2, "bytes": 100,
                    "has_kernel": True, "has_tiling": True, "has_host": True, "has_api": False,
                    "kernel_files": ["op_kernel/mock.cpp"],
                    "tiling_files": ["op_host/mock_tiling.cpp"],
                    "host_files": [], "api_files": [], "config_files": [],
                    "kb_cards": [], "in_kb": False,
                }]}, fh)
            sm_replay = os.path.join(tmp, "replay_source.json")
            with open(sm_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "source", "path": "ops_math/ascendc/mock_swiglu/op_kernel/mock.cpp"},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD_SM\n"}},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "source done"},
                ]}, fh)
            out_sm = os.path.join(tmp, "runs_sourcemap")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", sm_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "4",
                "--out-dir", out_sm, "--no-rsi", "--no-require-plan", "--ops-catalog", catalog,
            ])
            assert rc == 0, "source-map loop returned %s" % rc
            run_dir = [os.path.join(out_sm, d) for d in os.listdir(out_sm)][0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            assert meta["source_map_ops"] == ["mock_swiglu"], meta
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            first = next(r for r in transcript if r.get("role") == "user")["content"]
            assert "CANN official source map" in first, first[:500]
            assert "op_kernel/mock.cpp" in first, first[:500]
            source_idx = next((i for i, r in enumerate(transcript)
                               if r.get("role") == "source"), None)
            assert source_idx is not None, transcript
            assert "MOCK_SOURCE" in transcript[source_idx + 1]["content"], transcript[source_idx + 1]
        check("CANN source map: prompt + source action", source_map)

        def tool_loop():
            kb = os.path.join(tmp, "kb_tools")
            card_dir = os.path.join(kb, "operators", "tool")
            os.makedirs(card_dir, exist_ok=True)
            with open(os.path.join(card_dir, "card.md"), "w", encoding="utf-8") as fh:
                fh.write("---\n"
                         "title: Mock tool recipe\n"
                         "type: operator\n"
                         "description: tool search target\n"
                         "tags: [ascendc, tool]\n"
                         "status: stable\n"
                         "platforms: [a5]\n"
                         "---\n\nMOCK_TOOL_CARD\n")
            mock_root = os.path.join(tmp, "mock_tools_cann")
            op_dir = os.path.join(mock_root, "ops_math", "ascendc", "mock_swiglu")
            os.makedirs(op_dir, exist_ok=True)
            with open(os.path.join(op_dir, "mock.cpp"), "w", encoding="utf-8") as fh:
                fh.write("// MOCK_TOOL_PATTERN\n")
            catalog = os.path.join(tmp, "tool_catalog.json")
            with open(catalog, "w", encoding="utf-8") as fh:
                json.dump({"impl_root": mock_root, "ops": [{
                    "name": "mock_swiglu", "key": "swiglu", "category": "ops_math",
                    "path": op_dir, "files": 1, "bytes": 20,
                    "has_kernel": True, "has_tiling": False, "has_host": False,
                    "has_api": False, "arch35_only": False,
                    "kernel_files": ["mock.cpp"], "tiling_files": [],
                    "host_files": [], "api_files": [], "config_files": [],
                    "kb_cards": [], "in_kb": False,
                }]}, fh)
            tools_replay = os.path.join(tmp, "replay_tools.json")
            with open(tools_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "kb_search", "query": "Mock tool", "top_k": 3},
                    {"action": "source_grep", "pattern": "MOCK_TOOL_PATTERN"},
                    {"action": "profile", "shape": "2,4", "dtype": "float16", "dim": -1},
                    {"action": "reflect", "lesson": "tools work", "next": "write"},
                    {"action": "write_files", "files": {"kernel.txt": "GOOD_TOOL\n"}},
                    {"action": "run_eval"},
                    {"action": "finish", "summary": "tool done"},
                ]}, fh)
            out_tools = os.path.join(tmp, "runs_tools")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", tools_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "7",
                "--out-dir", out_tools, "--no-rsi", "--no-require-plan",
                "--kb-root", kb, "--ops-catalog", catalog,
            ])
            assert rc == 0, "tool loop returned %s" % rc
            run_dir = [os.path.join(out_tools, d) for d in os.listdir(out_tools)][0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            assert meta["reflections"] and meta["reflections"][0]["lesson"] == "tools work", meta
            assert meta["profiles"] and meta["profiles"][0]["metrics"]["bottleneck"] == "MTE2", meta
            transcript = [json.loads(line) for line in
                          open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") if line.strip()]
            roles = {r.get("role") for r in transcript}
            assert {"kb_search", "source_grep", "profile", "reflect"}.issubset(roles), roles
            assert any("Mock tool recipe" in (r.get("content") or "") for r in transcript), transcript
            assert any("MOCK_TOOL_PATTERN" in (r.get("content") or "") for r in transcript), transcript
        check("agent tools: kb_search / source_grep / profile / reflect", tool_loop)

        def distill_gate():
            kb = os.path.join(tmp, "kb")
            out2 = os.path.join(tmp, "runs_distill")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", replay_path,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "4",
                "--out-dir", out2, "--knowledge-root", kb, "--distill",
            ])
            assert rc == 0, "agent loop returned %s" % rc
            run_dir = [os.path.join(out2, d) for d in os.listdir(out2)][0]
            meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
            distill = (meta.get("rsi") or {}).get("distill") or {}
            assert distill.get("provenance_valid") is False, distill
            assert distill.get("evidence") == 0, distill
        check("distill refuses CPU mechanics evidence", distill_gate)

        def failure_memory():
            kb = os.path.join(tmp, "kb_fail")
            bad_replay = os.path.join(tmp, "replay_bad.json")
            with open(bad_replay, "w", encoding="utf-8") as fh:
                json.dump({"responses": [
                    {"action": "write_files", "files": {"kernel.txt": "BAD\n"}},
                    {"action": "finish", "summary": "bad run"},
                ]}, fh)
            out_bad = os.path.join(tmp, "runs_bad")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", bad_replay,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "3",
                "--out-dir", out_bad, "--knowledge-root", kb, "--no-require-plan",
            ])
            assert rc == 1, "bad run should fail, got %s" % rc
            rows = [json.loads(line) for line in
                    open(os.path.join(kb, "failures.jsonl"), encoding="utf-8") if line.strip()]
            assert any(r.get("status") == "open" and r.get("stage") == "correctness"
                       for r in rows), rows

            out_good = os.path.join(tmp, "runs_good")
            rc = loop_main([
                "--spec", spec_path, "--provider", "replay", "--replay-file", replay_path,
                "--device-label", "cpu", "--target-speedup", "99", "--max-iters", "3",
                "--out-dir", out_good, "--knowledge-root", kb, "--no-require-plan",
            ])
            assert rc == 0, "good run should pass, got %s" % rc
            rows = [json.loads(line) for line in
                    open(os.path.join(kb, "failures.jsonl"), encoding="utf-8") if line.strip()]
            assert any(r.get("status") == "resolved" for r in rows), rows
        check("failure memory: record open -> resolve on pass", failure_memory)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\npassed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())