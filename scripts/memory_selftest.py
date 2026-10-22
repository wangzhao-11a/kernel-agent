"""Offline self-test for the operator memory module.

    python scripts/memory_selftest.py

No NPU, no API key, no network.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi.memory import (  # noqa: E402
    MemoryStore, normalize_features, scope_match, scope_of, shape_signature, version_ok,
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


def main():
    def shape_features():
        sig = shape_signature((4096, 8192), -1, "bf16")
        assert sig["half"] == 4096 and sig["half_aligned_32b"] is True, sig
        sig = shape_signature("(1022, 2047)", 0, "float16")
        assert sig["rank"] == 2 and sig["dim"] == 0, sig
        assert sig["half_aligned_32b"] is False, sig
        sig = shape_signature((1000003, 2), 1, "bfloat16")
        assert sig["half_is_1"] is True, sig
        assert sig["prime_outer"] is True, sig
    check("shape_signature: half/alignment/half_is_1/prime", shape_features)

    def soc_isolation():
        scope = {"soc": ["ascend950"], "form": ["ascendc"]}
        assert scope_match(scope, {"soc": "ascend-950pr", "form": ["ascendc"]})[0]
        assert not scope_match(scope, {"soc": "ascend-910b", "form": ["ascendc"]})[0]
    check("scope: 950 fact does not match 910B", soc_isolation)

    def dtype_alias():
        scope = {"dtype": ["bfloat16"]}
        assert scope_match(scope, {"dtype": ["bf16"]})[0]
        assert scope_match(scope, {"dtype": ["float16"]})[0] is False
    check("scope: bf16 aliases bfloat16", dtype_alias)

    def versions():
        assert version_ok("9.1.0", ">=9.0")
        assert not version_ok("8.5", ">=9.0")
        assert version_ok("0.30.0rc2", ">=0.27.1rc1")
    check("version ranges", versions)

    with tempfile.TemporaryDirectory() as tmp:
        def dedupe():
            store = MemoryStore(os.path.join(tmp, "dedupe"))
            fact = {
                "statement": "test fact", "scope": {"soc": ["ascend950"]},
                "authority": "human", "confidence": "verified",
                "evidence": [{"source": "a"}],
            }
            store.add_fact(fact)
            store.add_fact(fact)
            assert len(store.facts) == 1, store.facts
            assert len(store.facts[0]["evidence"]) == 2, store.facts[0]
            failure = {
                "stage": "build", "signature": "undefined reference to foo",
                "scope": {"soc": ["ascend950"]}, "evidence": [{"run_id": "r1"}],
            }
            store.add_failure(failure)
            store.add_failure(failure)
            assert len(store.failures) == 1, store.failures
            assert store.failures[0]["count"] == 2, store.failures[0]
            store.add_skill({"name": "tile", "scope": {"soc": ["ascend950"]}})
            store.add_skill({"name": "tile", "scope": {"soc": ["ascend950"]}})
            assert len(store.skills) == 1, store.skills
        check("dedupe: fact/failure/skill keys", dedupe)

        def extraction_and_resolve():
            store = MemoryStore(os.path.join(tmp, "extract"))
            features = {
                "op": "swi_glu", "op_family": "elementwise", "soc": "ascend-950pr",
                "dtype": ["float16"], "form": ["ascendc"],
                "shape": (1, 128), "dim": -1,
            }
            result = {
                "status": "fail_correctness",
                "case_summary": {"failed": 1, "total": 3},
                "cases": [{"passed": False, "shape": "(1, 128)",
                           "dtype": "float16", "dim": -1}],
            }
            row = store.record_failure_from_result(result, features, "run-1", "swi_glu")
            assert row and row["stage"] == "correctness", row
            assert row["scope"]["op"] == ["swi_glu"], row
            assert row["minimal_repro"]["failed_cases"], row
            resolved = store.resolve_failures(features, "run-2", "pass")
            assert resolved == 1, resolved
            assert store.failures[0]["status"] == "resolved"
        check("failure extraction -> resolve on pass", extraction_and_resolve)

        def context_budget():
            store = MemoryStore(os.path.join(tmp, "context"))
            store.add_fact({
                "statement": "950 only fact", "scope": {"soc": ["ascend950"]},
                "authority": "human", "confidence": "verified",
            })
            store.add_failure({
                "stage": "runtime", "signature": "507035", "scope": {"soc": ["ascend950"]},
                "workaround": "SIMT",
            })
            store.add_skill({
                "name": "tile", "scope": {"soc": ["ascend950"]},
                "template_ref": "x.cpp", "confidence": "candidate",
            })
            ctx = store.context({"soc": "ascend-950pr", "form": ["ascendc"]})
            assert "950 only fact" in ctx and "507035" in ctx and "tile" in ctx, ctx
            assert "why:" in ctx, ctx
            ctx910 = store.context({"soc": "ascend-910b", "form": ["ascendc"]})
            assert "950 only fact" not in ctx910, ctx910
        check("context: scope-filtered + explains why", context_budget)

        def concurrent_stale_views():
            d = os.path.join(tmp, "concurrent")
            a = MemoryStore(d)
            b = MemoryStore(d)
            a.add_fact({"statement": "f1", "scope": {"soc": ["ascend950"]}})
            b.add_fact({"statement": "f2", "scope": {"soc": ["ascend950"]}})
            c = MemoryStore(d)
            assert len(c.facts) == 2, [r.get("statement") for r in c.facts]
        check("transaction: stale view does not drop the other writer", concurrent_stale_views)

        def decay_verify():
            d = os.path.join(tmp, "decay")
            store = MemoryStore(d)
            fact = store.add_fact({"statement": "old fact", "scope": {"soc": ["ascend950"]}})
            store.facts[0]["updated"] = "2020-01-01T00:00:00+00:00"
            store.save()
            dry = store.decay(apply=False)
            assert any(x["id"] == fact["id"] for x in dry["would_mark"]), dry
            applied = store.decay(apply=True)
            assert any(x["id"] == fact["id"] for x in applied["marked"]), applied
            assert next(r for r in store.facts if r["id"] == fact["id"])["status"] == "stale"
            row = store.verify(fact["id"], note="rechecked")
            assert row and row["status"] == "active", row
        check("decay marks stale; verify restores active", decay_verify)

        def promote_skill():
            d = os.path.join(tmp, "promote")
            store = MemoryStore(d)
            template = os.path.join(tmp, "tile.cpp")
            with open(template, "w", encoding="utf-8") as fh:
                fh.write("// tile template\n")
            features = {
                "op": "swi_glu", "op_family": "elementwise", "soc": "ascend-950pr",
                "dtype": ["float16"], "form": ["ascendc"],
            }
            run_dirs = []
            for i in (1, 2):
                run_dir = os.path.join(tmp, "run%d" % i)
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
                    json.dump({"task": "swi_glu", "run_id": "r%d" % i,
                               "device": "npu", "hardware": "ascend-950pr",
                               "best": {"status": "pass", "speedup": 1.2}}, fh)
                run_dirs.append(run_dir)
            first = store.promote_skill_from_run(run_dirs[0], "tile", template, features)
            assert first["status"] == "upserted" and first["confidence"] == "candidate", first
            second = store.promote_skill_from_run(run_dirs[1], "tile", template, features)
            assert second["status"] == "upserted" and second["confidence"] == "verified", second
        check("promote-skill: candidate -> verified on 2nd pass", promote_skill)

        def conflicts():
            d = os.path.join(tmp, "conflicts")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "strategies.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "id": "stg-test", "key": "tile", "statement": "tile it",
                    "conditions": {"hardware": ["ascend-950pr"], "dtypes": ["float16"],
                                   "op_class": ["elementwise"]},
                    "evidence": [
                        {"outcome": "success", "hardware": "ascend-950pr",
                         "dtype": "float16", "measured_speedup": 1.2},
                        {"outcome": "failure", "hardware": "ascend-950pr",
                         "dtype": "float16", "measured_speedup": 0.9},
                    ],
                    "stats": {}, "confidence": "candidate",
                }) + "\n")
            store = MemoryStore(d)
            report = store.conflicts()
            assert report["strategies"], report
            features = {"op": "swi_glu", "op_family": "elementwise",
                        "soc": "ascend-950pr", "dtype": ["float16"], "form": ["ascendc"]}
            store.add_failure({"stage": "runtime", "signature": "boom",
                               "scope": scope_of(features), "status": "open"})
            store.add_skill({"name": "tile", "scope": scope_of(features),
                             "confidence": "verified", "template_ref": "x"})
            report = store.conflicts()
            assert report["failure_vs_skill"], report
            assert report["total"] >= 2, report
        check("conflicts: strategy success/failure + open failure vs verified skill", conflicts)

        def prompt_memory():
            from agent.prompts import build_task_prompt
            from harness.tasks import load_task
            task = load_task(os.path.join(ROOT, "tasks", "level1", "vector_add.py"))
            prompt = build_task_prompt(task, "cpu", 4, memory="## Operator memory\n- fact-test")
            assert "fact-test" in prompt, prompt[:500]
        check("agent prompt carries the memory block", prompt_memory)

        def python_loop_memory():
            kb = os.path.join(tmp, "py_kb")
            out_dir = os.path.join(tmp, "py_runs")
            cmd = [
                sys.executable, "-m", "agent.loop",
                "--task", os.path.join(ROOT, "tasks", "level1", "vector_add.py"),
                "--provider", "replay",
                "--replay-file", os.path.join(ROOT, "configs", "replay_vector_add.example.json"),
                "--device", "cpu", "--hardware", "ascend-910b",
                "--knowledge-root", kb, "--out-dir", out_dir,
                "--max-iters", "6", "--target-speedup", "0.5", "--timeout", "180",
                "--no-distill", "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=600)
            assert proc.returncode == 0, (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:])
            metas = glob.glob(os.path.join(out_dir, "*", "meta.json"))
            assert metas, out_dir
            meta = json.load(open(metas[0], encoding="utf-8"))
            assert meta["rsi"]["memory"]["failures"] >= 1, meta["rsi"]["memory"]
            rows = [json.loads(line) for line in
                    open(os.path.join(kb, "failures.jsonl"), encoding="utf-8") if line.strip()]
            assert rows, "no failure written"
            assert rows[0]["scope"].get("soc") == ["cpu"], rows[0]["scope"]
            assert rows[0]["status"] == "resolved", rows[0]["status"]
        check("python loop: CPU mechanics failure stays under soc=cpu and resolves", python_loop_memory)


    print("\npassed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())