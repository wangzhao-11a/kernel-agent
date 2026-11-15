"""Learning edge self-test: facts/failures reach the curriculum plan.

The plan used to be built from run statistics alone; the constraints recorded
in operator memory (open failures, active facts) never influenced which task
runs next.  This checks that build_plan matches those records against each
task's scope, attaches them to the plan row, and lets an open failure raise the
task's priority.
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi.evolve import build_audit, build_plan  # noqa: E402
from rsi.memory import MemoryStore  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS.append(("FAIL", name, exc))
        print("FAIL  %s: %s" % (name, exc))
        import traceback
        traceback.print_exc()
        return
    RESULTS.append(("PASS", name, None))
    print("PASS  %s" % name)


def _plans(tmp, knowledge_root):
    report = build_audit(runs_root=os.path.join(tmp, "no_runs"),
                         knowledge_root=knowledge_root,
                         tasks_root=os.path.join(ROOT, "tasks"))
    return report, {row["task"]: row for row in build_plan(report, top=200)}


def check_plan_uses_memory(tmp):
    empty_kb = os.path.join(tmp, "kb_empty")
    kb = os.path.join(tmp, "kb_mem")
    mem = MemoryStore(kb)
    failure = mem.add_failure({
        "statement": "errcode 264 on the reduction path when the outer dim is prime",
        "stage": "build", "signature": "errcode264-prime-outer",
        "why": "observed on the softmax family",
        "scope": {"soc": ["ascend-910b"], "dtype": ["float32"], "op_family": ["reduction"]},
    })
    fact = mem.add_fact({
        "statement": "DataCopyPad blockCount must stay below 4095 on 910B",
        "why": "hardware limit",
        "scope": {"soc": ["ascend-910b"], "dtype": ["float32"], "op_family": ["reduction"]},
        "authority": "measurement", "status": "active", "confidence": "candidate",
    })

    report, with_memory = _plans(tmp, kb)
    _report0, without_memory = _plans(tmp, empty_kb)

    assert "softmax" in with_memory and "softmax" in without_memory, sorted(with_memory)[:5]
    soft_with = with_memory["softmax"]
    soft_without = without_memory["softmax"]
    ids = [row["id"] for row in soft_with["constraints"] if row.get("kind") == "failure"]
    assert failure["id"] in ids, soft_with["constraints"]
    assert any(row.get("kind") == "fact" and row.get("id") == fact["id"]
               for row in soft_with["constraints"]), soft_with["constraints"]
    assert soft_with["score"] > soft_without["score"], (soft_with["score"], soft_without["score"])
    assert any("open failure memory" in reason for reason in soft_with["reasons"]), soft_with["reasons"]
    # The audit carries the records the planner matched on.
    assert report["memory"].get("open_failure_records"), report["memory"]

    # A task outside the failure's scope is unaffected.
    assert "vector_add" in with_memory, sorted(with_memory)[:5]
    vec_with = with_memory["vector_add"]
    vec_without = without_memory["vector_add"]
    assert not [row for row in vec_with["constraints"] if row.get("kind") == "failure"], \
        vec_with["constraints"]
    assert vec_with["score"] == vec_without["score"], (vec_with["score"], vec_without["score"])


def main():
    tmp = tempfile.mkdtemp(prefix="plan-memory-")
    check("open failures and facts are matched into the plan by scope",
          lambda: check_plan_uses_memory(tmp))
    failed = [row for row in RESULTS if row[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())