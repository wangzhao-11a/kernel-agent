"""M2 re-acceptance: first-turn memory blocks on a realistically filled knowledge base.

The original M2 check ran on a near-empty pool (1986 -> 1758 characters), which
proved almost nothing.  This one fills the store the way a used project looks:

    50 facts + 20 failures + 10 exemplars (+ the bootstrapped strategies)

then builds the exact three blocks agent.loop injects on the first turn and
requires their sum to stay under 3000 characters.  It also checks that the
priors block is an index (one line per practice, body via recall), because it
used to be the largest of the three.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.tasks import load_task  # noqa: E402
from rsi.exemplars import record_exemplar, render_exemplars_index, retrieve_exemplars  # noqa: E402
from rsi.features import hardware_from_device, task_features  # noqa: E402
from rsi.memory import MemoryStore  # noqa: E402
from rsi.retrieve import combos_for, priors_block, retrieve  # noqa: E402
from rsi.store import StrategyStore  # noqa: E402

RESULTS = []
LIMIT = 3000


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


def _fill(root):
    """50 facts + 20 failures + 10 exemplars, all scoped to match the task."""
    mem = MemoryStore(root)
    body = ("UB 到 GM 的写必须走 VECOUT 队列;直接写 GlobalTensor 会 errcode 264。"
            "在该形状下把 blockCount 拆到 4095 以下,并在尾块用 DataCopyPad。") * 4
    for i in range(50):
        mem.add_fact({
            "statement": "fact %02d: DataCopyPad 的 blockCount 上限与 32B 对齐约束,尾块要用 pad" % i,
            "why": "soc=cpu dtype=float32 shape=%d" % i,
            "repro": body, "workaround": body,
            "scope": {"soc": ["cpu"], "dtype": ["float32"]},
            "confidence": "candidate",
        })
    for i in range(20):
        mem.add_failure({
            "statement": "failure %02d: errcode 264 在 shape 外维为质数时出现,先检查对齐" % i,
            "stage": "build", "signature": "errcode264-%d" % i,
            "why": "op_family=reduction form=torch",
            "repro": body, "workaround": body, "status": "open",
            "scope": {"soc": ["cpu"], "dtype": ["float32"]},
        })
    for i in range(10):
        mem.add_skill({
            "name": "skill_%02d" % i,
            "statement": "skill %02d: 用 TQue 分块搬运并用 DataCopyPad 收尾,避免尾块越界" % i,
            "why": "form=ascendc soc=cpu dtype=float32",
            "template": body, "scope": {"soc": ["cpu"], "dtype": ["float32"]},
            "confidence": "candidate",
        })
    for i in range(10):
        source = ("import torch\n\n# variant %d\ndef run(x):\n"
                  "    return torch.softmax(x, -1)\n" % i)
        record_exemplar(root, "task_%d" % i, source, 1.2 + i * 0.1,
                        hardware="cpu", dtypes=["float32"], op_class="reduction",
                        tags=["softmax"], profile_top=["MTE2"])


def _seed_combos(root):
    """Combination rules over the practices the fixture matches.

    A used pool has them, they cannot be recalled by id, and they are the part
    of the priors block that pushed it past its share before the index form.
    """
    store = StrategyStore(root)
    ids = sorted(store.strategies)[:5]
    assert len(ids) >= 5
    for index, members in enumerate(((ids[0], ids[1]), (ids[2], ids[3]), (ids[0], ids[4]))):
        store.upsert_combo(list(members), measured_speedup=1.8 + index * 0.1,
                           task="softmax", trajectory="fixture")


def _blocks(root):
    """The same three blocks agent.loop builds before the first model turn."""
    task = load_task(os.path.join(ROOT, "tasks", "level1", "softmax.py"))
    feats = task_features(task)
    hardware = hardware_from_device("cpu")
    mem_features = {
        "op": task.name,
        "op_family": feats.get("op_class") or "",
        "soc": hardware,
        "dtype": feats.get("dtypes") or [],
        "form": ["triton"],
        "task": task.name,
    }
    store = StrategyStore(root)
    matched = retrieve(store, feats, hardware, top_k=5)
    combos = combos_for(store, [s["id"] for s, _score, _match in matched])
    priors = priors_block(feats, hardware, matched, combos)
    exemplars = retrieve_exemplars(root, feats, hardware, top_k=2)
    exemplar_block = render_exemplars_index(exemplars)
    mem = MemoryStore(root)
    memory_block = mem.context_index(mem_features, kinds=("fact", "failure", "skill"))
    return {
        "memory": memory_block, "exemplars": exemplar_block, "priors": priors,
        "memory_len": len(memory_block), "exemplars_len": len(exemplar_block),
        "priors_len": len(priors),
        "total": len(memory_block) + len(exemplar_block) + len(priors),
        "matched": len(matched),
    }


def check_budget(tmp):
    root = os.path.join(tmp, "kb")
    _fill(root)
    _seed_combos(root)
    blocks = _blocks(root)
    print("  memory=%d exemplars=%d priors=%d total=%d (limit %d, %d priors)"
          % (blocks["memory_len"], blocks["exemplars_len"], blocks["priors_len"],
             blocks["total"], LIMIT, blocks["matched"]))
    assert blocks["matched"] >= 1, "fixture must match at least one prior"
    assert blocks["memory_len"] > 200, "fixture memory block is suspiciously empty"
    assert blocks["priors_len"] > 200, "fixture priors block is suspiciously empty"
    assert blocks["total"] < LIMIT, blocks["total"]
    assert blocks["memory_len"] <= 1400, blocks["memory_len"]
    assert blocks["priors_len"] <= 1400, blocks["priors_len"]


def check_priors_is_an_index(tmp):
    root = os.path.join(tmp, "kb_index")
    _fill(root)
    _seed_combos(root)
    blocks = _blocks(root)
    priors = blocks["priors"]
    assert "recall" in priors, priors[:300]
    lines = priors.splitlines()
    strategy_lines = [line for line in lines if line.startswith("- [stg-")]
    assert strategy_lines, priors
    # One line per practice: no continuation/history line under a strategy.
    for line in strategy_lines:
        assert "no history" in line or "hist " in line, line
    assert len(lines) <= 2 + 5 + 3 + 2, lines  # header + priors + combos + footer/note
    # The body is reachable with the exact id shown in the line.
    from agent.loop import _recall_record
    store = StrategyStore(root)
    mem = MemoryStore(root)
    first_id = strategy_lines[0].split("]")[0].lstrip("- [").strip()
    body = _recall_record(store, mem, first_id, root)
    assert body, first_id
    assert json.loads(body).get("statement"), body[:200]


def main():
    tmp = tempfile.mkdtemp(prefix="m2-budget-")
    check("first-turn memory+exemplars+priors < 3000 on a filled KB", lambda: check_budget(tmp))
    check("priors block is an index with recallable bodies", lambda: check_priors_is_an_index(tmp))
    failed = [row for row in RESULTS if row[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())