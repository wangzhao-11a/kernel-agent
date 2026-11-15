"""M2 self-test: memory/exemplar/priors injection becomes index + on-demand recall.

Checks (with before/after character counts printed):
  1. first-turn memory+exemplars+priors block is < 3000 chars
  2. recall resolves all five id prefixes (f-/fl-/sk-/s-/ex-)
  3. unknown id returns a clear error, not "Unknown action"
  4. context_index output never exceeds max_chars
  5. replay e2e: the message the MODEL receives contains the record body
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.loop import _recall_record  # noqa: E402
from rsi.exemplars import record_exemplar, render_exemplars, render_exemplars_index  # noqa: E402
from rsi.features import task_features  # noqa: E402
from rsi.memory import MemoryStore  # noqa: E402
from rsi.retrieve import priors_block, retrieve  # noqa: E402
from rsi.store import StrategyStore  # noqa: E402

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


def _seed(root):
    store = StrategyStore(root)
    mem = MemoryStore(root)
    st = store.find_by_key("online_softmax")
    store.record_evidence(st["id"], "run-1", "softmax", 1.4, "success",
                          run_id="run-1", hardware="cpu", dtype="float32")
    mem.add_fact({"statement": "DataCopyPad blockCount limit is 4095; split above that",
                  "scope": {"soc": ["cpu"]}, "confidence": "verified"})
    mem.add_failure({"stage": "runtime", "signature": "errcode 264 on prime outer",
                     "scope": {"op_family": ["reduction"]}, "minimal_repro": {"task": "softmax"}})
    mem.add_skill({"name": "softmax_tile", "scope": {"soc": ["cpu"]},
                   "template_ref": "solutions/examples/softmax_manual.py",
                   "confidence": "verified"})
    record_exemplar(root, "softmax", "import torch\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n",
                    1.4, hardware="cpu", profile_top=["aten::softmax"], op_class="reduction")
    return store, mem


def check_char_budget(tmp):
    root = os.path.join(tmp, "kb")
    store, mem = _seed(root)
    feats = {"name": "softmax", "tags": ["softmax", "reduction"], "op_class": "reduction",
             "dtypes": ["float32"]}
    matched = retrieve(store, feats, "cpu", top_k=5)
    combos = []
    priors = priors_block(feats, "cpu", matched, combos)
    exemplars = [{"id": "ex-1", "task": "softmax", "form": "torch_fused", "speedup": 1.4,
                  "bottleneck": "aten::softmax"}]

    before_memory = mem.context({"op": "softmax", "op_family": "reduction", "soc": "cpu",
                                 "dtype": ["float32"], "task": "softmax"},
                                kinds=("fact", "failure", "skill"))
    before_ex = render_exemplars(exemplars)
    before_priors = priors_block(feats, "cpu", matched, combos)
    before = len(before_memory) + len(before_ex) + len(before_priors)

    after_memory = mem.context_index({"op": "softmax", "op_family": "reduction", "soc": "cpu",
                                      "dtype": ["float32"], "task": "softmax"},
                                     kinds=("fact", "failure", "skill"))
    after_ex = render_exemplars_index(exemplars)
    after_priors = priors_block(feats, "cpu", matched, combos)
    after = len(after_memory) + len(after_ex) + len(after_priors)

    print("  BEFORE memory=%d exemplars=%d priors=%d total=%d" % (
        len(before_memory), len(before_ex), len(before_priors), before))
    print("  AFTER  memory=%d exemplars=%d priors=%d total=%d" % (
        len(after_memory), len(after_ex), len(after_priors), after))
    assert after < 3000, "first-turn memory+exemplars+priors = %d (must be < 3000)" % after
    assert len(after_memory) <= 1400, len(after_memory)
    assert len(after_ex) <= 900, len(after_ex)


def check_recall_prefixes(tmp):
    root = os.path.join(tmp, "kb")
    store, mem = _seed(root)
    fact_id = mem.facts[0]["id"]
    failure_id = mem.failures[0]["id"]
    skill_id = mem.skills[0]["id"]
    strategy_id = store.find_by_key("online_softmax")["id"]
    from rsi.exemplars import _read_jsonl, EXEMPLARS_FILE
    ex_rows = _read_jsonl(os.path.join(root, EXEMPLARS_FILE))
    exemplar_id = ex_rows[0]["id"] if ex_rows else "ex-"
    cases = {
        fact_id: "DataCopyPad",
        failure_id: "errcode 264",
        skill_id: "softmax_tile",
        strategy_id: "online softmax",
        exemplar_id: "softmax",
    }
    for rid, needle in cases.items():
        body = _recall_record(store, mem, rid, root)
        assert body is not None, rid
        assert needle.lower() in body.lower(), (rid, needle, body[:200])
    # the ex- prefix must also resolve to one exemplar
    assert _recall_record(store, mem, "ex-", root) is not None
    unknown = _recall_record(store, mem, "zz-999", root)
    assert unknown is None, unknown


def check_index_max_chars(tmp):
    root = os.path.join(tmp, "kb")
    store, mem = _seed(root)
    feats = {"op": "softmax", "op_family": "reduction", "soc": "cpu",
             "dtype": ["float32"], "task": "softmax"}
    out = mem.context_index(feats, max_chars=200)
    assert len(out) <= 200, len(out)
    out2 = mem.context_index(feats, max_chars=50)
    assert len(out2) <= 50, len(out2)


def check_replay_e2e(tmp):
    """The model must receive the record body, not an 'Unknown action'."""
    root = os.path.join(tmp, "kb")
    store, mem = _seed(root)
    fact_id = mem.facts[0]["id"]
    task_path = os.path.join(ROOT, "tasks", "level1", "softmax.py")
    good = "import torch\n\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n"
    replay = os.path.join(tmp, "replay.json")
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": [
            {"action": "recall", "id": fact_id},
            {"action": "write_solution", "content": good, "strategies": [],
             "expected_speedup": 1.0},
            {"action": "finish", "summary": "recall e2e"},
        ]}, fh)
    out_dir = os.path.join(tmp, "runs")
    proc = subprocess.run([
        sys.executable, "-m", "agent.loop", "--task", task_path, "--provider", "replay",
        "--replay-file", replay, "--device", "cpu", "--knowledge-root", root,
        "--out-dir", out_dir, "--max-iters", "3", "--target-speedup", "0.5",
        "--timeout", "300", "--profiler", "off", "--no-distill",
    ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    metas = [os.path.join(out_dir, d, "meta.json") for d in os.listdir(out_dir)]
    run_dir = os.path.dirname(metas[0])
    transcript = [json.loads(line) for line in
                  open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8")
                  if line.strip()]
    user_msgs = [r.get("content") or "" for r in transcript if r.get("role") == "user"]
    assert any("DataCopyPad" in msg for msg in user_msgs), user_msgs[-4:]
    assert any("Memory record %s" % fact_id in msg for msg in user_msgs), user_msgs[-4:]
    assert not any("Unknown action 'recall'" in msg for msg in user_msgs), user_msgs[-4:]
    # also the recall log row exists
    assert any(r.get("role") == "recall" for r in transcript), transcript[-5:]


def main():
    tmp = tempfile.mkdtemp(prefix="m2-index-")
    check("first-turn memory+exemplars+priors < 3000 (before/after printed)",
          lambda: check_char_budget(tmp))
    check("recall resolves f-/fl-/sk-/s-/ex- ids", lambda: check_recall_prefixes(tmp))
    check("context_index never exceeds max_chars", lambda: check_index_max_chars(tmp))
    check("replay e2e: model receives the record body", lambda: check_replay_e2e(tmp))
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print("\npassed: %d | failed: %d" % (len(RESULTS) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
