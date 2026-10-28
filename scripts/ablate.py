"""Component ablations with explicit, reproducible metrics (CPU only).

memory:
  Builds a labeled fact set (attention/gemm/elementwise/norm) and queries
  (a) the same scope, (b) the same op on a different SoC, (c) a family removed
  from the store (leave-one-family-out).  Strict retrieval must have zero recall
  on (b) and (c); soft-scope adds recall on (b) but labels every hit unverified.

profiling:
  Replay runs cannot measure a real profiler's effect, so this uses the staged
  screening vs single-chain search result as an observation-budget proxy and
  labels it as such.  It is not a hardware profiler ablation.

    python scripts/ablate.py --tag ci-ablation --seeds 1 --evals 12 --tokens 800
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsi.memory import MemoryStore  # noqa: E402
from rsi.search import POLICIES  # noqa: E402
from rsi.terrain import SyntheticTerrain  # noqa: E402

FAMILIES = ("attention", "gemm", "elementwise", "norm")
FACTS = {
    "attention": ("online softmax rescaling avoids a second pass", "softmax"),
    "gemm": ("block tiling keeps the working set on chip", "matmul"),
    "elementwise": ("vectorized access removes scalar overhead", "vector_add"),
    "norm": ("two-pass reduction reuses the mean and variance", "rmsnorm"),
}


def _build(root, exclude=None):
    store = MemoryStore(root)
    for family in FAMILIES:
        if family == exclude:
            continue
        statement, op = FACTS[family]
        store.add_fact({
            "statement": "[%s] %s" % (family, statement),
            "scope": {"op_family": [family], "op": [op], "soc": ["ascend-910b"]},
            "authority": "experiment", "confidence": "verified",
        })
    return store


def _query(family, soc):
    return {"op_family": family, "op": FACTS[family][1], "soc": soc, "dtype": ["float16"]}


def _hits(store, features, allow_soft):
    return store.retrieve(features, top_k=3, kinds=("fact",), allow_soft=allow_soft)


def _relevant(hit, family):
    return hit["record"].get("statement", "").startswith("[%s]" % family)


def memory_metrics(tmp):
    out = {}
    for variant, allow_soft in (("strict", False), ("soft", True)):
        same = cross = 0
        soft_flags = 0
        store = _build(os.path.join(tmp, "m_" + variant))
        for family in FAMILIES:
            hits = _hits(store, _query(family, "ascend-910b"), allow_soft)
            same += int(any(_relevant(h, family) for h in hits))
            hits = _hits(store, _query(family, "ascend-910c"), allow_soft)
            cross += int(any(_relevant(h, family) for h in hits))
            soft_flags += sum(1 for h in hits if h.get("soft"))
        out[variant] = {
            "same_scope_recall": round(same / len(FAMILIES), 3),
            "cross_soc_recall": round(cross / len(FAMILIES), 3),
            "cross_soc_soft_hits": soft_flags,
        }
    out["memory_off"] = {"same_scope_recall": 0.0, "cross_soc_recall": 0.0,
                         "cross_soc_soft_hits": 0}
    return out


def lofo_metrics(tmp):
    rows = {}
    for family in FAMILIES:
        store = _build(os.path.join(tmp, "lofo_" + family), exclude=family)
        strict = _hits(store, _query(family, "ascend-910b"), False)
        soft = _hits(store, _query(family, "ascend-910c"), True)
        contamination = [h["id"] for h in strict + soft if not _relevant(h, family)]
        rows[family] = {
            "strict_recall": int(any(_relevant(h, family) for h in strict)),
            "soft_recall": int(any(_relevant(h, family) for h in soft)),
            "contaminating_ids": contamination,
        }
    return rows


def screening_metrics(seeds, evals, tokens, threshold):
    result = {}
    for name in ("single-chain", "staged"):
        rows = []
        for seed in seeds:
            terrain = SyntheticTerrain(seed, threshold)
            rows.append(POLICIES[name](terrain, evals, tokens))
        result[name] = {
            "best_mean": round(sum(r["best_reward"] for r in rows) / len(rows), 4),
            "pass@1": round(sum(1 for r in rows if r["threshold_hit"]) / len(rows), 4),
            "tokens_mean": round(sum(r["tokens"] for r in rows) / len(rows), 1),
        }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ablate")
    ap.add_argument("--tag", default="ablation")
    ap.add_argument("--components", default="memory,lofo,screening")
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--threshold", type=float, default=2.2)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "meta_eval"))
    args = ap.parse_args(argv)
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    tmp = tempfile.mkdtemp(prefix="ka_ablate_")
    payload = {
        "schema_version": 1,
        "tag": args.tag,
        "device": "cpu",
        "note": "CPU mechanism checks only; memory metrics are synthetic, "
                "screening is a search-budget proxy (not a profiler ablation)",
        "seeds": [int(args.seed0) + i for i in range(max(1, args.seeds))],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if "memory" in wanted:
        payload["memory"] = memory_metrics(tmp)
    if "lofo" in wanted:
        payload["leave_one_family_out"] = lofo_metrics(tmp)
    if "screening" in wanted:
        payload["screening_proxy"] = screening_metrics(
            payload["seeds"], args.evals, args.tokens, args.threshold)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.tag + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("schema_version", "tag",
                                                                   "device", "note", "timestamp")},
                     ensure_ascii=False, indent=2))
    print("result: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())