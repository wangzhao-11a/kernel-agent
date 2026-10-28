"""Self-test: soft-scope analogy priors and failure text similarity (CPU only)."""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsi.memory import MemoryStore, soft_scope_match  # noqa: E402


def build_store(tmp):
    store = MemoryStore(os.path.join(tmp, "mem"))
    store.add_fact({
        "statement": "softmax on 910b benefits from online rescaling",
        "scope": {"op": ["softmax"], "soc": ["ascend-910b"], "dtype": ["float16"]},
        "authority": "experiment", "confidence": "verified",
    })
    store.add_fact({
        "statement": "matmul wants fractal layouts",
        "scope": {"op": ["matmul"], "soc": ["ascend-910b"]},
        "authority": "experiment", "confidence": "verified",
    })
    store.add_failure({
        "stage": "runtime",
        "signature": "bank conflict on transposed shared-memory load",
        "scope": {"op": ["transpose"], "soc": ["ascend-910b"]},
        "workaround": "pad the shared tile by one element",
        "status": "resolved",
    })
    return store


def check_strict_vs_soft(tmp):
    store = build_store(tmp)
    features = {"op": "softmax", "soc": "ascend-910c", "dtype": ["float16"]}
    strict = store.retrieve(features, kinds=("fact",), top_k=5)
    assert not any("softmax" in (h["record"].get("statement") or "") for h in strict), strict
    soft = store.retrieve(features, kinds=("fact",), top_k=5, allow_soft=True)
    hit = next(h for h in soft if "softmax" in (h["record"].get("statement") or ""))
    assert hit["soft"] is True, hit
    assert "UNVERIFIED analogy" in " ".join(hit["why"]), hit
    assert hit["score"] < 0.3, hit
    # unrelated op must not become an analogy even though the SoC matches
    assert not any("matmul" in (h["record"].get("statement") or "") for h in soft), soft

    score, reasons, conflicts = soft_scope_match(
        {"op": ["softmax"], "soc": ["ascend-910b"]},
        {"op": "softmax", "soc": "ascend-910c"})
    assert score > 0 and any("soc differs" in c for c in conflicts), (reasons, conflicts)


def check_context_label(tmp):
    store = build_store(tmp)
    features = {"op": "softmax", "soc": "ascend-910c", "dtype": ["float16"]}
    strict_ctx = store.context(features, kinds=("fact",))
    assert "online rescaling" not in strict_ctx, strict_ctx
    soft_ctx = store.context(features, kinds=("fact",), allow_soft=True)
    assert "UNVERIFIED ANALOGY" in soft_ctx and "online rescaling" in soft_ctx, soft_ctx
    assert "verify with one experiment" in soft_ctx, soft_ctx


def check_failure_similarity(tmp):
    store = build_store(tmp)
    hits = store.similar_failures("bank conflict when loading transposed tile")
    assert hits, "expected a text-similar past failure"
    rec = hits[0]["record"]
    assert "bank conflict" in rec["signature"], rec
    assert "pad the shared tile" in rec.get("workaround", ""), rec
    assert store.similar_failures("qqzzxx") == []


def main():
    tmp = tempfile.mkdtemp(prefix="ka_soft_")
    check_strict_vs_soft(tmp)
    check_context_label(tmp)
    check_failure_similarity(tmp)
    print("passed: 3 | failed: 0")


if __name__ == "__main__":
    main()