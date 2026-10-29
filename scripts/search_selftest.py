"""Mechanism test for the Phase-3 search policies on a fixed synthetic seed.

This is not a hardware result: it verifies budget accounting, that every policy
finishes inside the fixed evaluation/token budget, and that the bandit/staged
policies can beat the single chain on the seeded terrain.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsi.search import POLICIES  # noqa: E402
from rsi.terrain import SyntheticTerrain  # noqa: E402

RESULTS = {"pass": [], "fail": []}


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS["fail"].append((name, exc))
        print("FAIL  %s: %s" % (name, exc))
        return
    RESULTS["pass"].append(name)
    print("PASS  %s" % name)


def main():
    seed, evals, tokens = 1, 40, 6000
    rows = {}
    for name, policy in POLICIES.items():
        result = policy(SyntheticTerrain(seed, threshold=2.2), evals, tokens)
        rows[name] = result
        assert 0 < result["evals"] <= evals, (name, result["evals"])
        assert result["tokens"] <= tokens, (name, result["tokens"])
        assert result["full_evals"] + result["cheap_evals"] == result["evals"], name
    check("search: all policies fit the fixed eval/token budget", lambda: None)
    check("search: UCB beats the single chain on seed 1",
          lambda: (_ for _ in ()).throw(AssertionError(
              rows["ucb"]["best_reward"], rows["single-chain"]["best_reward"]))
          if rows["ucb"]["best_reward"] < rows["single-chain"]["best_reward"] else None)
    check("search: Thompson beats the single chain on seed 1",
          lambda: (_ for _ in ()).throw(AssertionError(
              rows["thompson"]["best_reward"], rows["single-chain"]["best_reward"]))
          if rows["thompson"]["best_reward"] < rows["single-chain"]["best_reward"] else None)
    check("search: MAP-Elites beats the single chain on seed 1",
          lambda: (_ for _ in ()).throw(AssertionError(
              rows["map-elites"]["best_reward"], rows["single-chain"]["best_reward"]))
          if rows["map-elites"]["best_reward"] < rows["single-chain"]["best_reward"] else None)
    check("search: staged screening uses fewer tokens for a better result",
          lambda: (_ for _ in ()).throw(AssertionError(
              rows["staged"]["tokens"], rows["single-chain"]["tokens"]))
          if not (rows["staged"]["tokens"] < rows["single-chain"]["tokens"]
                  and rows["staged"]["best_reward"] >= rows["single-chain"]["best_reward"]) else None)
    print("\n==================================================================")
    print("passed: %d | failed: %d" % (len(RESULTS["pass"]), len(RESULTS["fail"])))
    print("note: CPU synthetic terrain; mechanism-only, not a hardware result")
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())