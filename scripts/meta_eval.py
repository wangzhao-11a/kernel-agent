"""Minimal meta-eval on a synthetic operator terrain (CPU only).

The real agent needs an API and a device.  This script isolates the *search
policy* question on a deterministic, noisy terrain:

    candidate = (tile, split, mode)
    reward    = synthetic speedup with multiple local optima + evaluation noise

The baseline policy is the original single chain: propose one neighbouring
candidate, evaluate it, keep it only if it beats the current best.  Phase 3 adds
beam / bandit / early-stop policies and compares them under the same fixed
evaluation and token budgets.

    python scripts/meta_eval.py --tag baseline-single-chain --seeds 5 \
        --evals 40 --tokens 6000

CPU synthetic results are mechanism checks, not hardware performance results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TILES = (64, 128, 256, 512, 1024, 2048, 4096)
MODES = (0, 1, 2)
TOKENS_PER_PROPOSAL = 120


class SyntheticTerrain:
    def __init__(self, seed: int, threshold: float = 1.55):
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.rng = random.Random(seed)

    def reward(self, candidate, step: int = 0) -> float:
        tile, split, mode = candidate
        x = math.log2(max(1, int(tile)))
        # Two peaks (512 / 2048) whose position shifts with mode; several local basins.
        peak_a = 0.55 * math.exp(-((x - (9.0 + 0.45 * mode)) ** 2) / 0.55)
        peak_b = 0.40 * math.exp(-((x - (11.0 - 0.35 * mode)) ** 2) / 0.45)
        base = 1.0 + peak_a + peak_b
        # Split sweet spot near 4; too many splits lose to overhead.
        split_f = 0.72 + 0.14 * split - 0.025 * max(0, split - 5) ** 2
        mode_f = 1.0 + 0.07 * mode
        noise = 1.0 + self.rng.gauss(0.0, 0.045)
        return max(0.1, base * split_f * mode_f * noise)

    def neighbour(self, candidate, rng: random.Random):
        tile, split, mode = candidate
        which = rng.randrange(3)
        if which == 0:
            i = TILES.index(tile)
            j = max(0, min(len(TILES) - 1, i + rng.choice((-1, 1))))
            return (TILES[j], split, mode)
        if which == 1:
            return (tile, max(1, min(8, split + rng.choice((-1, 1)))), mode)
        return (tile, split, (mode + rng.choice((-1, 1))) % len(MODES))


def single_chain(terrain: SyntheticTerrain, max_evals: int, token_budget: int):
    """Original loop analogue: one candidate, accept only improvements, no lookahead."""
    rng = random.Random(terrain.seed ^ 0x5EED)
    current = (128, 1, 0)
    best = current
    best_reward = terrain.reward(current, 0)
    history = [(current, round(best_reward, 4))]
    evals, tokens = 1, TOKENS_PER_PROPOSAL
    while evals < max_evals and tokens + TOKENS_PER_PROPOSAL <= token_budget:
        proposal = terrain.neighbour(current, rng)
        reward = terrain.reward(proposal, evals)
        evals += 1
        tokens += TOKENS_PER_PROPOSAL
        history.append((proposal, round(reward, 4)))
        if reward > best_reward:
            best, best_reward = proposal, reward
            current = proposal
    return {
        "best_candidate": list(best),
        "best_reward": round(best_reward, 4),
        "evals": evals,
        "tokens": tokens,
        "threshold_hit": bool(best_reward >= terrain.threshold),
        "history": [{"candidate": list(c), "reward": r} for c, r in history],
    }


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_tag():
    try:
        return subprocess.check_output(["git", "describe", "--tags", "--exact-match", "HEAD"],
                                       cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def run(args) -> dict:
    seeds = [int(args.seed0) + i for i in range(max(1, int(args.seeds)))]
    rows = [single_chain(SyntheticTerrain(s, args.threshold), args.evals, args.tokens)
            for s in seeds]
    bests = [r["best_reward"] for r in rows]
    payload = {
        "schema_version": 1,
        "tag": args.tag,
        "git_commit": _git_commit(),
        "git_tag": _git_tag(),
        "device": "cpu",
        "mode": "synthetic-terrain",
        "note": "CPU synthetic mechanism check only; not a hardware/operator result",
        "policy": args.policy,
        "threshold": args.threshold,
        "budget": {"evals": args.evals, "tokens": args.tokens,
                   "tokens_per_proposal": TOKENS_PER_PROPOSAL},
        "seeds": seeds,
        "summary": {
            "pass@1": round(sum(1 for r in rows if r["threshold_hit"]) / len(rows), 4),
            "best_mean": round(statistics.fmean(bests), 4),
            "best_stdev": round(statistics.pstdev(bests), 4) if len(bests) > 1 else 0.0,
            "best_min": round(min(bests), 4),
            "best_max": round(max(bests), 4),
            "evals_mean": round(statistics.fmean([r["evals"] for r in rows]), 2),
            "tokens_mean": round(statistics.fmean([r["tokens"] for r in rows]), 1),
        },
        "runs": [dict(seed=s, **row) for s, row in zip(seeds, rows)],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="meta_eval")
    ap.add_argument("--tag", default="single-chain-baseline")
    ap.add_argument("--policy", default="single-chain")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--evals", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--threshold", type=float, default=1.55)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "meta_eval"))
    args = ap.parse_args(argv)
    if args.policy != "single-chain":
        print("only --policy single-chain is implemented in the baseline", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    payload = run(args)
    path = os.path.join(args.out, args.tag + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    s = payload["summary"]
    print("meta-eval tag=%s policy=%s seeds=%d evals<=%d tokens<=%d" % (
        args.tag, args.policy, len(payload["seeds"]), args.evals, args.tokens))
    print("pass@1=%.2f best_mean=%.3f best_stdev=%.3f best_min=%.3f best_max=%.3f "
          "evals_mean=%.1f tokens_mean=%.0f" % (
              s["pass@1"], s["best_mean"], s["best_stdev"], s["best_min"],
              s["best_max"], s["evals_mean"], s["tokens_mean"]))
    print("result: %s" % path)
    print("note: CPU synthetic terrain; mechanism-only, not a hardware result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())