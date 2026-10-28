"""Meta-eval on a synthetic operator terrain (CPU only).

Compares search policies under the same fixed evaluation and token budgets:

    single-chain : one candidate, accept only improvements (the original loop)
    beam         : top-k beam with restarts
    ucb          : UCB bandit over tile/split/mode moves
    thompson     : Thompson-sampling bandit over the same moves
    staged       : cheap noisy screens, then full evaluation of the top few

The terrain has a local optimum at 512, a valley at 1024 and a better optimum
at 2048.  CPU synthetic results are mechanism checks, not hardware results.

    python scripts/meta_eval.py --tag phase3-search-compare \
        --policies single-chain,beam,ucb,thompson,staged \
        --seeds 5 --evals 40 --tokens 6000
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsi.search import POLICIES  # noqa: E402
from rsi.terrain import SyntheticTerrain  # noqa: E402


def _git(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _summary(rows, threshold):
    bests = [r["best_reward"] for r in rows]
    return {
        "pass@1": round(sum(1 for r in rows if r["threshold_hit"]) / len(rows), 4),
        "best_mean": round(statistics.fmean(bests), 4),
        "best_stdev": round(statistics.pstdev(bests), 4) if len(bests) > 1 else 0.0,
        "best_min": round(min(bests), 4),
        "best_max": round(max(bests), 4),
        "evals_mean": round(statistics.fmean([r["evals"] for r in rows]), 2),
        "full_evals_mean": round(statistics.fmean([r["full_evals"] for r in rows]), 2),
        "cheap_evals_mean": round(statistics.fmean([r["cheap_evals"] for r in rows]), 2),
        "tokens_mean": round(statistics.fmean([r["tokens"] for r in rows]), 1),
        "restarts_mean": round(statistics.fmean([r.get("restarts", 0) for r in rows]), 2),
    }


def run(args):
    policies = [p.strip() for p in str(args.policies or args.policy).split(",") if p.strip()]
    for p in policies:
        if p not in POLICIES:
            raise SystemExit("unknown policy %r (choose %s)" % (p, ",".join(POLICIES)))
    seeds = [int(args.seed0) + i for i in range(max(1, int(args.seeds)))]
    payload = {
        "schema_version": 1,
        "tag": args.tag,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_tag": _git(["git", "describe", "--tags", "--exact-match", "HEAD"]),
        "device": "cpu",
        "mode": "synthetic-terrain",
        "note": "CPU synthetic mechanism check only; not a hardware/operator result",
        "threshold": float(args.threshold),
        "budget": {"evals": int(args.evals), "tokens": int(args.tokens)},
        "seeds": seeds,
        "policies": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for name in policies:
        rows = []
        for seed in seeds:
            terrain = SyntheticTerrain(seed, args.threshold)
            result = POLICIES[name](terrain, args.evals, args.tokens)
            rows.append(dict(seed=seed, **result))
        payload["policies"][name] = {
            "summary": _summary(rows, args.threshold),
            "runs": rows,
        }
    base = payload["policies"].get("single-chain")
    if base:
        for name, block in payload["policies"].items():
            if name == "single-chain":
                continue
            s = block["summary"]
            block["vs_single_chain"] = {
                "pass@1_delta": round(s["pass@1"] - base["summary"]["pass@1"], 4),
                "best_mean_delta": round(s["best_mean"] - base["summary"]["best_mean"], 4),
                "tokens_delta": round(s["tokens_mean"] - base["summary"]["tokens_mean"], 1),
            }
    # Keep the single-policy shape for backwards compatibility with the baseline file.
    if len(policies) == 1:
        block = payload["policies"][policies[0]]
        payload["policy"] = policies[0]
        payload["summary"] = block["summary"]
        payload["runs"] = block["runs"]
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="meta_eval")
    ap.add_argument("--tag", default="single-chain-baseline")
    ap.add_argument("--policy", default="single-chain")
    ap.add_argument("--policies", default="",
                    help="comma-separated: single-chain,beam,ucb,thompson,staged")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--evals", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--threshold", type=float, default=2.2)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "meta_eval"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    payload = run(args)
    path = os.path.join(args.out, args.tag + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("meta-eval tag=%s seeds=%d evals<=%d tokens<=%d" % (
        args.tag, len(payload["seeds"]), args.evals, args.tokens))
    print("%-14s %7s %10s %9s %9s %9s %9s %9s" % (
        "policy", "pass@1", "best_mean", "best_std", "best_min", "best_max",
        "evals", "tokens"))
    for name, block in payload["policies"].items():
        s = block["summary"]
        print("%-14s %7.2f %10.3f %9.3f %9.3f %9.3f %9.1f %9.0f" % (
            name, s["pass@1"], s["best_mean"], s["best_stdev"], s["best_min"],
            s["best_max"], s["evals_mean"], s["tokens_mean"]))
    print("result: %s" % path)
    print("note: CPU synthetic terrain; mechanism-only, not a hardware result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())