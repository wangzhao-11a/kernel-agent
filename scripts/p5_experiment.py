"""P5 controlled experiment: does RSI (kb@T) beat no-RSI on held-out tasks?

Design (see the plan):
  - 12 tasks split by operator family; 4 holdout tasks are never trained on.
  - Training: P4 scheduler on the train tasks with a fixed iteration budget,
    knowledge snapshot kb@T.
  - Evaluation: every holdout task x seed x group (kb@T | no_rsi | seed-only),
    same model/temperature/max_iters/device.
  - Metrics: first-pass iteration, best speedup (P1 ABAB measurements), token
    cost per gain, negative-transfer rate, cheat interceptions; bootstrap CIs.

    export DEEPSEEK_API_KEY=...        # never write it into the repo
    python scripts/p5_experiment.py --stage all --device npu \
        --provider openai --model deepseek-chat --base-url https://api.deepseek.com/v1 \
        --api-key-env DEEPSEEK_API_KEY --train-budget 8 --seeds 2 --max-iters 3 \
        --groups kb,no_rsi --out runs/p5/pilot

Every number comes from an actual run; a pilot is labelled as such.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness import stats as stats_mod  # noqa: E402
from harness.tasks import load_task  # noqa: E402

DEFAULT_HOLDOUT = "softmax,matmul,mla_attention,moe_permute"
DEFAULT_TRAIN = ("vector_add,rmsnorm,decode_attention,grouped_matmul,"
                 "moe_finalize_routing,rope,sdpa_attention,grouped_gemm_swiglu")
TASK_ROOTS = (os.path.join(ROOT, "tasks", "level1"), os.path.join(ROOT, "tasks", "level2"))


def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _find_task(name: str):
    for root in TASK_ROOTS:
        for path in sorted(glob.glob(os.path.join(root, "*.py"))):
            if os.path.basename(path).startswith("_"):
                continue
            try:
                task = load_task(path)
            except Exception:
                continue
            if task.name == name:
                return path
    return None


def _copy_train_tasks(out_dir: str, names) -> str:
    root = os.path.join(out_dir, "train_tasks")
    os.makedirs(root, exist_ok=True)
    missing = []
    for name in names:
        path = _find_task(name)
        if not path:
            missing.append(name)
            continue
        shutil.copy2(path, os.path.join(root, os.path.basename(path)))
    if missing:
        raise SystemExit("train task(s) not found: %s" % ", ".join(missing))
    return root


def _make_seed_kb(out_dir: str) -> str:
    from rsi.store import StrategyStore

    root = os.path.join(out_dir, "kb_seed")
    StrategyStore(root)  # bootstraps the 14 seed practices, no evidence/exemplars
    return root


def _run(cmd, timeout):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _scheduler_cmd(args, out_dir: str, train_root: str, knowledge_root: str,
                   budget: int, stage: int):
    cmd = [
        sys.executable, "-m", "rsi.schedule", "--budget-iters", str(int(budget)),
        "--approve", "auto", "--top", str(max(1, len(args.train_tasks))),
        "--seeds", str(args.train_seeds), "--variants", "base",
        "--max-iters", str(args.train_max_iters), "--target-speedup", str(args.target_speedup),
        "--provider", args.provider, "--model", args.model, "--base-url", args.base_url,
        "--api-key-env", args.api_key_env, "--device", args.device,
        "--timeout", str(args.timeout), "--knowledge-root", knowledge_root,
        "--runs-root", os.path.join(out_dir, "train_runs"),
        "--tasks-root", train_root,
        "--out", os.path.join(out_dir, "schedule", "stage%d" % stage),
        "--versions-root", os.path.join(out_dir, "versions"),
        "--holdout-config", os.path.join(ROOT, "configs", "holdout.json"),
    ]
    return cmd


def _train(args, out_dir: str, train_root: str) -> dict:
    knowledge_root = os.path.join(out_dir, "kb")
    stages = max(1, int(args.stages))
    per_stage = max(1, int(args.train_budget) // stages)
    reports = []
    for stage in range(1, stages + 1):
        cmd = _scheduler_cmd(args, out_dir, train_root, knowledge_root, per_stage, stage)
        proc = _run(cmd, timeout=args.timeout + 300)
        audits = glob.glob(os.path.join(out_dir, "schedule", "stage%d" % stage, "*", "audit.json"))
        audit = json.load(open(audits[-1], encoding="utf-8")) if audits else {}
        reports.append({"stage": stage, "budget": per_stage, "rc": proc.returncode,
                        "audit": audit,
                        "stdout_tail": (proc.stdout or "")[-1200:],
                        "stderr_tail": (proc.stderr or "")[-1200:]})
        if proc.returncode != 0 and not audit:
            break
    kb_at_t = os.path.join(out_dir, "kb_at_T")
    if os.path.isdir(knowledge_root):
        if os.path.isdir(kb_at_t):
            shutil.rmtree(kb_at_t)
        shutil.copytree(knowledge_root, kb_at_t)
    return {"stages": reports, "kb_at_T": kb_at_t}


def _eval_one(args, task_path: str, task_name: str, seed: int, group: str,
              out_dir: str, kb_at_t: str, kb_seed: str) -> dict:
    run_root = os.path.join(out_dir, "eval", task_name, "seed%d" % seed, group)
    os.makedirs(run_root, exist_ok=True)
    knowledge = ""
    if group == "kb":
        knowledge = os.path.join(run_root, "kb")
        if os.path.isdir(kb_at_t):
            shutil.copytree(kb_at_t, knowledge)
        else:
            knowledge = kb_at_t
    elif group == "seed":
        knowledge = os.path.join(run_root, "kb")
        shutil.copytree(kb_seed, knowledge)
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path,
        "--provider", args.provider, "--model", args.model, "--base-url", args.base_url,
        "--api-key-env", args.api_key_env, "--device", args.device,
        "--out-dir", run_root, "--max-iters", str(args.max_iters),
        "--target-speedup", str(args.target_speedup), "--timeout", str(args.timeout),
        "--profiler", "off", "--no-distill",
    ]
    if group == "no_rsi":
        cmd.append("--no-rsi")
    else:
        cmd += ["--knowledge-root", knowledge]
    started = time.time()
    proc = _run(cmd, timeout=args.timeout + 300)
    metas = glob.glob(os.path.join(run_root, "*", "meta.json"))
    meta = json.load(open(metas[-1], encoding="utf-8")) if metas else {}
    attempts = []
    if metas:
        path = os.path.join(os.path.dirname(metas[-1]), "attempts.jsonl")
        if os.path.isfile(path):
            attempts = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    first_pass = None
    for attempt in attempts:
        if attempt.get("status") in ("pass", "pass_unstable"):
            first_pass = attempt.get("iter")
            break
    best = meta.get("best") or {}
    best_status = best.get("status")
    best_speedup = (float(best.get("speedup") or 0.0)
                    if best_status in ("pass", "pass_unstable") else 0.0)
    usage = meta.get("usage") or {}
    return {
        "task": task_name, "seed": seed, "group": group,
        "rc": proc.returncode, "status": best_status,
        "iterations": int(meta.get("iterations") or 0),
        "first_pass_iter": first_pass if first_pass is not None else int(args.max_iters) + 1,
        "best_speedup": round(best_speedup, 4),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "fail_suspicious": sum(1 for a in attempts if a.get("status") == "fail_suspicious"),
        "wall_s": round(time.time() - started, 1),
        "run_dir": os.path.dirname(metas[-1]) if metas else None,
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def _eval(args, out_dir: str, kb_at_t: str, kb_seed: str, stage_label: str) -> list:
    rows = []
    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    for task_name in args.holdout_tasks:
        task_path = _find_task(task_name)
        if not task_path:
            rows.append({"task": task_name, "error": "task not found"})
            continue
        for seed in range(1, int(args.seeds) + 1):
            for group in groups:
                row = _eval_one(args, task_path, task_name, seed, group,
                                out_dir, kb_at_t, kb_seed)
                row["stage"] = stage_label
                rows.append(row)
    return rows


def _median_ci(values) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "ci": None}
    return {"n": len(vals), "median": round(statistics.median(vals), 4),
            "ci": [round(x, 4) for x in stats_mod.bootstrap_ci(vals)]}


SPEEDUP_FLOOR = 1e-2  # log floor for a failed run (avoids log(0))


def _log_speedup(speedup) -> float:
    return math.log(max(float(speedup or 0.0), SPEEDUP_FLOOR))


def _cost_per_gain(row) -> float:
    """Tokens per unit of log gain; undefined (None) when the run has no gain."""
    speedup = float(row.get("best_speedup") or 0.0)
    if speedup <= 1.0:
        return None
    gain = math.log(speedup)
    if gain <= 0:
        return None
    return float(row.get("total_tokens") or 0.0) / gain


def _median_or_none(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def _paired(rows: list) -> dict:
    """Per task x seed: log(speedup) ratio and first-pass iteration difference.

    A log ratio is scale-free, so a 12x task cannot swamp a 1.1x task when the
    pairs are pooled, and it is symmetric between the two groups.
    """
    by_key = {}
    for row in rows:
        by_key.setdefault((row["task"], row["seed"]), {})[row["group"]] = row
    first_diff, log_diff, cost_diff = [], [], []
    cost_missing = 0
    for pair in by_key.values():
        kb, base = pair.get("kb"), pair.get("no_rsi")
        if not kb or not base:
            continue
        first_diff.append(base["first_pass_iter"] - kb["first_pass_iter"])
        log_diff.append(_log_speedup(kb["best_speedup"]) - _log_speedup(base["best_speedup"]))
        kb_cost, base_cost = _cost_per_gain(kb), _cost_per_gain(base)
        if kb_cost is None or base_cost is None:
            cost_missing += 1
        else:
            cost_diff.append(kb_cost - base_cost)
    return {
        "first_pass_improvement": _median_ci(first_diff),
        "log_speedup_ratio": _median_ci(log_diff),
        "log_speedup_ratio_geomean": (round(math.exp(statistics.median(log_diff)), 4)
                                      if log_diff else None),
        "cost_per_gain_difference": _median_ci(cost_diff),
        "cost_pairs_missing": cost_missing,
        "pairs": len(first_diff),
    }


def _negative_transfer(rows: list, threshold: float = 0.03) -> dict:
    kb, base = {}, {}
    for row in rows:
        if row["group"] == "kb":
            kb.setdefault(row["task"], []).append(row["best_speedup"])
        elif row["group"] == "no_rsi":
            base.setdefault(row["task"], []).append(row["best_speedup"])
    worse, tasks = [], []
    for task in sorted(set(kb) & set(base)):
        k = statistics.median(kb[task])
        b = statistics.median(base[task])
        tasks.append({"task": task, "kb": k, "no_rsi": b, "delta": round(k - b, 4)})
        if k < b - threshold:
            worse.append(task)
    return {"tasks": tasks, "worse": worse,
            "rate": (len(worse) / len(tasks)) if tasks else None,
            "threshold": threshold}


def _report(rows: list, args, train: dict) -> dict:
    groups = {}
    for group in sorted({row["group"] for row in rows if row.get("group")}):
        subset = [row for row in rows if row["group"] == group]
        groups[group] = {
            "first_pass_iter": _median_ci([r["first_pass_iter"] for r in subset]),
            "best_speedup": _median_ci([r["best_speedup"] for r in subset]),
            "cost_per_gain": _median_ci([_cost_per_gain(r) for r in subset
                                         if _cost_per_gain(r) is not None]),
            "cost_per_gain_runs": sum(1 for r in subset if _cost_per_gain(r) is not None),
            "tokens": _median_ci([r["total_tokens"] for r in subset]),
            "passes": sum(1 for r in subset if r["status"] in ("pass", "pass_unstable")),
            "n": len(subset),
            "cheat_interceptions": sum(r.get("fail_suspicious", 0) for r in subset),
        }
    paired = _paired(rows)
    transfer = _negative_transfer(rows)
    first_ok = bool((paired["first_pass_improvement"]["ci"] or [0, 0])[0] > 0)
    best_ok = bool((paired["log_speedup_ratio"]["ci"] or [0, 0])[0] > 0)
    # Cost per unit log-gain: only runs that actually have a gain contribute, so a
    # group cannot look better merely by spending more tokens on a flat result.
    cost_ok = None
    kb_cost = (groups.get("kb") or {}).get("cost_per_gain", {}).get("median")
    base_cost = (groups.get("no_rsi") or {}).get("cost_per_gain", {}).get("median")
    if kb_cost is not None and base_cost is not None:
        cost_ok = kb_cost <= base_cost
    return {
        "holdout_tasks": args.holdout_tasks,
        "train_tasks": args.train_tasks,
        "groups": groups,
        "paired": paired,
        "negative_transfer": transfer,
        "pass_lines": {
            "first_pass_or_best": first_ok or best_ok,
            "first_pass": first_ok,
            "best_speedup": best_ok,
            "cost_not_worse": cost_ok,
            "negative_transfer_ok": (transfer["rate"] is not None and transfer["rate"] <= 0.25),
        },
        "train": {"stages": [{"stage": s["stage"], "rc": s["rc"],
                              "used_iters": (s.get("audit") or {}).get("used_iters"),
                              "holdout_score": (s.get("audit") or {}).get("holdout_score")}
                             for s in train.get("stages", [])],
                  "kb_at_T": train.get("kb_at_T")},
    }


def render_markdown(report: dict, args, started: str) -> str:
    lines = [
        "# P5 experiment — RSI on/off on held-out tasks",
        "",
        "> This run is a **%s**: model `%s` via `%s`, device `%s`, temperature 0.2, "
        "max_iters %d, train budget %d iters, seeds %d."
        % ("pilot" if args.pilot else "full-protocol run", args.model, args.provider,
           args.device, args.max_iters, args.train_budget, args.seeds),
        "",
        "- started: %s" % started,
        "- holdout tasks: %s" % ", ".join(args.holdout_tasks),
        "- train tasks: %s" % ", ".join(args.train_tasks),
        "- groups: %s" % args.groups,
        "- holdout tasks were never in the training tasks root.",
        "",
        "## Results",
        "",
        "| group | n | first-pass iters (median, CI) | best speedup (median, CI) | tokens/gain | passes | cheat interceptions |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group, row in report["groups"].items():
        fp, bs, cg = row["first_pass_iter"], row["best_speedup"], row["cost_per_gain"]
        lines.append("| %s | %d | %s %s | %s %s | %s | %d | %d |" % (
            group, row["n"], fp["median"], fp["ci"], bs["median"], bs["ci"],
            cg["median"], row["passes"], row["cheat_interceptions"]))
    paired = report["paired"]
    lines += [
        "",
        "## Paired comparisons (kb - no_rsi, per task x seed)",
        "",
        "- pairs: %d" % paired["pairs"],
        "- first-pass improvement (no_rsi - kb; positive = RSI needs fewer iterations): %s CI %s"
        % (paired["first_pass_improvement"]["median"], paired["first_pass_improvement"]["ci"]),
        "- log-speedup ratio (kb / no_rsi; median %s CI %s, geomean %.3fx)"
        % (paired["log_speedup_ratio"]["median"], paired["log_speedup_ratio"]["ci"],
           paired.get("log_speedup_ratio_geomean") or 1.0),
        "- tokens/gain difference: %s CI %s"
        % (paired["cost_per_gain_difference"]["median"], paired["cost_per_gain_difference"]["ci"]),
        "",
        "## Pass lines",
        "",
        "| line | required | observed | verdict |",
        "| --- | --- | --- | --- |",
        "| first-pass or best | at least one CI lower bound > 0 | first=%s best=%s | %s |" % (
            report["pass_lines"]["first_pass"], report["pass_lines"]["best_speedup"],
            report["pass_lines"]["first_pass_or_best"]),
        "| cost per gain | not worse than control | %s | %s |" % (
            report["pass_lines"]["cost_not_worse"], report["pass_lines"]["cost_not_worse"]),
        "| negative transfer | <= 1/4 of holdout tasks | %s (rate %s) | %s |" % (
            report["negative_transfer"]["worse"], report["negative_transfer"]["rate"],
            report["pass_lines"]["negative_transfer_ok"]),
    ]
    lines += [
        "",
        "## Negative transfer by task",
        "",
        "| task | kb best | no_rsi best | delta |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["negative_transfer"]["tasks"]:
        lines.append("| %s | %s | %s | %s |" % (
            row["task"], row["kb"], row["no_rsi"], row["delta"]))
    lines += [
        "",
        "## Limits",
        "",
        "- The seed-only control is a seed pool, not token-length-matched; it is only run when requested.",
        "- Bootstrap CIs resample task x seed pairs; a pilot has few pairs and wide intervals.",
        "- This is a mechanism/scale-limited run unless the full protocol args are used.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="p5_experiment", description="RSI on/off held-out experiment.")
    ap.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    ap.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    ap.add_argument("--train-tasks", default=DEFAULT_TRAIN)
    ap.add_argument("--groups", default="kb,no_rsi")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-seeds", type=int, default=1,
                    help="seeds per training task (coverage-first default)")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--train-max-iters", type=int, default=None,
                    help="iterations per training run (default: --max-iters)")
    ap.add_argument("--target-speedup", type=float, default=5.0)
    ap.add_argument("--train-budget", type=int, default=24)
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--device", default="npu")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--pilot", action="store_true",
                    help="smaller sample; mark the report as a pilot")
    args = ap.parse_args(argv)
    args.train_max_iters = args.train_max_iters or args.max_iters
    args.holdout_tasks = [x.strip() for x in str(args.holdout).split(",") if x.strip()]
    args.train_tasks = [x.strip() for x in str(args.train_tasks).split(",") if x.strip()]

    stamp = _now()
    out_dir = args.out or os.path.join(ROOT, "runs", "p5", stamp)
    os.makedirs(out_dir, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    train_root = _copy_train_tasks(out_dir, args.train_tasks)
    kb_seed = _make_seed_kb(out_dir)
    train = {"stages": [], "kb_at_T": os.path.join(out_dir, "kb_at_T")}
    if args.stage in ("all", "train"):
        train = _train(args, out_dir, train_root)
    rows = []
    if args.stage in ("all", "eval"):
        rows = _eval(args, out_dir, train.get("kb_at_T"), kb_seed, "final")
    report = _report(rows, args, train)
    result = {
        "schema_version": 1, "kind": "p5", "pilot": bool(args.pilot),
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "provider": args.provider, "base_url": args.base_url,
        "temperature": 0.2, "device": args.device,
        "max_iters": args.max_iters, "seeds": args.seeds,
        "train_budget": args.train_budget, "target_speedup": args.target_speedup,
        "holdout_tasks": args.holdout_tasks, "train_tasks": args.train_tasks,
        "groups": args.groups,
        "rows": rows, "summary": report,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "EXPERIMENT.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report, args, started))
    print(json.dumps({"out": out_dir, "rows": len(rows), "summary": report["pass_lines"]},
                     ensure_ascii=False, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
