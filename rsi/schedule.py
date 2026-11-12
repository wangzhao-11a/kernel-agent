"""Budget-limited outer loop: plan -> run the curriculum -> audit -> maybe roll back.

    python -m rsi.schedule --budget-iters 24 --approve auto --provider openai --device npu
    python -m rsi.schedule --budget-iters 3 --approve auto --provider replay \
        --replay-file configs/replay_softmax.json --variants base --seeds 1

The scheduler reads the ranked plan from `rsi.evolve`, runs `agent.loop` for the
protocol (seeds x variants), stops at the iteration budget, snapshots the
knowledge base before/after, computes the holdout score, and rolls back when the
score drops by more than `--rollback-eps`. Synthetic tasks are approval-gated.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi import version  # noqa: E402
from rsi.evolve import build_audit, build_plan, read_json  # noqa: E402

DEFAULT_KNOWLEDGE = os.path.join(ROOT, "rsi", "knowledge")
DEFAULT_RUNS = os.path.join(ROOT, "runs", "agent")
DEFAULT_OUT = os.path.join(ROOT, "runs", "schedule")
DEFAULT_TASKS = os.path.join(ROOT, "tasks")
DEFAULT_HOLDOUT = os.path.join(ROOT, "configs", "holdout.json")


def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _task_meta(path: str) -> dict:
    from harness.tasks import load_task

    try:
        return load_task(path).meta
    except Exception:
        return {}


def _sidecar(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def _synthetic_approved(path: str) -> bool:
    try:
        return bool(read_json(_sidecar(path)).get("approved"))
    except Exception:
        return False


def _find_task_path(tasks_root: str, name: str) -> str:
    for path in sorted(glob.glob(os.path.join(tasks_root, "**", "*.py"), recursive=True)):
        if "__pycache__" in path or os.path.basename(path).startswith("_"):
            continue
        if os.path.splitext(os.path.basename(path))[0] == name:
            return path
    return ""


def _holdout_eval(args, knowledge_root: str, task_paths, out_dir: str) -> dict:
    if args.skip_holdout_eval or not task_paths:
        return {"score": None, "metric": "skipped", "n_tasks": 0, "tasks": {},
                "reason": "holdout evaluation disabled" if args.skip_holdout_eval
                          else "no holdout tasks configured"}

    def runner(task_path, kb_root, seed, run_dir, max_iters=4, timeout=900.0):
        # Reuse the cycle's provider/model/device settings: the default runner
        # reads them from the environment, which the scheduler does not set.
        return _run_agent_for_holdout(args, task_path, kb_root, run_dir,
                                      max_iters, timeout)

    return version.evaluate_holdout(
        knowledge_root, task_paths, out_dir, runner=runner,
        seeds=args.holdout_eval_seeds, max_iters=args.holdout_eval_iters,
        timeout=args.timeout)


def _run_agent_for_holdout(args, task_path: str, knowledge_root: str, run_dir: str,
                           max_iters: int, timeout: float) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path,
        "--provider", args.provider, "--model", args.model,
        "--base-url", args.base_url, "--api-key-env", args.api_key_env,
        "--device", args.device, "--knowledge-root", knowledge_root,
        "--out-dir", run_dir, "--max-iters", str(max(1, int(max_iters))),
        "--timeout", str(timeout), "--profiler", "off", "--no-distill", "--quiet",
    ]
    if args.replay_file:
        cmd += ["--replay-file", args.replay_file]
    try:
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout + 240)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "speedup": 0.0}
    metas = glob.glob(os.path.join(run_dir, "*", "meta.json"))
    if not metas:
        return {"status": "no_meta", "speedup": 0.0}
    meta = read_json(sorted(metas)[-1])
    best = meta.get("best") or {}
    return {"status": best.get("status"), "speedup": float(best.get("speedup") or 0.0),
            "task": meta.get("task")}


def _approval(args, task_path: str, seed: int, variant: str):
    meta = _task_meta(task_path)
    synthetic = bool(meta.get("synthetic")) or "synth" in task_path.replace("\\", "/").split("/")
    if synthetic and not args.approve_synthetic and not _synthetic_approved(task_path):
        return False, "synthetic task awaits approval"
    if args.approve == "auto":
        return True, "auto-approved"
    if not sys.stdin.isatty():
        return False, "approval required (non-interactive)"
    answer = input("Run %s seed=%d variant=%s? [y/N] " % (
        os.path.basename(task_path), seed, variant))
    return answer.strip().lower() in ("y", "yes"), "operator decision"


def _variant_task(path: str, variant: str, out_dir: str):
    if variant in ("", "base"):
        return path
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError:
        return None
    if variant == "shape-x2":
        new = re.sub(r'("shape"\s*:\s*\[)(\d+)',
                     lambda m: m.group(1) + str(int(m.group(2)) * 2), source, count=1)
    elif variant == "dtype-bf16":
        new = source.replace('"dtype": "float32"', '"dtype": "bfloat16"', 1)
    else:
        return None
    if new == source:
        return None
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s_%s.py" % (os.path.splitext(os.path.basename(path))[0], variant))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(new)
    return out


def _run_agent(args, task_path: str, max_iters: int, cycle_runs: str) -> dict:
    before = set(glob.glob(os.path.join(cycle_runs, "*", "meta.json")))
    cmd = [
        sys.executable, "-m", "agent.loop", "--task", task_path,
        "--provider", args.provider, "--device", args.device,
        "--knowledge-root", args.knowledge_root, "--out-dir", cycle_runs,
        "--max-iters", str(max(1, int(max_iters))),
        "--target-speedup", str(args.target_speedup),
        "--timeout", str(args.timeout), "--profiler", "off",
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.replay_file:
        cmd += ["--replay-file", args.replay_file]
    if args.api_key_env:
        cmd += ["--api-key-env", args.api_key_env]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=args.timeout + 180)
        rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, stdout, stderr = 1, (exc.stdout or "") if isinstance(exc.stdout, str) else "", "timeout"
    after = set(glob.glob(os.path.join(cycle_runs, "*", "meta.json")))
    new = sorted(after - before)
    meta = read_json(new[-1]) if new else {}
    best = meta.get("best") or {}
    return {
        "rc": rc,
        "iterations": int(meta.get("iterations") or (1 if rc else 0)),
        "status": best.get("status"),
        "speedup": best.get("speedup"),
        "run_dir": os.path.dirname(new[-1]) if new else None,
        "distill": (meta.get("rsi") or {}).get("distill"),
        "stdout_tail": (stdout or "")[-800:],
        "stderr_tail": (stderr or "")[-800:],
    }


def _git_commit(knowledge_root: str, cycle_id: str) -> dict:
    if os.path.abspath(knowledge_root) != os.path.abspath(DEFAULT_KNOWLEDGE):
        return {"committed": False, "reason": "knowledge root is not the repository's rsi/knowledge"}
    add = subprocess.run(["git", "add", "rsi/knowledge"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    commit = subprocess.run(["git", "commit", "-m", "curriculum: knowledge after %s" % cycle_id],
                            cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"committed": commit.returncode == 0, "add_rc": add.returncode,
            "commit_rc": commit.returncode, "head": (rev.stdout or "").strip(),
            "tail": ((commit.stdout or "") + (commit.stderr or ""))[-400:]}


def run_cycle(args) -> dict:
    cycle_id = _now()
    cycle_dir = os.path.join(args.out, cycle_id)
    cycle_runs = os.path.join(cycle_dir, "runs")
    os.makedirs(cycle_runs, exist_ok=True)
    report = build_audit(runs_root=args.runs_root, knowledge_root=args.knowledge_root,
                         tasks_root=args.tasks_root)
    plans = build_plan(report, top=args.top)
    holdout_tasks = list(((read_json(args.holdout_config) or {}).get("tasks") or {}).keys())
    holdout_paths = [p for p in (_find_task_path(args.tasks_root, name) for name in holdout_tasks) if p]

    before = version.snapshot(args.knowledge_root, os.path.join(args.versions_root, "snapshots"))
    # Dream pass: replay verified records and demote non-reproducing ones.  Placed
    # after the snapshot so a rollback to `before` also undoes the dream writes.
    from rsi.dream import dream
    dream_report = dream(args.knowledge_root, runs_root=args.runs_root,
                         apply=True, allow_promote=False, device=args.device,
                         timeout=args.timeout)
    # Real re-evaluation, not a historical average: same holdout tasks, the same
    # runner, one score per knowledge snapshot.
    before_kb = os.path.join(cycle_dir, "kb_before")
    if os.path.isdir(before["path"]):
        shutil.copytree(before["path"], before_kb)
    score_before = _holdout_eval(args, before_kb or args.knowledge_root, holdout_paths,
                                 os.path.join(cycle_dir, "holdout_before"))

    remaining = int(args.budget_iters)
    rows = []
    skipped = []
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    for plan in plans:
        if remaining <= 0:
            break
        for seed in range(1, max(1, int(args.seeds)) + 1):
            for variant in variants:
                if remaining <= 0:
                    break
                allowed, reason = _approval(args, plan["path"], seed, variant)
                if not allowed:
                    skipped.append({"task": plan["task"], "seed": seed,
                                    "variant": variant, "reason": reason})
                    continue
                task_path = _variant_task(plan["path"], variant,
                                          os.path.join(cycle_dir, "variants"))
                if not task_path:
                    skipped.append({"task": plan["task"], "seed": seed,
                                    "variant": variant, "reason": "variant not generated"})
                    continue
                result = _run_agent(args, task_path, min(args.max_iters, remaining), cycle_runs)
                used = max(1, int(result["iterations"] or 1))
                remaining -= used
                rows.append({"task": plan["task"], "source_path": plan["path"],
                             "run_task": task_path, "seed": seed, "variant": variant,
                             "iterations": used, "budget_left": remaining, **result})
                if rows[-1]["rc"] != 0 and not result.get("status"):
                    # a broken provider/key should stop the cycle, not spin on it
                    if "api key" in (result.get("stdout_tail") or "").lower() or \
                            "api key" in (result.get("stderr_tail") or "").lower():
                        remaining = 0
                        break

    after = version.snapshot(args.knowledge_root, os.path.join(args.versions_root, "snapshots"))
    score_after = _holdout_eval(args, args.knowledge_root, holdout_paths,
                                os.path.join(cycle_dir, "holdout_after"))
    rollback = version.rollback_if_worse(
        args.knowledge_root, before, score_before, score_after, args.rollback_eps)
    final_version = version.knowledge_version(args.knowledge_root)
    git = _git_commit(args.knowledge_root, cycle_id) if args.git_commit else {
        "committed": False, "reason": "git commit disabled (use --git-commit)"}

    audit = {
        "cycle": cycle_id,
        "budget_iters": int(args.budget_iters),
        "used_iters": int(args.budget_iters) - max(0, remaining),
        "remaining_iters": max(0, remaining),
        "approve": args.approve,
        "seeds": args.seeds,
        "variants": variants,
        "plan": [{"task": p["task"], "path": p["path"], "score": p["score"],
                  "synthetic": bool(p.get("synthetic")), "holdout": p["protocol"].get("holdout")}
                 for p in plans],
        "runs": rows,
        "skipped": skipped,
        "knowledge": {
            "before": before,
            "after": after,
            "final_version": final_version,
            "rollback": rollback,
            "git": git,
            "dream": dream_report,
        },
        "holdout_score": {"before": score_before, "after": score_after,
                          "epsilon": args.rollback_eps},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(cycle_dir, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(cycle_dir, "audit.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(audit))
    audit["audit_json"] = os.path.join(cycle_dir, "audit.json")
    audit["audit_md"] = os.path.join(cycle_dir, "audit.md")
    return audit


def render_markdown(audit: dict) -> str:
    lines = [
        "# Curriculum audit %s" % audit["cycle"],
        "",
        "- budget: %d iters, used %d, remaining %d" % (
            audit["budget_iters"], audit["used_iters"], audit["remaining_iters"]),
        "- approval mode: %s | seeds=%s | variants=%s" % (
            audit["approve"], audit["seeds"], ",".join(audit["variants"])),
        "- knowledge before: %s" % (audit["knowledge"]["before"] or {}).get("id"),
        "- knowledge after:  %s" % (audit["knowledge"]["after"] or {}).get("id"),
        "- final version: %s" % audit["knowledge"]["final_version"],
        "- holdout score: %s -> %s (epsilon %.3f)" % (
            (audit["holdout_score"]["before"] or {}).get("score"),
            (audit["holdout_score"]["after"] or {}).get("score"),
            audit["holdout_score"]["epsilon"]),
        "- rollback: %s (%s)" % (
            audit["knowledge"]["rollback"].get("rolled_back"),
            audit["knowledge"]["rollback"].get("reason")),
        "",
        "| task | seed | variant | iters | status | speedup |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit["runs"]:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            row["task"], row["seed"], row["variant"], row["iterations"],
            row.get("status"), row.get("speedup")))
    if audit["skipped"]:
        lines.append("")
        lines.append("Skipped:")
        for row in audit["skipped"][:20]:
            lines.append("- %s seed=%s variant=%s: %s" % (
                row["task"], row["seed"], row["variant"], row["reason"]))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.schedule", description="Budget-limited curriculum scheduler.")
    ap.add_argument("--budget-iters", type=int, required=True)
    ap.add_argument("--approve", choices=["auto", "ask"], default="ask")
    ap.add_argument("--approve-synthetic", action="store_true",
                    help="allow generated tasks without their sidecar approval")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="base,shape-x2,dtype-bf16")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--target-speedup", type=float, default=1.0)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--replay-file", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--knowledge-root", default=DEFAULT_KNOWLEDGE)
    ap.add_argument("--runs-root", default=DEFAULT_RUNS)
    ap.add_argument("--tasks-root", default=DEFAULT_TASKS)
    ap.add_argument("--holdout-config", default=DEFAULT_HOLDOUT)
    ap.add_argument("--versions-root", default=version.DEFAULT_VERSIONS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rollback-eps", type=float, default=0.01)
    ap.add_argument("--holdout-eval-iters", type=int, default=4,
                    help="iterations per holdout re-evaluation run")
    ap.add_argument("--holdout-eval-seeds", type=int, default=1)
    ap.add_argument("--skip-holdout-eval", action="store_true",
                    help="do not re-run the holdout tasks (rollback then stays disabled)")
    ap.add_argument("--git-commit", action="store_true")
    args = ap.parse_args(argv)

    audit = run_cycle(args)
    print(json.dumps({key: audit[key] for key in (
        "cycle", "budget_iters", "used_iters", "remaining_iters",
        "holdout_score", "knowledge", "audit_json", "audit_md")},
        ensure_ascii=False, indent=2))
    return 0 if audit["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
