"""Self-evolution outer loop: capability audit + curriculum planning.

The inner loop (`agent.loop`) improves one solution. The strategy pool learns what
worked. The missing outer loop is the one that decides **what to practice next** and
**whether the evidence is valid enough to learn from**.

This module is deliberately read-only. It gives the scheduler (or a human) two things:

    audit   capability matrix + evidence provenance + contamination warnings
    plan    ranked practice plan (task x variants x budget x success criteria)

The rule that matters most: a CPU run labelled `ascend-910b` is a mechanics test,
not performance evidence. The audit makes that distinction explicit before the
strategy pool absorbs another unverifiable data point.

    python -m rsi.evolve audit
    python -m rsi.evolve audit --fail-on-contamination
    python -m rsi.evolve plan --top 6
    python -m rsi.evolve plan --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from datetime import datetime, timezone

from .provenance import device_class, provenance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_ROOT = os.path.join(ROOT, "runs", "agent")
DEFAULT_TASKS_ROOT = os.path.join(ROOT, "tasks")
DEFAULT_KNOWLEDGE_ROOT = os.path.join(ROOT, "rsi", "knowledge")

# op families where a kernel form is the measured winner (see rsi/forms.py)
KERNEL_FIRST = {"attention", "gemm", "reduction"}


# --------------------------------------------------------------------------- io
def read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def read_jsonl(path: str) -> list:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_days(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        value = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return max(0.0, (_now() - value).total_seconds() / 86400.0)
    except ValueError:
        return None


# --------------------------------------------------------------------------- runs
def load_run(run_dir: str) -> dict:
    meta = read_json(os.path.join(run_dir, "meta.json"))
    attempts = read_jsonl(os.path.join(run_dir, "attempts.jsonl"))
    passes = [a for a in attempts if a.get("status") == "pass"]
    best = meta.get("best") or {}
    best_speedup = best.get("speedup")
    if best_speedup is None and passes:
        best_speedup = max(float(a.get("speedup") or 0.0) for a in passes)
    speedups = [float(a.get("speedup") or 0.0) for a in passes]
    prov = provenance(meta)
    strategies = set()
    for attempt in attempts:
        for key in ("strategies_used", "strategies_declared"):
            strategies.update(str(x) for x in (attempt.get(key) or []) if str(x).strip())
    return {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "run_dir": os.path.abspath(run_dir),
        "task": meta.get("task") or (attempts[0].get("task") if attempts else ""),
        "device": meta.get("device") or "",
        "hardware": meta.get("hardware") or prov.get("inferred_hardware") or "",
        "provider": meta.get("provider") or "",
        "model": meta.get("model") or "",
        "timestamp": meta.get("timestamp") or "",
        "attempts": len(attempts),
        "passes": len(passes),
        "pass_rate": (len(passes) / len(attempts)) if attempts else 0.0,
        "best_speedup": float(best_speedup) if best_speedup is not None else None,
        "median_pass_speedup": statistics.median(speedups) if speedups else None,
        "features": meta.get("features") or {},
        "strategies": sorted(strategies),
        "provenance": prov,
        "valid_evidence": bool(prov.get("valid")) and bool(passes),
    }


def load_runs(runs_root: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_root, "*"))):
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json")):
            rows.append(load_run(path))
    return rows


# --------------------------------------------------------------------------- tasks
def load_tasks(tasks_root: str) -> list:
    from harness.tasks import load_task
    from rsi.features import task_features

    rows, errors = [], []
    for path in sorted(glob.glob(os.path.join(tasks_root, "**", "*.py"), recursive=True)):
        if "__pycache__" in path or os.path.basename(path).startswith("_"):
            continue
        try:
            task = load_task(path)
            feats = task_features(task)
            rows.append({
                "path": path,
                "name": task.name,
                "kind": "python",
                "level": task.level,
                "expected_speedup": float(task.expected_speedup or 1.0),
                "features": feats,
                "op_class": feats.get("op_class") or "",
                "dtypes": feats.get("dtypes") or [],
                "tags": feats.get("tags") or [],
            })
        except Exception as exc:  # a broken task should not break the audit
            errors.append({"path": path, "error": str(exc)})

    return rows, errors


# --------------------------------------------------------------------------- audit
def build_audit(runs_root: str = DEFAULT_RUNS_ROOT,
                knowledge_root: str = DEFAULT_KNOWLEDGE_ROOT,
                tasks_root: str = DEFAULT_TASKS_ROOT) -> dict:
    runs = load_runs(runs_root)
    tasks, task_errors = load_tasks(tasks_root)
    strategies = read_jsonl(os.path.join(knowledge_root, "strategies.jsonl"))
    combos = read_jsonl(os.path.join(knowledge_root, "combos.jsonl"))
    exemplars = read_jsonl(os.path.join(knowledge_root, "exemplars.jsonl"))
    try:
        from rsi.memory import MemoryStore
        memory = MemoryStore(knowledge_root).audit()
    except Exception as exc:
        memory = {"error": str(exc)}
    run_by_id = {row["run_id"]: row for row in runs}

    # ---------------------------------------------------------------- capability
    capability = {}
    for run in runs:
        key = (run["task"], run["hardware"] or "unknown", run["device"] or "unknown")
        row = capability.setdefault(key, {
            "task": key[0], "hardware": key[1], "device": key[2],
            "runs": 0, "valid_runs": 0, "attempts": 0, "passes": 0,
            "best_speedup": None, "last_run": "",
        })
        row["runs"] += 1
        row["attempts"] += run["attempts"]
        row["passes"] += run["passes"]
        row["valid_runs"] += 1 if run["valid_evidence"] else 0
        if run["best_speedup"] is not None:
            row["best_speedup"] = max(row["best_speedup"] or 0.0, run["best_speedup"])
        row["last_run"] = max(row["last_run"], run["timestamp"])
    capability_rows = []
    for row in capability.values():
        row["pass_rate"] = (row["passes"] / row["attempts"]) if row["attempts"] else 0.0
        capability_rows.append(row)
    capability_rows.sort(key=lambda r: (r["task"], r["hardware"], r["device"]))

    by_feature = {}
    for run in runs:
        feats = run.get("features") or {}
        key = (feats.get("op_class") or "unknown", run["hardware"] or "unknown")
        row = by_feature.setdefault(key, {
            "op_class": key[0], "hardware": key[1], "tasks": set(),
            "runs": 0, "valid_runs": 0, "best_speedup": None,
        })
        row["tasks"].add(run["task"])
        row["runs"] += 1
        row["valid_runs"] += 1 if run["valid_evidence"] else 0
        if run["best_speedup"] is not None:
            row["best_speedup"] = max(row["best_speedup"] or 0.0, run["best_speedup"])
    feature_rows = []
    for row in by_feature.values():
        row["tasks"] = sorted(row["tasks"])
        feature_rows.append(row)
    feature_rows.sort(key=lambda r: (r["op_class"], r["hardware"]))

    # ---------------------------------------------------------------- strategies
    strategy_rows = []
    for card in strategies:
        evidence = card.get("evidence") or []
        valid = invalid = 0
        for item in evidence:
            path = str(item.get("trajectory") or "").replace("\\", "/")
            run_id = os.path.basename(os.path.dirname(path)) if path else ""
            run = run_by_id.get(run_id)
            if run and run["valid_evidence"]:
                valid += 1
            elif run:
                invalid += 1
            else:
                # no matching run: trust the card's own hardware label only if it is not CPU
                invalid += 1 if str(item.get("hardware") or "").lower() in ("ascend-910b", "") else 0
        strategy_rows.append({
            "id": card.get("id"),
            "key": card.get("key") or "",
            "statement": card.get("statement") or "",
            "confidence": card.get("confidence") or "",
            "applied": int((card.get("stats") or {}).get("applied") or 0),
            "evidence": len(evidence),
            "valid_evidence": valid,
            "invalid_evidence": invalid,
        })
    strategy_rows.sort(key=lambda r: (-r["invalid_evidence"], -r["evidence"], r["id"] or ""))

    # ---------------------------------------------------------------- issues
    invalid = [run for run in runs if not run["provenance"].get("valid")]
    contaminated = [
        run for run in invalid
        if str(run["provenance"].get("claimed_hardware") or "").lower()
        not in ("", "cpu", "unknown")
    ]
    tasks_with_runs = {run["task"] for run in runs}
    tasks_with_valid = {run["task"] for run in runs if run["valid_evidence"]}
    never_run = sorted({task["name"] for task in tasks} - tasks_with_runs)
    no_valid = sorted(tasks_with_runs - tasks_with_valid)

    return {
        "generated_at": _now().isoformat(timespec="seconds"),
        "runs_root": os.path.abspath(runs_root),
        "tasks_root": os.path.abspath(tasks_root),
        "summary": {
            "runs": len(runs),
            "valid_runs": sum(1 for r in runs if r["valid_evidence"]),
            "mechanics_only_runs": sum(1 for r in runs if not r["provenance"].get("valid")),
            "tasks_in_suite": len(tasks),
            "tasks_never_run": len(never_run),
            "tasks_without_valid_evidence": len(no_valid),
            "strategies": len(strategies),
            "strategies_with_evidence": sum(1 for s in strategy_rows if s["evidence"]),
            "strategies_with_valid_evidence": sum(1 for s in strategy_rows if s["valid_evidence"]),
            "combos": len(combos),
            "exemplars": len(exemplars),
            "contaminated_runs": len(contaminated),
            "invalid_runs": len(invalid),
            "facts": memory.get("facts", 0),
            "failures_open": memory.get("failures_open", 0),
            "skills_verified": memory.get("skills_verified", 0),
            "memory_stale": memory.get("stale", 0),
            "memory_disputed": memory.get("disputed_or_refuted", 0),
            "memory_conflicts": memory.get("conflicts", 0),
        },
        "capability": capability_rows,
        "by_feature": feature_rows,
        "strategies": strategy_rows,
        "memory": memory,
        "issues": {
            "contaminated_runs": [
                {"run_id": r["run_id"], "task": r["task"], "device": r["device"],
                 "claimed_hardware": r["provenance"].get("claimed_hardware"),
                 "reason": r["provenance"].get("reason")}
                for r in contaminated
            ],
            "invalid_runs": [
                {"run_id": r["run_id"], "task": r["task"], "device": r["device"],
                 "claimed_hardware": r["provenance"].get("claimed_hardware"),
                 "reason": r["provenance"].get("reason")}
                for r in invalid
            ],
            "tasks_never_run": never_run,
            "tasks_without_valid_evidence": no_valid,
            "task_load_errors": task_errors,
        },
        "runs": runs,
        "tasks": tasks,
    }


def print_audit(report: dict) -> None:
    s = report["summary"]
    print("# Kernel-Agent capability audit")
    print()
    print("runs: %d | valid performance runs: %d | mechanics-only: %d | mislabelled: %d" % (
        s["runs"], s["valid_runs"], s["mechanics_only_runs"], s["contaminated_runs"]))
    print("tasks in suite: %d | never run: %d | runs without valid evidence: %d" % (
        s["tasks_in_suite"], s["tasks_never_run"], s["tasks_without_valid_evidence"]))
    print("strategies: %d | with evidence: %d | with valid evidence: %d" % (
        s["strategies"], s["strategies_with_evidence"], s["strategies_with_valid_evidence"]))
    print("exemplars: %d | combos: %d" % (s["exemplars"], s["combos"]))
    print("memory: facts %d | open failures %d | verified skills %d | stale %d | disputed %d | conflicts %d" % (
        s.get("facts", 0), s.get("failures_open", 0), s.get("skills_verified", 0),
        s.get("memory_stale", 0), s.get("memory_disputed", 0), s.get("memory_conflicts", 0)))
    print()

    if report["issues"]["invalid_runs"]:
        print("## Invalid evidence (mechanics-only or mislabelled)")
        for row in report["issues"]["invalid_runs"]:
            label = "MISLABELLED" if str(row["claimed_hardware"] or "").lower() not in ("", "cpu", "unknown") else "MECHANICS"
            print("- [%s] %s [%s] device=%s claimed=%s: %s" % (
                label, row["run_id"], row["task"], row["device"],
                row["claimed_hardware"], row["reason"]))
        print()

    print("## Capability matrix")
    print("| task | hardware | device | runs | valid | pass rate | best | last run |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for row in report["capability"]:
        print("| %s | %s | %s | %d | %d | %.2f | %s | %s |" % (
            row["task"], row["hardware"], row["device"], row["runs"], row["valid_runs"],
            row["pass_rate"], _fmt(row["best_speedup"], "%.3fx"), row["last_run"][:19]))
    print()

    if report["by_feature"]:
        print("## By operator family")
        print("| op_class | hardware | tasks | runs | valid | best |")
        print("|---|---|---|---:|---:|---:|")
        for row in report["by_feature"]:
            print("| %s | %s | %s | %d | %d | %s |" % (
                row["op_class"], row["hardware"], ", ".join(row["tasks"]),
                row["runs"], row["valid_runs"], _fmt(row["best_speedup"], "%.3fx")))
        print()

    if report["strategies"]:
        print("## Strategy evidence")
        print("| id | key | applied | evidence | valid | invalid | confidence |")
        print("|---|---|---:|---:|---:|---:|---|")
        for row in report["strategies"]:
            if not row["evidence"] and not row["applied"]:
                continue
            print("| %s | %s | %d | %d | %d | %d | %s |" % (
                row["id"], row["key"], row["applied"], row["evidence"],
                row["valid_evidence"], row["invalid_evidence"], row["confidence"]))
        print()

    memory = report.get("memory") or {}
    if memory and not memory.get("error"):
        print("## Operator memory")
        print("facts %d (active %d) | failures %d (open %d) | skills %d (verified %d) | stale %d | disputed %d" % (
            memory.get("facts", 0), memory.get("facts_active", 0),
            memory.get("failures", 0), memory.get("failures_open", 0),
            memory.get("skills", 0), memory.get("skills_verified", 0),
            memory.get("stale", 0), memory.get("disputed_or_refuted", 0)))
        if memory.get("open_failure_ids"):
            print("open failures: %s" % ", ".join(memory["open_failure_ids"]))
        print()

    issues = report["issues"]
    if issues["tasks_never_run"]:
        print("## Never practiced")
        print(", ".join(issues["tasks_never_run"]))
        print()
    if issues["tasks_without_valid_evidence"]:
        print("## Has runs but no valid performance evidence")
        print(", ".join(issues["tasks_without_valid_evidence"]))
        print()
    if issues["task_load_errors"]:
        print("## Task load errors")
        for row in issues["task_load_errors"]:
            print("- %s: %s" % (row["path"], row["error"]))


def _fmt(value, fmt="%.2f"):
    return "-" if value is None else fmt % value


# --------------------------------------------------------------------------- plan
def build_plan(report: dict, top: int = 6, kind: str = "") -> list:
    runs = report["runs"]
    all_tasks = report["tasks"]
    tasks = [t for t in all_tasks
             if not kind or (t.get("kind") or "python") == kind]
    exemplar_tasks = set()
    exemplar_path = os.path.join(DEFAULT_KNOWLEDGE_ROOT, "exemplars.jsonl")
    for row in read_jsonl(exemplar_path):
        exemplar_tasks.add((row.get("task"), row.get("hardware")))

    runs_by_task = {}
    for run in runs:
        runs_by_task.setdefault(run["task"], []).append(run)

    op_class_counts = {}
    for task in all_tasks:
        op_class = task.get("op_class") or ""
        op_class_counts[op_class] = op_class_counts.get(op_class, 0) + 1

    plans = []
    for task in tasks:
        task_runs = runs_by_task.get(task["name"], [])
        valid_runs = [r for r in task_runs if r["valid_evidence"]]
        best = max((r["best_speedup"] or 0.0) for r in valid_runs) if valid_runs else None
        attempts = sum(r["attempts"] for r in task_runs)
        passes = sum(r["passes"] for r in task_runs)
        pass_rate = (passes / attempts) if attempts else 0.0
        last = max((r["timestamp"] for r in task_runs), default="")
        age = _age_days(last)

        score, reasons = 0.0, []
        if not valid_runs:
            score += 5.0
            reasons.append("no valid performance evidence")
        if best is not None and best < 1.10:
            score += 3.0 * (1.10 - best) / 0.10
            reasons.append("best %.3fx below 1.10x" % best)
        if attempts and pass_rate < 0.60:
            score += 2.0 * (0.60 - pass_rate) / 0.60
            reasons.append("pass rate %.0f%% below 60%%" % (pass_rate * 100))
        if not task_runs:
            score += 1.0
            reasons.append("never practiced")
        elif age is None or age > 7:
            score += 1.0
            reasons.append("stale (>7 days)")
        if (task["name"], "ascend-910b") not in exemplar_tasks:
            score += 1.0
            reasons.append("no 910B exemplar")
        transfer = min(1.5, op_class_counts.get(task.get("op_class") or "", 0) / 4.0)
        score += transfer
        if transfer > 0:
            reasons.append("family %s shared by %d tasks" % (
                task.get("op_class") or "?", op_class_counts.get(task.get("op_class") or "", 0)))
        score -= 0.5 * max(0, int(task.get("level") or 1) - 1)
        score += min(1.0, max(0.0, float(task.get("expected_speedup") or 1.0) - 1.0))

        op_class = task.get("op_class") or ""
        kind = task.get("kind") or "python"
        measured_form = ""
        try:
            from .forms import form_for
            measured_form = form_for(task["features"])
        except Exception:
            pass
        if measured_form == "triton_kernel":
            forms = ["triton"]
        elif measured_form == "torch_fused_first":
            forms = ["torch_fused"]
        else:
            forms = ["triton"] if op_class in KERNEL_FIRST else ["torch_fused"]
        variants = ["base", "shape-x2", "dtype-bf16"]
        holdout = "shape-x0.5"
        success = {
            "base_pass_rate": 0.80,
            "variant_pass_rate": 0.60,
            "best_speedup": max(1.10, float(task.get("expected_speedup") or 1.0) * 0.8),
            "replay_match": 1.0,
        }

        plans.append({
            "task": task["name"],
            "path": task["path"],
            "kind": kind,
            "level": task["level"],
            "op_class": op_class,
            "dtypes": task["dtypes"],
            "score": round(score, 3),
            "reasons": reasons,
            "evidence": {
                "runs": len(task_runs),
                "valid_runs": len(valid_runs),
                "attempts": attempts,
                "pass_rate": round(pass_rate, 3),
                "best_speedup": best,
                "last_run": last,
            },
            "protocol": {
                "seeds": 3,
                "forms": forms,
                "variants": variants,
                "holdout": holdout,
                "max_iters": 8,
                "success": success,
            },
        })
    plans.sort(key=lambda row: -row["score"])
    return plans[:max(1, int(top))]


def print_plan(plans: list) -> None:
    print("# Practice plan (outer-loop curriculum)")
    print()
    print("Ranking = evidence gap + weakness + family transfer + staleness - cost.")
    print()
    for index, row in enumerate(plans, 1):
        print("## %d. `%s` [%s] (%s, level %s) — score %.2f" % (
            index, row["task"], row.get("kind") or "python",
            row["op_class"] or "unknown", row["level"], row["score"]))
        print("- path: `%s`" % row["path"])
        print("- why: %s" % "; ".join(row["reasons"]))
        print("- evidence: %d runs / %d valid | pass %.0f%% | best %s | last %s" % (
            row["evidence"]["runs"], row["evidence"]["valid_runs"],
            row["evidence"]["pass_rate"] * 100, _fmt(row["evidence"]["best_speedup"], "%.3fx"),
            (row["evidence"]["last_run"] or "-")[:19]))
        protocol = row["protocol"]
        print("- protocol: %d seeds x %s x %s | %d iters/run | holdout `%s`" % (
            protocol["seeds"], ",".join(protocol["forms"]), ",".join(protocol["variants"]),
            protocol["max_iters"], protocol["holdout"]))
        success = protocol["success"]
        if success.get("official_cases"):
            print("- success: official cases %d/%d, build PASS, best >= %.2fx, replay %.0f%%" % (
                success["official_cases"], success["official_cases"],
                success["best_speedup"], success["replay_match"] * 100))
        else:
            print("- success: base pass >= %.0f%%, variant pass >= %.0f%%, best >= %.2fx, replay %.0f%%" % (
                success["base_pass_rate"] * 100, success["variant_pass_rate"] * 100,
                success["best_speedup"], success["replay_match"] * 100))
        print()


# --------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.evolve", description="Capability audit and practice planning.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("audit", "plan"):
        parser = sub.add_parser(name)
        parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
        parser.add_argument("--tasks-root", default=DEFAULT_TASKS_ROOT)
        parser.add_argument("--knowledge-root", default=DEFAULT_KNOWLEDGE_ROOT)
        parser.add_argument("--json", action="store_true")
        if name == "audit":
            parser.add_argument("--fail-on-contamination", action="store_true")
        else:
            parser.add_argument("--top", type=int, default=6)
            parser.add_argument("--kind", default="",
                                help="filter tasks by kind (currently: python)")

    args = ap.parse_args(argv)
    report = build_audit(args.runs_root, args.knowledge_root, args.tasks_root)

    if args.cmd == "audit":
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print_audit(report)
        if args.fail_on_contamination and report["summary"]["contaminated_runs"]:
            return 1
        return 0

    plans = build_plan(report, top=args.top, kind=getattr(args, "kind", ""))
    if args.json:
        print(json.dumps({"summary": report["summary"], "plans": plans},
                         ensure_ascii=False, indent=2, default=str))
    else:
        print_plan(plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())