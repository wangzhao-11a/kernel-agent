"""Dreaming pass: replay recorded solutions and demote memories that do not reproduce.

Design constraint (M0): timing and guards still run inside the same process as
the candidate code, so a replay can reproduce a *fake* result.  Therefore this
pass **demotes by default and never promotes**.  Only an explicit
`--allow-promote` *and* a harness result that reports `parent_timed: true` may
refresh `last_verified`.

    python -m rsi.dream --root rsi/knowledge --runs-root runs/agent --apply
    python -m rsi.dream --root rsi/knowledge --runs-root runs/agent --allow-promote
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi.memory import MemoryStore, utc_now  # noqa: E402
from rsi.store import StrategyStore  # noqa: E402
from rsi.trajectory import read_json, replay_trajectory  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_attempts(runs_root: str):
    """Yield (run_dir, attempts list) for every run under runs_root."""
    if not os.path.isdir(runs_root):
        return
    for name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        path = os.path.join(run_dir, "attempts.jsonl")
        if not os.path.isfile(path):
            continue
        attempts = []
        for line in open(path, encoding="utf-8"):
            if line.strip():
                attempts.append(json.loads(line))
        yield run_dir, attempts


def _replay_one(run_dir: str, device: str, timeout: float):
    """Replay one run; return (kind, detail).

    kind: "reproduced" | "not_reproduced" | "env_error"
    """
    try:
        replay = replay_trajectory(run_dir, device=device, timeout=timeout)
    except Exception as exc:
        return "env_error", "%s: %s" % (type(exc).__name__, exc)
    n = replay.get("n", 0)
    matches = replay.get("matches", 0)
    if n == 0:
        return "env_error", "trajectory has no replayable attempts"
    if matches != n:
        return "not_reproduced", replay
    # Status alone is not enough: a replay that lands outside the recorded CI is
    # also a memory problem, not a reproducibility success.
    outside = _outside_recorded_ci(run_dir, replay)
    if outside:
        return "not_reproduced", replay
    return "reproduced", replay


def _outside_recorded_ci(run_dir: str, replay: dict) -> bool:
    """True when any replayed speedup falls outside its recorded speedup_ci."""
    path = os.path.join(run_dir, "attempts.jsonl")
    if not os.path.isfile(path):
        return False
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by_iter = {row.get("iter"): row for row in rows}
    for result in replay.get("results") or []:
        attempt = by_iter.get(result.get("iter"))
        if not attempt:
            continue
        ci = attempt.get("speedup_ci")
        if not ci or len(ci) != 2:
            continue
        value = result.get("replay_speedup")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value < float(ci[0]) or value > float(ci[1]):
            return True
    return False


def _demote(row: dict, reason: str, replay_run_id: str, replay_data) -> dict:
    """Write a demotion with provenance.  Consecutive failures -> refuted."""
    prior = int(row.get("demote_count") or 0) + 1
    row["confidence"] = "refuted" if prior >= 2 else "candidate"
    row["demote_count"] = prior
    row["dispute_reason"] = reason
    row["replayed_at"] = _now()
    row["replay_run_id"] = replay_run_id
    row["replay_data"] = replay_data
    row["last_replayed"] = _now()
    row["updated"] = utc_now()
    return row


def _keep(row: dict, replay_run_id: str, allow_promote: bool, parent_timed: bool) -> dict:
    """Reproduced: only record that we checked.  Never promote by default."""
    row["last_replayed"] = _now()
    row["replayed_at"] = _now()
    row["replay_run_id"] = replay_run_id
    if allow_promote and parent_timed:
        row["last_verified"] = utc_now()
    row["updated"] = utc_now()
    return row


def _note_outcomes(notes: list, runs_root: str):
    """Count independent successful/failed runs that mention each note id."""
    outcomes = {note["id"]: {"success": set(), "failure": set()} for note in notes}
    if not notes:
        return outcomes
    for run_dir, attempts in _iter_attempts(runs_root):
        run_id = os.path.basename(run_dir)
        text = json.dumps(attempts, ensure_ascii=False)
        status = "success" if any(a.get("status") in ("pass", "pass_unstable") for a in attempts) else "failure"
        for note in notes:
            if note["id"] in text:
                outcomes[note["id"]][status].add(run_id)
    return outcomes


def _review_notes(mem: MemoryStore, runs_root: str, apply: bool) -> list:
    """Promote notes seen in >=2 successful runs; refute notes only seen in failures."""
    outcomes = _note_outcomes(mem.notes, runs_root)
    report = []
    for note in mem.notes:
        succ = outcomes[note["id"]]["success"]
        fail = outcomes[note["id"]]["failure"]
        action = None
        if len(succ) >= 2:
            action = "promote"
            note["confidence"] = "candidate"
            note["status"] = "active"
            if apply:
                mem.add_fact({
                    "statement": note["statement"],
                    "scope": note.get("scope") or {},
                    "confidence": "candidate",
                    "status": "active",
                    "authority": "experiment",
                    "type": "process_note",
                })
        elif len(fail) >= 3 and not succ:
            action = "refute"
            note["confidence"] = "refuted"
        if action:
            note["updated"] = utc_now()
            report.append({"id": note["id"], "action": action,
                           "success_runs": sorted(succ), "failure_runs": sorted(fail)})
    return report


def dream(root: str, runs_root: str = None, apply: bool = False,
          allow_promote: bool = False, device: str = "cpu", timeout: float = 600.0,
          limit: int = 0) -> dict:
    """Replay verified strategies/skills and demote those that do not reproduce."""
    store = StrategyStore(root)
    mem = MemoryStore(root)
    runs_root = runs_root or os.path.join(ROOT, "runs", "agent")
    report = {
        "root": os.path.abspath(root),
        "runs_root": os.path.abspath(runs_root),
        "apply": bool(apply),
        "allow_promote": bool(allow_promote),
        "checked": [],
        "notes": [],
        "summary": {"reproduced": 0, "not_reproduced": 0, "env_error": 0},
    }

    candidates = []
    for sid, row in store.strategies.items():
        if row.get("confidence") == "verified":
            candidates.append(("strategy", sid, row))
    for row in mem.skills:
        if row.get("confidence") == "verified":
            candidates.append(("skill", row.get("id"), row))
    candidates.sort(key=lambda item: (item[2].get("last_replayed") or item[2].get("updated") or ""))
    if limit:
        candidates = candidates[:max(0, int(limit))]

    for kind, rid, row in candidates:
        run_id = str(row.get("run_id") or "")
        # A verified row may carry several evidence run ids; replay the latest.
        if not run_id:
            evidence = row.get("evidence") or []
            if evidence:
                last = evidence[-1]
                run_id = str(last.get("run_id") or last.get("trajectory") or "")
        run_dir = run_id if os.path.isabs(run_id) else os.path.join(runs_root, run_id)
        if not os.path.isdir(run_dir):
            report["checked"].append({
                "kind": kind, "id": rid, "outcome": "env_error",
                "detail": "run directory not found: %s" % run_dir,
            })
            report["summary"]["env_error"] += 1
            continue
        outcome, detail = _replay_one(run_dir, device, timeout)
        entry = {"kind": kind, "id": rid, "run_dir": run_dir, "outcome": outcome,
                 "detail": detail}
        if outcome == "reproduced":
            report["summary"]["reproduced"] += 1
            if apply:
                _keep(row, run_id, allow_promote, parent_timed=bool(detail.get("parent_timed")))
        elif outcome == "not_reproduced":
            report["summary"]["not_reproduced"] += 1
            if apply:
                speedups = []
                for res in (detail.get("results") or []):
                    value = res.get("replay_speedup")
                    if value is not None:
                        speedups.append(value)
                reason = ("replay did not reproduce: run %s, replayed speedups %s"
                          % (run_id, speedups))
                _demote(row, reason, run_id, detail)
        else:
            report["summary"]["env_error"] += 1
            if apply:
                _demote(row, "replay environment error: %s" % detail, run_id, detail)
        report["checked"].append(entry)

    if apply:
        store.save()
        mem.save()

    # Notes: promote/refute based on independent run outcomes.
    report["notes"] = _review_notes(mem, runs_root, apply)
    if apply:
        mem.save()
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.dream", description="Replay and demote non-reproducing memories.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--apply", action="store_true", help="write demotions/promotions (default: report only)")
    ap.add_argument("--allow-promote", action="store_true",
                    help="allow last_verified refresh ONLY when the harness reports parent_timed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0, help="max verified records to replay")
    args = ap.parse_args(argv)

    report = dream(args.root, runs_root=args.runs_root or None, apply=args.apply,
                   allow_promote=args.allow_promote, device=args.device,
                   timeout=args.timeout, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
