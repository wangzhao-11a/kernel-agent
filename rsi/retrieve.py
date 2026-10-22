"""Retrieve condition-matched strategies from the pool and render tuning priors.

    python -m rsi.retrieve --task tasks/level2/mla_attention.py --hardware ascend-910b

Ranking = condition match (tags / op class / hardware / dtype) + quality
(success rate, measured EMA speedup, calibration) + an uncertainty bonus so
rarely-applied strategies still get explored.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.tasks import load_task  # noqa: E402
from rsi.features import hardware_from_device, task_features  # noqa: E402
from rsi.store import DEFAULT_ROOT, StrategyStore  # noqa: E402


def _quality(strategy: dict) -> float:
    stats = strategy.get("stats") or {}
    applied = int(stats.get("applied") or 0)
    succeeded = int(stats.get("succeeded") or 0)
    rate = (succeeded / applied) if applied else 0.4
    ema = float(stats.get("ema_speedup") or 1.0)
    base = rate * (1.0 + 0.5 * (ema - 1.0))
    explore = 0.25 / math.sqrt(1.0 + applied)
    confidence = {"seed": 0.15, "verified": 0.30, "candidate": 0.05}.get(strategy.get("confidence"), 0.1)
    return base + explore + confidence


def _match(strategy: dict, feats: dict, hardware: str) -> float:
    tags = set(strategy.get("tags") or [])
    ftags = set(feats.get("tags") or [])
    score = len(tags & ftags) / max(1, len(ftags))
    cond = strategy.get("conditions") or {}
    op_class = feats.get("op_class") or ""
    classes = cond.get("op_class") or []
    if classes and op_class:
        score += 0.4 if op_class in classes else -0.5
    hw = cond.get("hardware") or []
    if hw:
        score += 0.3 if hardware in hw else -0.6
    dtypes = cond.get("dtypes") or []
    if dtypes:
        score += 0.2 if (feats.get("dtypes") or [""])[0] in dtypes else -0.3
    return score


def retrieve(store: StrategyStore, feats: dict, hardware: str, top_k: int = 5,
             min_match: float = 0.3, exclude=()) -> list:
    scored = []
    for sid, strategy in store.strategies.items():
        if sid in exclude:
            continue
        match = _match(strategy, feats, hardware)
        if match < min_match:
            continue
        scored.append((strategy, match + 0.6 * _quality(strategy), match))
    scored.sort(key=lambda item: -item[1])
    return scored[:top_k]


def combos_for(store: StrategyStore, member_ids) -> list:
    members = set(member_ids)
    return [c for c in store.combos.values() if set(c.get("members") or []) <= members]


def _calibration_note(matched: list) -> str:
    """Tell the agent how far off its own predictions have been on matched conditions."""
    errors, biases = [], []
    for strategy, _score, _match in matched:
        stats = strategy.get("stats") or {}
        if stats.get("calibration_mae") is None:
            continue
        errors.append(float(stats["calibration_mae"]))
        for evidence in strategy.get("evidence") or []:
            expected = evidence.get("expected_speedup")
            measured = evidence.get("measured_speedup")
            if expected and measured:
                biases.append(float(expected) - float(measured))
    if not errors:
        return ""
    mean_error = sum(errors) / len(errors)
    if mean_error < 0.3:
        return ""
    bias = sum(biases) / len(biases) if biases else 0.0
    direction = "overshooting" if bias > 0 else "undershooting"
    return ("Calibration note: on matched conditions your declared expected_speedup has historically been off "
            "by ~%.2fx on average (%s by ~%.2fx). Predict conservatively and verify with the measurement."
            % (mean_error, direction, abs(bias)))


def priors_block(feats: dict, hardware: str, matched: list, combos: list, attempted=()) -> str:
    lines = [
        "## Tuning priors (retrieved from the strategy pool by condition match)",
        "matching context: hardware=%s | op_class=%s | dtypes=%s | tags=%s" % (
            hardware or "-", feats.get("op_class") or "-",
            ",".join(feats.get("dtypes") or []), ",".join(feats.get("tags") or [])),
    ]
    if not matched:
        lines.append("(nothing in the pool matches this task yet — establish working practices and declare them so they can be reused)")
        return "\n".join(lines)
    attempted = set(attempted)
    for strategy, _score, _match in matched:
        stats = strategy.get("stats") or {}
        applied = int(stats.get("applied") or 0)
        succeeded = int(stats.get("succeeded") or 0)
        ema = stats.get("ema_speedup")
        calibration = stats.get("calibration_mae")
        marks = []
        if applied:
            marks.append("history %d/%d ok" % (succeeded, applied))
        if ema:
            marks.append("measured EMA %.2fx" % ema)
        if calibration is not None:
            marks.append("expectation error %.2f" % calibration)
        cond = strategy.get("conditions") or {}
        cond_bits = []
        if cond.get("hardware"):
            cond_bits.append("hw=%s" % ",".join(cond["hardware"]))
        if cond.get("dtypes"):
            cond_bits.append("dtype=%s" % ",".join(cond["dtypes"]))
        if cond.get("refined_note"):
            cond_bits.append("refined: %s" % cond["refined_note"])
        lines.append("- [%s] %s" % (strategy["id"], strategy["statement"]))
        lines.append("    expected ~%.2fx | %s | conditions: %s | confidence: %s%s" % (
            float((strategy.get("expected") or {}).get("speedup") or 1.0),
            "; ".join(marks) or "no history yet",
            ", ".join(cond_bits) or "any",
            strategy.get("confidence"),
            " | already attempted this run" if strategy["id"] in attempted else ""))
    if combos:
        lines.append("Combination rules (practices that historically co-occurred with gains):")
        for combo in combos[:3]:
            stats = combo.get("stats") or {}
            lines.append("- [%s] %s -> best measured %.2fx (applied %sx)" % (
                combo["id"], " + ".join(combo.get("members") or []),
                float(stats.get("best_speedup") or 0.0), stats.get("applied")))
    lines.append("In write_solution include \"strategies\" (names or ids you applied) and "
                 "\"expected_speedup\" (your prediction): they update this pool for later tasks.")
    note = _calibration_note(matched)
    if note:
        lines.append(note)
    return "\n".join(lines)


def failure_hints(matched: list, attempted, top: int = 2) -> str:
    attempted = set(attempted)
    remaining = [strategy for strategy, _s, _m in matched if strategy["id"] not in attempted]
    if not remaining:
        return ""
    bits = ["[%s] %s" % (s["id"], s["statement"]) for s in remaining[:top]]
    return "Strategy-pool hints (condition-matched, not yet tried on this task): " + " | ".join(bits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.retrieve", description="Show the tuning priors retrieved for a task.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    task = load_task(args.task)
    feats = task_features(task)
    hardware = args.hardware or feats.get("hardware") or hardware_from_device(args.device)
    store = StrategyStore(args.root)
    matched = retrieve(store, feats, hardware, top_k=args.top)
    combos = combos_for(store, [s["id"] for s, _s, _m in matched])
    if args.json:
        print(json.dumps({
            "task": feats, "hardware": hardware,
            "matched": [{"id": s["id"], "key": s.get("key"), "score": round(score, 3),
                         "statement": s["statement"], "stats": s.get("stats")} for s, score, _m in matched],
            "combos": [{"id": c["id"], "members": c["members"], "stats": c.get("stats")} for c in combos],
        }, ensure_ascii=False, indent=2))
        return 0
    print(priors_block(feats, hardware, matched, combos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())