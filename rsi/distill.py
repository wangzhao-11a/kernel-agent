"""Distill trajectories into strategy-pool updates (the RSI update step).

    python -m rsi.distill --run-dir runs/agent/<run> [--root rsi/knowledge] [--verify --device npu]

For every (not yet distilled) attempt it:
  1. attributes the practices used — agent-declared names first, source keywords as fallback;
  2. links practice <-> measured effect <-> declared expectation (calibration error);
  3. updates stats, narrows applicability conditions, and records combination rules.
Already-distilled attempts are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rsi.exemplars import record_exemplar  # noqa: E402
from rsi.provenance import provenance  # noqa: E402
from rsi.store import DEFAULT_ROOT, StrategyStore  # noqa: E402
from rsi.trajectory import build_trajectory, read_json, replay_trajectory, write_jsonl  # noqa: E402


def _norm(text) -> str:
    return " ".join(str(text or "").lower().split())


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", _norm(text)))


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _token_match(a: str, b: str) -> bool:
    a, b = _stem(a), _stem(b)
    if a == b:
        return True
    return len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a))


def _key_text(strategy: dict) -> str:
    return _norm(str(strategy.get("key") or "").replace("_", " "))


def resolve_declared(name: str, store: StrategyStore):
    """Resolve an agent-declared practice name to an existing strategy id, or None.

    Matching order: exact id, exact key ("block tiling" -> block_tiling) or statement,
    substring, keyword containment, stem/prefix-aware token overlap, fuzzy ratio.
    """
    n = _norm(name)
    if not n:
        return None
    if n in store.strategies:
        return n
    n_tokens = _tokens(n)
    best, best_score = None, 0.0
    for sid, st in store.strategies.items():
        statement = _norm(st.get("statement"))
        score = 0.0
        if n == _key_text(st) or n == statement:
            score = 100.0
        elif n in statement or (n and n in _key_text(st)):
            score = 50.0
        else:
            keywords = [_norm(k) for k in st.get("keywords", []) if k]
            if any(k and (k in n or n in k) for k in keywords):
                score = 30.0
            else:
                statement_tokens = _tokens(statement)
                matched = sum(1 for t in n_tokens if any(_token_match(t, s) for s in statement_tokens))
                ratio = matched / max(1, len(n_tokens))
                if ratio >= 0.6:
                    score = 40.0 * ratio
                else:
                    fuzzy = difflib.SequenceMatcher(None, n, statement).ratio()
                    if fuzzy >= 0.5:
                        score = 25.0 * fuzzy
        if score > best_score:
            best, best_score = sid, score
    return best if best_score >= 20.0 else None


def expand_declared(names, store: StrategyStore) -> list:
    """Expand agent-declared names: combo ids become their member strategy ids.

    Keeps other names untouched so the caller can resolve/promote them normally.
    """
    expanded = []
    for name in names:
        n = _norm(name)
        combo = store.combos.get(n) if n else None
        if combo:
            expanded.extend(combo.get("members") or [])
        else:
            expanded.append(name)
    return expanded


def _code_only(source: str) -> str:
    """Strip comments and string literals: attribution must match code, not prose."""
    try:
        pieces = []
        for tok in tokenize.generate_tokens(io.StringIO(source or "").readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(tok.string)
        return " ".join(pieces).lower()
    except Exception:
        return (source or "").lower()


def _usable_keywords(strategy: dict) -> list:
    """Keywords meaningful for source attribution (identifier-like ones are too weak)."""
    out = []
    for kw in strategy.get("keywords", []):
        k = _norm(kw)
        if not k or len(k) < 4:
            continue
        if re.search(r"[_()\[\]/]", k):  # w_uk / c_kv style identifiers -> drop
            continue
        out.append(k)
    return out


def attribute_from_source(source: str, store: StrategyStore, feats: dict) -> list:
    """Fallback attribution: which practices appear in the solution's code (not prose)."""
    code = _code_only(source)
    ftags = set(feats.get("tags") or [])
    ids = []
    for sid, st in store.strategies.items():
        if not (set(st.get("tags") or []) & ftags):
            continue  # only attribute practices that match the task's tags at all
        for k in _usable_keywords(st):
            if k in code:
                ids.append(sid)
                break
    return sorted(set(ids))


def distill_run(run_dir: str, store: StrategyStore = None, verify: bool = False,
                device: str = "auto", timeout: float = 600.0, force: bool = False,
                results_file: str = None, strict_provenance: bool = False) -> dict:
    store = store or StrategyStore()
    traj_path = os.path.join(run_dir, "trajectory.json")
    traj = read_json(traj_path) or build_trajectory(run_dir)

    report = {
        "run_dir": os.path.abspath(run_dir),
        "task": traj.get("task"),
        "hardware": traj.get("hardware"),
        "replay_ok": None,
        "evidence": [],
        "created": [],
        "combos": [],
        "refined": [],
        "skipped": [],
        "exemplar": None,
    }
    prov = provenance({"device": traj.get("device"), "hardware": traj.get("hardware")})
    report["provenance"] = prov
    if strict_provenance and not prov.get("valid"):
        report["skipped"].append({
            "iter": "*",
            "reason": "provenance mismatch: %s" % (prov.get("reason") or "invalid evidence"),
        })
        return report

    if verify and not traj.get("replay"):
        replay_trajectory(run_dir, device=device, timeout=timeout, results_file=results_file)
        traj = read_json(traj_path)

    replay = traj.get("replay")
    if replay:
        report["replay_ok"] = replay.get("matches") == replay.get("n")

    feats = traj.get("features") or {}
    hardware = traj.get("hardware") or ""
    dtype = (feats.get("dtypes") or [""])[0]
    traj_ref = os.path.relpath(traj_path, ROOT)
    touched = set()
    best = None  # (attempt, used_ids, speedup)

    for attempt in traj.get("attempts", []):
        if attempt.get("distilled") and not force:
            report["skipped"].append({"iter": attempt.get("iter"), "reason": "already distilled"})
            continue
        if report["replay_ok"] is False:
            report["skipped"].append({"iter": attempt.get("iter"), "reason": "replay mismatch"})
            continue

        status = attempt.get("status")
        speedup = float(attempt.get("speedup") or 0.0)
        expected = attempt.get("expected_speedup")
        declared = [str(d) for d in (attempt.get("strategies_declared") or []) if str(d).strip()]
        declared_for_resolution = expand_declared(declared, store)

        resolved = [r for r in (resolve_declared(d, store) for d in declared_for_resolution) if r]
        used = sorted(set(resolved)) or attribute_from_source(attempt.get("solution_source") or "", store, feats)
        outcome = "failure" if status != "pass" else ("success" if speedup >= 1.0 else "neutral")
        ablation = attempt.get("ablation") or {}
        eligibility = bool(attempt.get("evidence_eligible", True))
        promote = bool(attempt.get("promote", ablation.get("significant", False)))
        if not eligibility:
            report["skipped"].append({
                "iter": attempt.get("iter"),
                "reason": "ablation not significant; candidate is not eligible for evidence/confidence",
                "ablation": ablation,
            })
            attempt["distilled"] = True
            continue

        if status == "pass" and eligibility and (best is None or speedup > best[2]):
            best = (attempt, used, speedup)

        for sid in used:
            store.record_evidence(sid, traj_ref, traj.get("task") or "", speedup if status == "pass" else 0.0,
                                  outcome, expected_speedup=expected, hardware=hardware, dtype=dtype,
                                  promote=promote)
            touched.add(sid)
            report["evidence"].append({"strategy": sid, "iter": attempt.get("iter"),
                                       "outcome": outcome, "measured_speedup": speedup,
                                       "expected_speedup": expected})

        # Promote agent-declared practices we cannot resolve yet (only with a real gain).
        if status == "pass" and speedup > 1.05 and promote:
            for name in declared_for_resolution:
                if resolve_declared(name, store) is None:
                    st = store.upsert(
                        statement=name, tags=(feats.get("tags") or [])[:4],
                        op_class=[feats["op_class"]] if feats.get("op_class") else [],
                        keywords=[name], expected=speedup, source="agent-declared", confidence="candidate",
                    )
                    store.record_evidence(st["id"], traj_ref, traj.get("task") or "", speedup, outcome,
                                          expected_speedup=expected, hardware=hardware, dtype=dtype,
                                          promote=promote)
                    touched.add(st["id"])
                    report["created"].append({"strategy": st["id"], "statement": st["statement"]})

        attempt["distilled"] = True

    if best and len(best[1]) >= 2:
        combo = store.upsert_combo(best[1], measured_speedup=best[2],
                                   task=traj.get("task") or "", trajectory=traj_ref)
        report["combos"].append({"combo": combo["id"], "members": combo["members"],
                                 "best_speedup": (combo.get("stats") or {}).get("best_speedup")})

    # code exemplar: the best verified solution of the run, as a few-shot example
    if best and best[2] >= 1.0 and (best[0].get("solution_source") or "").strip():
        attempt, used_ids, speedup = best
        profile_top = [str(row.get("op")) for row in ((attempt.get("profile") or {}).get("top_ops") or [])][:3]
        row = record_exemplar(
            store.root, traj.get("task") or "", attempt["solution_source"], speedup,
            hardware=hardware, dtypes=feats.get("dtypes"), op_class=feats.get("op_class") or "",
            tags=feats.get("tags"), strategies=used_ids, profile_top=profile_top, run_ref=traj_ref,
        )
        report["exemplar"] = {"id": row["id"], "task": row["task"], "speedup": row["speedup"]}

    for sid in sorted(touched):
        note = store.refine_conditions(sid)
        if note:
            report["refined"].append({"strategy": sid, "note": note})

    traj["distilled"] = True
    with open(traj_path, "w", encoding="utf-8") as fh:
        json.dump(traj, fh, ensure_ascii=False, indent=2)
    write_jsonl(os.path.join(run_dir, "attempts.jsonl"), traj.get("attempts", []))
    store.save()
    with open(os.path.join(run_dir, "distill_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.distill", description="Distill a run's trajectory into the strategy pool.")
    ap.add_argument("--run-dir", action="append", required=True, help="agent run directory (repeatable)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="knowledge root (strategies.jsonl / combos.jsonl)")
    ap.add_argument("--verify", action="store_true", help="replay the trajectory before distilling")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--force", action="store_true", help="re-distill attempts already distilled")
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--strict-provenance", action="store_true",
                    help="refuse to learn performance evidence when the actual device does not match "
                         "the claimed hardware (e.g. a CPU run labelled ascend-910b)")
    args = ap.parse_args(argv)

    store = StrategyStore(args.root)
    exit_code = 0
    for run_dir in args.run_dir:
        report = distill_run(run_dir, store=store, verify=args.verify, device=args.device,
                             timeout=args.timeout, force=args.force, results_file=args.results_file,
                             strict_provenance=args.strict_provenance)
        prov = report.get("provenance") or {}
        print("[%s] task=%s hardware=%s device_class=%s valid_evidence=%s replay_ok=%s" % (
            os.path.basename(run_dir.rstrip("/\\")), report["task"], report["hardware"],
            prov.get("device_class"), prov.get("valid"), report["replay_ok"]))
        print("  evidence: %d | created: %d | combos: %d | refined: %d | skipped: %d" % (
            len(report["evidence"]), len(report["created"]), len(report["combos"]),
            len(report["refined"]), len(report["skipped"])))
        for row in report["skipped"][:3]:
            print("    skipped iter=%s: %s" % (row.get("iter"), row.get("reason")))
        for row in report["evidence"][:6]:
            print("    %s iter=%s -> %s (measured %.3fx, expected %s)" % (
                row["strategy"], row["iter"], row["outcome"], row["measured_speedup"], row["expected_speedup"]))
        for row in report["combos"]:
            print("    combo %s: %s -> best %.3fx" % (row["combo"], " + ".join(row["members"]), row["best_speedup"] or 0.0))
        for row in report["refined"]:
            print("    refined %s: %s" % (row["strategy"], row["note"]))
        if report["replay_ok"] is False:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())