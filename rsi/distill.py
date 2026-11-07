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

from harness.noise import classify, epsilon_for, promotion_floor  # noqa: E402
from rsi.detectors import code_only as _detector_code_only  # noqa: E402
from rsi.detectors import match as detector_match  # noqa: E402
from rsi.exemplars import record_exemplar, update_descendant_gain  # noqa: E402
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
    return best if best_score >= 30.0 else None


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
    return _detector_code_only(source)


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
                results_file: str = None, strict_provenance: bool = True) -> dict:
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
        "claimed_only": [],
        "exemplar": None,
        "descendant_gain": [],
        "skills": [],
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
    epsilon = epsilon_for(device=traj.get("device"), hardware=hardware)
    run_id = os.path.basename(os.path.abspath(str(run_dir).rstrip("/\\")))
    traj_ref = os.path.relpath(traj_path, ROOT)
    best_path = os.path.abspath(str(((traj.get("best") or {}).get("path")) or ""))
    ablations = {}
    for row in traj.get("attempts", []):
        if row.get("kind") == "ablation" and row.get("ablation_of"):
            ablations[str(row["ablation_of"])] = row
    touched = set()
    best = None  # (attempt, used_ids, speedup, outcome, contributions)

    def _ablation_info(sid, full_ci):
        """(field for the evidence row, whether the removal significantly hurt)."""
        row = ablations.get(sid)
        if not row:
            return {"performed": False, "without_speedup_ci": None}, False
        without_ci = row.get("speedup_ci")
        status = row.get("status")
        valid = row.get("ablation_valid", True)
        performed = bool(status in ("pass", "pass_unstable") and without_ci
                         and len(without_ci) >= 2 and valid)
        contributed = bool(status in ("pass", "pass_unstable") and full_ci
                           and len(full_ci) >= 2
                           and float(without_ci[1]) < float(full_ci[0]))
        return ({
            "performed": bool(performed),
            "valid": bool(valid),
            "problem": row.get("ablation_problem") or "",
            "without_speedup": row.get("speedup"),
            "without_speedup_ci": without_ci,
            "status": status,
            "unstable": status == "pass_unstable",
            "contributed": contributed,
        }, contributed)

    for attempt in traj.get("attempts", []):
        if attempt.get("kind") == "ablation":
            continue
        if attempt.get("distilled") and not force:
            report["skipped"].append({"iter": attempt.get("iter"), "reason": "already distilled"})
            continue
        if report["replay_ok"] is False:
            report["skipped"].append({"iter": attempt.get("iter"), "reason": "replay mismatch"})
            continue

        status = attempt.get("status")
        # Prefer the epsilon recorded at measurement time: the gate used for this
        # attempt, on the device that actually produced the timing.
        attempt_epsilon = attempt.get("epsilon")
        eps = float(attempt_epsilon) if attempt_epsilon is not None else epsilon
        if status == "pass_unstable":
            # Instability widens the CI (the worker already did that); it does not
            # throw the attempt away.  A lower bound above 1 + epsilon is still
            # evidence, so it is distilled like any other pass.
            report.setdefault("unstable_kept", []).append({
                "iter": attempt.get("iter"),
                "stability_delta": attempt.get("stability_delta"),
                "epsilon": eps,
            })
        # P2.3: only correctness failures and significant regressions are strategy
        # failures.  Build/import/JSON/evaluator errors go to failure memory, not
        # to the strategy statistics.
        if status not in ("pass", "pass_unstable", "fail_correctness"):
            report["skipped"].append({
                "iter": attempt.get("iter"),
                "reason": "stage failure (%s): not attributed to strategies" % status,
            })
            attempt["distilled"] = True
            continue
        speedup = float(attempt.get("speedup") or 0.0)
        expected = attempt.get("expected_speedup")
        declared = [str(d) for d in (attempt.get("strategies_declared") or []) if str(d).strip()]
        declared_for_resolution = expand_declared(declared, store)

        resolved = [r for r in (resolve_declared(d, store) for d in declared_for_resolution) if r]
        source = attempt.get("solution_source") or ""
        verified_declared, claimed_only = [], []
        for sid in resolved:
            detector = (store.strategies.get(sid) or {}).get("detector")
            if detector and not detector_match(detector, source):
                claimed_only.append(sid)
            else:
                verified_declared.append(sid)
        if claimed_only:
            report["claimed_only"].append({
                "iter": attempt.get("iter"),
                "strategies": sorted(set(claimed_only)),
                "reason": "declared practice not found in the solution source",
            })
        if declared:
            # Declarations are checked; do not fall back to keyword attribution,
            # otherwise an unverifiable claim would still buy evidence.
            used = sorted(set(verified_declared))
        else:
            used = attribute_from_source(source, store, feats)
        full_ci = attempt.get("speedup_ci")
        if status not in ("pass", "pass_unstable"):
            outcome = "failure"
        else:
            outcome = classify(full_ci, eps)
        is_best = bool(best_path and status in ("pass", "pass_unstable")
                       and os.path.abspath(str(attempt.get("solution") or "")) == best_path)
        contributions = {}
        for sid in used:
            if is_best:
                ablation, contributed = _ablation_info(sid, full_ci)
            else:
                # Only the best solution was ablated; other attempts cannot claim
                # a strategy success without their own ablation.
                ablation, contributed = {"performed": False, "without_speedup_ci": None}, False
            sid_outcome = "neutral" if (outcome == "success" and not contributed) else outcome
            contributions[sid] = contributed
            store.record_evidence(sid, traj_ref, traj.get("task") or "",
                                  speedup if status in ("pass", "pass_unstable") else 0.0,
                                  sid_outcome, expected_speedup=expected, hardware=hardware,
                                  dtype=dtype, run_id=run_id, ablation=ablation,
                                  holdout_ok=attempt.get("holdout_ok"),
                                  shape_signature=attempt.get("shape_signature"))
            touched.add(sid)
            report["evidence"].append({"strategy": sid, "iter": attempt.get("iter"),
                                       "outcome": sid_outcome, "measured_speedup": speedup,
                                       "expected_speedup": expected,
                                       "speedup_ci": full_ci,
                                       "holdout_ok": attempt.get("holdout_ok"),
                                       "shape_signature": attempt.get("shape_signature"),
                                       "ablation": ablation})

        if is_best and (best is None or speedup > best[2]):
            best = (attempt, used, speedup, outcome, contributions)

        # Promote agent-declared practices we cannot resolve yet (only when the
        # new strategy clears the stricter 1 + 2*epsilon promotion floor).  It is
        # a candidate: without an ablation it cannot claim a success yet.
        promote_floor = promotion_floor(eps)
        if (status in ("pass", "pass_unstable") and promote_floor is not None
                and full_ci
                and float(full_ci[0]) > promote_floor):
            for name in declared_for_resolution:
                if resolve_declared(name, store) is None:
                    st = store.upsert(
                        statement=name, tags=(feats.get("tags") or [])[:4],
                        op_class=[feats["op_class"]] if feats.get("op_class") else [],
                        keywords=[name], expected=speedup, source="agent-declared", confidence="candidate",
                    )
                    store.record_evidence(
                        st["id"], traj_ref, traj.get("task") or "", speedup, "neutral",
                        expected_speedup=expected, hardware=hardware, dtype=dtype,
                        run_id=run_id, ablation={"performed": False, "without_speedup_ci": None})
                    touched.add(st["id"])
                    report["created"].append({"strategy": st["id"], "statement": st["statement"]})

        attempt["distilled"] = True

    # P4.4: a strategy that just became verified compiles its best solution into
    # a parameterized skill; two independent harness runs validate the template.
    if best and (best[0].get("solution_source") or "").strip() and traj.get("task_file"):
        from rsi.memory import MemoryStore
        from rsi.skills import auto_compile_from_run

        memory = MemoryStore(store.root)
        for sid in sorted(touched):
            strategy = store.strategies.get(sid) or {}
            if strategy.get("confidence") != "verified" or sid not in best[1]:
                continue
            task_file = traj.get("task_file") or ""
            if not os.path.isfile(task_file):
                continue
            try:
                result = auto_compile_from_run(
                    memory, strategy, task_file, best[0]["solution_source"], feats,
                    hardware, best[2], device=traj.get("device") or "cpu",
                    source_ref=best[0].get("solution") or "")
            except Exception as exc:
                report["skills"].append({"strategy": sid, "status": "error",
                                         "error": "%s: %s" % (type(exc).__name__, exc)})
                continue
            report["skills"].append({
                "strategy": sid,
                "status": result.get("status"),
                "name": (result.get("skill") or {}).get("name"),
                "confidence": (result.get("skill") or {}).get("confidence"),
                "passes": result.get("passes"),
            })

    if best and best[3] == "success" and len(best[1]) >= 2:
        contributions = best[4]
        missing = [sid for sid in best[1] if not contributions.get(sid)]
        if missing:
            report["skipped"].append({
                "iter": "combo",
                "reason": "members without ablation evidence: %s" % ", ".join(missing),
            })
        else:
            combo = store.upsert_combo(best[1], measured_speedup=best[2],
                                       task=traj.get("task") or "", trajectory=traj_ref,
                                       contributions=contributions)
            report["combos"].append({"combo": combo["id"], "members": combo["members"],
                                     "contributions": combo.get("contributions"),
                                     "best_speedup": (combo.get("stats") or {}).get("best_speedup")})

    # code exemplar: only a statistically significant win is a winning example
    if (best and best[2] >= 1.0 and best[3] == "success"
            and (best[0].get("solution_source") or "").strip()):
        attempt, used_ids, speedup, _outcome, _contributions = best
        profile_top = [str(row.get("op")) for row in ((attempt.get("profile") or {}).get("top_ops") or [])][:3]
        row = record_exemplar(
            store.root, traj.get("task") or "", attempt["solution_source"], speedup,
            hardware=hardware, dtypes=feats.get("dtypes"), op_class=feats.get("op_class") or "",
            tags=feats.get("tags"), strategies=used_ids, profile_top=profile_top, run_ref=traj_ref,
            device_class=prov.get("device_class") or "", evidence_valid=bool(prov.get("valid")),
        )
        report["exemplar"] = {"id": row["id"], "task": row["task"], "speedup": row["speedup"]}

    # P3.5: credit the exemplars this run started from when their descendants
    # reached a better speedup than the exemplar itself.
    used_exemplars = sorted({str(eid)
                             for attempt in traj.get("attempts", [])
                             for eid in (attempt.get("exemplars_used") or []) if str(eid)})
    if used_exemplars and best:
        updated = update_descendant_gain(store.root, used_exemplars, best[2])
        report["descendant_gain"] = [
            {"id": row["id"], "descendant_gain": row.get("descendant_gain")}
            for row in updated
        ]

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
    ap.add_argument("--strict-provenance", dest="strict_provenance", action="store_true", default=True,
                    help="(default) refuse to learn performance evidence when the actual device does not match "
                         "the claimed hardware (e.g. a CPU run labelled ascend-910b)")
    ap.add_argument("--allow-mechanics-evidence", dest="strict_provenance", action="store_false",
                    help="explicitly allow mechanics-only runs (e.g. CPU) to write performance evidence "
                         "under the claimed hardware label")
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
        print("  evidence: %d | created: %d | combos: %d | refined: %d | skipped: %d | "
              "skills: %d | claimed_only: %d" % (
            len(report["evidence"]), len(report["created"]), len(report["combos"]),
            len(report["refined"]), len(report["skipped"]), len(report.get("skills") or []),
            len(report.get("claimed_only") or [])))
        for row in (report.get("skills") or [])[:3]:
            print("    skill %s: %s (%s, passes=%s)" % (
                row.get("name"), row.get("status"), row.get("confidence"), row.get("passes")))
        for row in (report.get("claimed_only") or [])[:3]:
            print("    claimed-only iter=%s: %s" % (row.get("iter"), ", ".join(row.get("strategies") or [])))
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