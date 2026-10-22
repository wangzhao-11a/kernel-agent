"""Strategy pool + knowledge base store (human-editable JSONL).

rsi/knowledge/strategies.jsonl : practices with conditions, expected effect, evidence, stats
rsi/knowledge/combos.jsonl     : combination rules (practices that co-occur with measured gains)

The store bootstraps from `practices.PRACTICES` on first use and then evolves:
evidence updates stats/calibration, and `refine_conditions` narrows applicability
(hardware/dtype) when the record shows a clean success/failure split.
"""
from __future__ import annotations

import json
import os
import time

from .practices import PRACTICES

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path: str) -> list:
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


def _write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class StrategyStore:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self.strategies_path = os.path.join(root, "strategies.jsonl")
        self.combos_path = os.path.join(root, "combos.jsonl")
        self.strategies = {}
        self.combos = {}
        self.load()

    # ---------------------------------------------------------------- persistence
    def load(self) -> None:
        for row in _read_jsonl(self.strategies_path):
            if row.get("id"):
                self.strategies[row["id"]] = row
        for row in _read_jsonl(self.combos_path):
            if row.get("id"):
                self.combos[row["id"]] = row
        if not self.strategies:
            self.bootstrap()

    def bootstrap(self) -> None:
        for p in PRACTICES:
            self.upsert(
                statement=p["statement"], tags=p["tags"], op_class=p["op_class"],
                keywords=p["keywords"], expected=p["expected_speedup"],
                source="seed", key=p["key"], confidence="seed",
            )
        self.save()

    def save(self) -> None:
        _write_jsonl(self.strategies_path, [self.strategies[k] for k in sorted(self.strategies)])
        _write_jsonl(self.combos_path, [self.combos[k] for k in sorted(self.combos)])

    # ---------------------------------------------------------------- strategies
    def _next_id(self, prefix: str) -> str:
        top = 0
        for sid in self.strategies:
            if sid.startswith(prefix + "-"):
                try:
                    top = max(top, int(sid.split("-")[1]))
                except (IndexError, ValueError):
                    pass
        for cid in self.combos:
            if cid.startswith(prefix + "-"):
                try:
                    top = max(top, int(cid.split("-")[1]))
                except (IndexError, ValueError):
                    pass
        return "%s-%04d" % (prefix, top + 1)

    def upsert(self, statement, tags=None, op_class=None, keywords=None, expected=None,
               source="distilled", key=None, confidence="candidate"):
        normalized = statement.strip().lower()
        for st in self.strategies.values():
            if (key and st.get("key") == key) or st["statement"].strip().lower() == normalized:
                for kw in keywords or []:
                    if kw not in st.setdefault("keywords", []):
                        st["keywords"].append(kw)
                st["updated"] = _now()
                return st
        sid = self._next_id("stg")
        rec = {
            "id": sid,
            "key": key or "",
            "statement": statement,
            "tags": list(tags or []),
            "conditions": {"op_class": list(op_class or []), "hardware": [], "dtypes": []},
            "expected": {"speedup": float(expected) if expected else 1.0},
            "keywords": list(keywords or []),
            "confidence": confidence,
            "source": source,
            "evidence": [],
            "stats": {"applied": 0, "succeeded": 0, "failed": 0, "neutral": 0,
                      "ema_speedup": None, "calibration_mae": None},
            "created": _now(),
            "updated": _now(),
        }
        self.strategies[sid] = rec
        return rec

    def find_by_key(self, key: str):
        for st in self.strategies.values():
            if st.get("key") == key:
                return st
        return None

    def record_evidence(self, sid, trajectory, task, measured_speedup, outcome,
                        expected_speedup=None, hardware="", dtype=""):
        st = self.strategies[sid]
        st["evidence"].append({
            "trajectory": trajectory, "task": task,
            "measured_speedup": round(float(measured_speedup or 0.0), 4),
            "expected_speedup": expected_speedup,
            "outcome": outcome, "hardware": hardware, "dtype": dtype, "at": _now(),
        })
        st["evidence"] = st["evidence"][-40:]
        stats = st["stats"]
        stats["applied"] = int(stats.get("applied", 0)) + 1
        if outcome == "success":
            stats["succeeded"] = int(stats.get("succeeded", 0)) + 1
        elif outcome == "failure":
            stats["failed"] = int(stats.get("failed", 0)) + 1
        else:
            stats["neutral"] = int(stats.get("neutral", 0)) + 1
        if outcome in ("success", "neutral") and measured_speedup:
            prev = stats.get("ema_speedup")
            stats["ema_speedup"] = round(float(measured_speedup), 4) if prev is None else round(0.6 * prev + 0.4 * float(measured_speedup), 4)
        if expected_speedup is not None and measured_speedup:
            err = abs(float(expected_speedup) - float(measured_speedup))
            prev = stats.get("calibration_mae")
            stats["calibration_mae"] = round(err, 4) if prev is None else round(0.6 * prev + 0.4 * err, 4)
        if outcome == "success" and st.get("confidence") != "seed":
            st["confidence"] = "verified"
        st["updated"] = _now()
        return st

    def refine_conditions(self, sid):
        """Narrow hardware/dtypes to the subset with a clean success record."""
        st = self.strategies[sid]
        evidence = [e for e in st["evidence"] if e.get("outcome") in ("success", "failure")]
        if len(evidence) < 2:
            return None
        groups = {}
        for e in evidence:
            key = (e.get("hardware") or "unknown", e.get("dtype") or "unknown")
            g = groups.setdefault(key, {"success": 0, "failure": 0})
            g[e["outcome"]] += 1
        winners = [k for k, g in groups.items() if g["success"] and not g["failure"]]
        losers = [k for k, g in groups.items() if g["failure"] and not g["success"]]
        if not winners or not losers:
            return None
        cond = st["conditions"]
        notes = []
        hw = sorted({k[0] for k in winners if k[0] != "unknown"})
        dtypes = sorted({k[1] for k in winners if k[1] != "unknown"})
        if hw and set(hw) != set(cond.get("hardware") or []):
            cond["hardware"] = hw
            notes.append("hardware->%s" % ",".join(hw))
        if dtypes and set(dtypes) != set(cond.get("dtypes") or []):
            cond["dtypes"] = dtypes
            notes.append("dtypes->%s" % ",".join(dtypes))
        if not notes:
            return None
        note = "narrowed after evidence (%s); excluded %s" % (
            "; ".join(notes), ", ".join("%s/%s" % k for k in losers))
        cond["refined_note"] = note
        st["updated"] = _now()
        return note

    # ---------------------------------------------------------------- combos
    def upsert_combo(self, member_ids, measured_speedup=None, task="", trajectory=""):
        key = tuple(sorted(member_ids))
        for combo in self.combos.values():
            if tuple(sorted(combo.get("members", []))) == key:
                combo["stats"]["applied"] = int(combo["stats"].get("applied", 0)) + 1
                if measured_speedup:
                    best = combo["stats"].get("best_speedup") or 0.0
                    combo["stats"]["best_speedup"] = round(max(best, float(measured_speedup)), 4)
                combo["evidence"].append({"trajectory": trajectory, "task": task,
                                          "measured_speedup": measured_speedup, "at": _now()})
                combo["evidence"] = combo["evidence"][-20:]
                combo["updated"] = _now()
                return combo
        cid = self._next_id("combo")
        rec = {
            "id": cid,
            "members": list(key),
            "conditions": {},
            "stats": {"applied": 1, "best_speedup": round(float(measured_speedup), 4) if measured_speedup else None},
            "evidence": [{"trajectory": trajectory, "task": task,
                          "measured_speedup": measured_speedup, "at": _now()}],
            "created": _now(),
            "updated": _now(),
        }
        self.combos[cid] = rec
        return rec