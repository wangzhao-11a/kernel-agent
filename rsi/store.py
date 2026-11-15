"""Strategy pool + knowledge base store (human-editable JSONL).

rsi/knowledge/strategies.jsonl : practices with conditions, expected effect, evidence, stats
rsi/knowledge/combos.jsonl     : combination rules (practices that co-occur with measured gains)

The store bootstraps from `practices.PRACTICES` on first use and then evolves:
evidence updates stats/calibration, and `refine_conditions` narrows applicability
(hardware/dtype) when the record shows a clean success/failure split.
"""
from __future__ import annotations

import contextlib
import json
import os
import time

from .detectors import detector_for
from .practices import PRACTICES
from rsi._io import _FileLock, write_jsonl as _write_jsonl

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path: str) -> list:
    if not os.path.isfile(path):
        return []
    # Windows: an unlocked reader (the constructor's load) can hit a file that
    # another process is atomically replacing; open() then fails with
    # PermissionError instead of returning either version.  Retry briefly.
    last = None
    for _attempt in range(40):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                rows = []
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
            return rows
        except PermissionError as exc:
            last = exc
            time.sleep(0.05)
    raise last


class StrategyStore:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self.strategies_path = os.path.join(root, "strategies.jsonl")
        self.combos_path = os.path.join(root, "combos.jsonl")
        self.strategies = {}
        self.combos = {}
        self._tx_depth = 0
        self._snapshot = {}
        self._combo_snapshot = {}
        self.load()

    # ---------------------------------------------------------------- persistence
    @staticmethod
    def _merge_rows(current: dict, rows: list) -> dict:
        """Reload rows in place so callers holding a row dict see the update.

        A transaction must re-read the file, but replacing the dicts would make
        every reference obtained from upsert()/find_by_key() stale -- the tests
        (and the loop) hold those rows across record_evidence() calls.
        """
        merged = {}
        for row in rows:
            rid = row.get("id")
            if not rid:
                continue
            existing = current.get(rid)
            if existing is not None:
                existing.clear()
                existing.update(row)
                row = existing
            merged[rid] = row
        return merged

    def _load_files(self) -> None:
        """Replace in-memory state with what is on disk (no bootstrap)."""
        self.strategies = self._merge_rows(self.strategies, _read_jsonl(self.strategies_path))
        self.combos = self._merge_rows(self.combos, _read_jsonl(self.combos_path))
        # Snapshot before attaching seed detectors: a detector filled in here is
        # a real change that the next save must persist.
        self._snapshot = {rid: self._row_sig(row) for rid, row in self.strategies.items()}
        self._combo_snapshot = {cid: self._row_sig(row) for cid, row in self.combos.items()}
        self._attach_seed_detectors()

    @staticmethod
    def _row_sig(row: dict) -> str:
        return json.dumps(row, ensure_ascii=False, sort_keys=True)

    def load(self) -> None:
        # Read under the same lock as the writers: on Windows an unlocked reader
        # can open the file in the middle of another process's os.replace and
        # fail with PermissionError (and the writer then retries for seconds).
        if self._tx_depth:
            self._load_files()
        else:
            with _FileLock(os.path.join(self.root, ".knowledge.lock")):
                self._load_files()
        if not self.strategies:
            self.bootstrap()

    @contextlib.contextmanager
    def transaction(self):
        """Cross-process read-modify-write: reload, mutate, write, all under one lock.

        The id counter reads the in-memory rows, so allocating an id before the
        lock lets two processes pick the same id and the later save() silently
        drops the earlier row.  Every mutating entry point therefore reloads
        inside the lock before it allocates or edits anything (same contract as
        MemoryStore.transaction).  Re-entrant: nested calls reuse the outer lock
        and the outer transaction writes once.
        """
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1
            return
        with _FileLock(os.path.join(self.root, ".knowledge.lock")):
            # Keep edits made outside a transaction: a caller may set fields on
            # a row (e.g. confidence) and call another mutating method next.
            # Only rows that actually differ from the last load are re-applied;
            # untouched (stale) rows still lose to a newer disk version.
            # Copy the values: _load_files updates the existing row dicts in
            # place, so holding the same object here would make the restore a
            # no-op (the local edit would already have been overwritten).
            local = {rid: dict(row) for rid, row in self.strategies.items()
                     if self._snapshot.get(rid) != self._row_sig(row)}
            local_combos = {cid: dict(row) for cid, row in self.combos.items()
                            if self._combo_snapshot.get(cid) != self._row_sig(row)}
            self._load_files()
            for rid, row in local.items():
                existing = self.strategies.get(rid)
                if existing is None:
                    self.strategies[rid] = row
                else:
                    existing.update(row)
            for cid, row in local_combos.items():
                existing = self.combos.get(cid)
                if existing is None:
                    self.combos[cid] = row
                else:
                    existing.update(row)
            self._tx_depth = 1
            try:
                yield
                self._write_all()
            finally:
                self._tx_depth = 0

    @staticmethod
    def _merge_key(row: dict) -> str:
        """Second line of defence: the same practice never lands twice.

        Rows are identified by their practice key when they have one, so two
        writers that picked different ids for the same practice still collapse
        into one row instead of producing a duplicate on disk.
        """
        key = str(row.get("key") or "").strip().lower()
        return "key:" + key if key else "id:" + str(row.get("id") or "")

    def _write_all(self) -> None:
        """Merge disk + locally changed rows, then write both files.

        Caller holds the lock.  Only rows this instance actually changed since
        its last load may overwrite disk: writing every in-memory row would let
        a process that loaded early clobber a row another process just updated
        (the same lost-update bug, one level down from id allocation).
        """
        disk_strategies = {}
        for row in _read_jsonl(self.strategies_path):
            if row.get("id"):
                disk_strategies[self._merge_key(row)] = row
        for rid, row in self.strategies.items():
            changed = self._snapshot.get(rid) != self._row_sig(row)
            if changed or self._merge_key(row) not in disk_strategies:
                disk_strategies[self._merge_key(row)] = row
        disk_combos = {}
        for row in _read_jsonl(self.combos_path):
            if row.get("id"):
                disk_combos[self._merge_key(row)] = row
        for cid, row in self.combos.items():
            changed = self._combo_snapshot.get(cid) != self._row_sig(row)
            if changed or self._merge_key(row) not in disk_combos:
                disk_combos[self._merge_key(row)] = row
        _write_jsonl(self.strategies_path, [disk_strategies[k] for k in sorted(disk_strategies)])
        _write_jsonl(self.combos_path, [disk_combos[k] for k in sorted(disk_combos)])
        self._snapshot = {rid: self._row_sig(row) for rid, row in self.strategies.items()}
        self._combo_snapshot = {cid: self._row_sig(row) for cid, row in self.combos.items()}

    def _attach_seed_detectors(self) -> None:
        """Older pools predate detectors; fill them in by practice key."""
        for st in self.strategies.values():
            if st.get("detector"):
                continue
            detector = detector_for(st.get("key") or "")
            if detector:
                st["detector"] = detector

    def bootstrap(self) -> None:
        with self.transaction():
            for p in PRACTICES:
                self.upsert(
                    statement=p["statement"], tags=p["tags"], op_class=p["op_class"],
                    keywords=p["keywords"], expected=p["expected_speedup"],
                    source="seed", key=p["key"], confidence="seed",
                    detector=detector_for(p["key"]),
                )

    def save(self) -> None:
        with _FileLock(os.path.join(self.root, ".knowledge.lock")):
            self._write_all()

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
               source="distilled", key=None, confidence="candidate", detector=None):
        with self.transaction():
            return self._upsert(statement, tags=tags, op_class=op_class, keywords=keywords,
                                expected=expected, source=source, key=key,
                                confidence=confidence, detector=detector)

    def _upsert(self, statement, tags=None, op_class=None, keywords=None, expected=None,
                source="distilled", key=None, confidence="candidate", detector=None):
        normalized = statement.strip().lower()
        for st in self.strategies.values():
            if (key and st.get("key") == key) or st["statement"].strip().lower() == normalized:
                for kw in keywords or []:
                    if kw not in st.setdefault("keywords", []):
                        st["keywords"].append(kw)
                if detector and not st.get("detector"):
                    st["detector"] = detector
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
            "detector": detector or None,
            "confidence": confidence,
            "source": source,
            "evidence": [],
            "stats": {"applied": 0, "succeeded": 0, "failed": 0, "regressed": 0, "neutral": 0,
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
                        expected_speedup=None, hardware="", dtype="", run_id="",
                        holdout_ok=None, shape_signature=None, ablation=None):
        with self.transaction():
            return self._record_evidence(
                sid, trajectory, task, measured_speedup, outcome,
                expected_speedup=expected_speedup, hardware=hardware, dtype=dtype,
                run_id=run_id, holdout_ok=holdout_ok,
                shape_signature=shape_signature, ablation=ablation)

    def _record_evidence(self, sid, trajectory, task, measured_speedup, outcome,
                         expected_speedup=None, hardware="", dtype="", run_id="",
                         holdout_ok=None, shape_signature=None, ablation=None):
        st = self.strategies.get(sid)
        if st is None:
            raise KeyError("unknown strategy id %r (the store changed under this caller)" % sid)
        st["evidence"].append({
            "trajectory": trajectory, "run_id": run_id, "task": task,
            "measured_speedup": round(float(measured_speedup or 0.0), 4),
            "expected_speedup": expected_speedup,
            "outcome": outcome, "hardware": hardware, "dtype": dtype,
            "ablation": ablation or {"performed": False},
            "holdout_ok": holdout_ok,
            "shape_signature": dict(shape_signature or {}),
            "at": _now(),
        })
        st["evidence"] = st["evidence"][-40:]
        stats = st["stats"]
        stats["applied"] = int(stats.get("applied", 0)) + 1
        if outcome == "success":
            stats["succeeded"] = int(stats.get("succeeded", 0)) + 1
        elif outcome == "regression":
            stats["regressed"] = int(stats.get("regressed", 0)) + 1
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
        if outcome == "success":
            # Verification is a reproduction claim: at least two independent runs
            # must have succeeded, and each one must also have passed the holdout
            # shapes.  A run with no holdout configured (holdout_ok is None) can
            # never verify a strategy (P3.1).
            runs = {str(e.get("run_id") or e.get("trajectory") or "")
                    for e in st["evidence"]
                    if e.get("outcome") == "success"
                    and e.get("holdout_ok") is True}
            runs.discard("")
            st["confidence"] = "verified" if len(runs) >= 2 else "candidate"
        st["updated"] = _now()
        return st

    def refine_conditions(self, sid):
        """Narrow hardware/dtypes to the subset with a clean success record."""
        st = self.strategies[sid]
        evidence = [e for e in st["evidence"]
                    if e.get("outcome") in ("success", "failure", "regression")]
        if len(evidence) < 2:
            return None
        groups = {}
        for e in evidence:
            sig = e.get("shape_signature") or {}
            key = (e.get("hardware") or "unknown", e.get("dtype") or "unknown",
                   sig.get("half_aligned_32b"), sig.get("prime_outer"),
                   sig.get("outer_bucket"))
            g = groups.setdefault(key, {"success": 0, "failure": 0})
            if e["outcome"] == "success":
                g["success"] += 1
            else:
                g["failure"] += 1
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

        # Shape features that are consistent across the winning group narrow the
        # applicability too (P3.2), using the same signature keys as operator memory.
        shape_values = {}
        for e in evidence:
            if e.get("outcome") != "success":
                continue
            sig = e.get("shape_signature") or {}
            key = (e.get("hardware") or "unknown", e.get("dtype") or "unknown",
                   sig.get("half_aligned_32b"), sig.get("prime_outer"),
                   sig.get("outer_bucket"))
            if key not in winners:
                continue
            for field in ("half_aligned_32b", "prime_outer", "outer_bucket"):
                if sig.get(field) is not None:
                    shape_values.setdefault(field, set()).add(sig[field])
        if shape_values and all(len(values) == 1 for values in shape_values.values()):
            narrowed = {k: next(iter(v)) for k, v in shape_values.items()}
            if cond.get("shape") != narrowed:
                cond["shape"] = narrowed
                notes.append("shape->%s" % json.dumps(narrowed, sort_keys=True))

        if not notes:
            return None

        def _fmt_group(k):
            base = "%s/%s" % (k[0], k[1])
            shape = {field: value for field, value in
                     zip(("half_aligned_32b", "prime_outer", "outer_bucket"), k[2:])
                     if value is not None}
            return base + ("/shape:" + json.dumps(shape, sort_keys=True) if shape else "")

        note = "narrowed after evidence (%s); excluded %s" % (
            "; ".join(notes), ", ".join(_fmt_group(k) for k in losers))
        cond["refined_note"] = note
        st["updated"] = _now()
        return note

    # ---------------------------------------------------------------- combos
    def upsert_combo(self, member_ids, measured_speedup=None, task="", trajectory="",
                     contributions=None):
        with self.transaction():
            return self._upsert_combo(member_ids, measured_speedup=measured_speedup,
                                      task=task, trajectory=trajectory,
                                      contributions=contributions)

    def _upsert_combo(self, member_ids, measured_speedup=None, task="", trajectory="",
                      contributions=None):
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
                if contributions:
                    combo["contributions"] = dict(contributions)
                combo["updated"] = _now()
                return combo
        cid = self._next_id("combo")
        rec = {
            "id": cid,
            "members": list(key),
            "contributions": dict(contributions or {}),
            "conditions": {},
            "stats": {"applied": 1, "best_speedup": round(float(measured_speedup), 4) if measured_speedup else None},
            "evidence": [{"trajectory": trajectory, "task": task,
                          "measured_speedup": measured_speedup, "at": _now()}],
            "created": _now(),
            "updated": _now(),
        }
        self.combos[cid] = rec
        return rec