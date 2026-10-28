"""Operator-development memory: facts, failures, skills, strategies + scope-aware retrieval.

The existing strategy pool answers "what worked". Operator development also needs:

    fact     - a hard constraint from code/docs/human (no experiment required)
    failure  - an error signature, its stage/repro/workaround (negative memory)
    skill    - an executable template with preconditions and validation evidence

All four live in the same knowledge root and are retrieved through the same
scope filter.  The rule is deliberately strict: an item whose scope does not
match the current SOC/CANN/dtype/shape/form is not injected, because a wrong
prior is worse than no prior.

    python -m rsi.memory audit
    python -m rsi.memory context --op swi_glu --soc ascend950 --dtype bf16 \
        --shape 4096,8192 --dim -1 --form ascendc
    python -m rsi.memory add-fact --file fact.json
    python -m rsi.memory distill-run --run-dir runs/agent/<run> \
        --op swi_glu --soc ascend950 --dtype bf16 --form ascendc
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
FACTS_FILE = "facts.jsonl"
FAILURES_FILE = "failures.jsonl"
SKILLS_FILE = "skills.jsonl"
STRATEGIES_FILE = "strategies.jsonl"

DTYPE_BYTES = {
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float32": 4, "fp32": 4, "float": 4,
    "int8": 1, "uint8": 1, "int32": 4, "int64": 8,
}

TYPE_WEIGHT = {"fact": 1.00, "failure": 0.92, "skill": 0.86, "strategy": 0.80}
CONFIDENCE_WEIGHT = {"verified": 1.0, "candidate": 0.6, "seed": 0.35, "refuted": 0.05}
AUTHORITY_WEIGHT = {"code": 1.0, "human": 1.0, "experiment": 0.85, "llm": 0.35}


class MemoryLockError(RuntimeError):
    pass


class _FileLock:
    """Cross-process lock for the memory JSONL files.

    Multiple agent runs (different cards/containers) may distill at the same time.
    Without a lock each process does load -> mutate -> save and silently drops the
    others' updates.  The lock file contains the holder pid; a stale lock (crashed
    process) is removed automatically.
    """

    def __init__(self, path: str, timeout: float = 20.0, poll: float = 0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self.acquired = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            try:
                out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                                     capture_output=True, text=True, timeout=10)
                return str(pid) in out.stdout
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    with open(self.path, "r", encoding="utf-8") as fh:
                        held = json.load(fh)
                except (OSError, ValueError):
                    held = {}
                pid = int(held.get("pid") or 0)
                if not self._pid_alive(pid):
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                if time.time() >= deadline:
                    raise MemoryLockError("memory lock held by pid %s: %s" % (pid, self.path))
                time.sleep(self.poll)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "created": utc_now()}, fh)
            self.acquired = True
            return self

    def __exit__(self, *exc):
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False


# --------------------------------------------------------------------------- utils
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def _match_norm(value) -> str:
    """Loose key for names/SOC/dtype matching: ignore case, hyphens, underscores, spaces."""
    return re.sub(r"[-_\s]+", "", str(value or "").lower())


def _text_tokens(text: str) -> set:
    """Token set for free-text memory search; adds CJK bigrams so Chinese text matches."""
    raw = str(text or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", raw))
    for run in re.findall(r"[\u4e00-\u9fff]+", raw):
        tokens.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
        tokens.update(run)
    return {t for t in tokens if len(t) > 1 or t.isdigit()}


DTYPE_ALIASES = {
    "fp16": "float16", "half": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp32": "float32", "float": "float32", "float32": "float32",
    "int8": "int8", "uint8": "uint8", "int32": "int32", "int64": "int64",
}


def _canonical_dtype(value) -> str:
    return DTYPE_ALIASES.get(_match_norm(value), _match_norm(value))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if str(value).strip() else []


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
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:n]


def _contains(wanted: str, actual: str) -> bool:
    w, a = _match_norm(wanted), _match_norm(actual)
    if not w or not a:
        return False
    if w == "*" or a == "*":
        return True
    return w == a or w in a or a in w


def _version_tuple(text: str) -> tuple:
    nums = re.findall(r"\d+", str(text or ""))
    return tuple(int(n) for n in nums[:5])


def version_ok(actual: str, wanted: str) -> bool:
    wanted = str(wanted or "").strip()
    if wanted in ("", "*", "any"):
        return True
    match = re.match(r"^(>=|<=|>|<|==|=)?\s*(.+)$", wanted)
    op = match.group(1) or "=="
    target = match.group(2)
    a, t = _version_tuple(actual), _version_tuple(target)
    if not a or not t:
        return _norm(actual) == _norm(target)
    width = max(len(a), len(t))
    a += (0,) * (width - len(a))
    t += (0,) * (width - len(t))
    return {
        "=": a == t, "==": a == t, ">=": a >= t,
        "<=": a <= t, ">": a > t, "<": a < t,
    }.get(op, False)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, int(math.isqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True


def _parse_shape(shape) -> tuple:
    if isinstance(shape, (list, tuple)):
        return tuple(int(x) for x in shape)
    text = str(shape or "").strip().strip("()[]")
    if not text:
        return ()
    return tuple(int(x) for x in re.findall(r"-?\d+", text))


def _outer_bucket(outer: int) -> str:
    if outer <= 1:
        return "1"
    if outer <= 64:
        return "2-64"
    if outer <= 1024:
        return "65-1024"
    return "1025+"


def shape_signature(shape, dim: int = -1, dtype: str = "") -> dict:
    """Compact shape features used for scope matching in operator memory."""
    dims = _parse_shape(shape)
    if not dims:
        return {}
    ndim = len(dims)
    d = int(dim)
    if d < 0:
        d += ndim
    if d < 0 or d >= ndim:
        return {}
    inner = 1
    for size in dims[d + 1:]:
        inner *= size
    outer = 1
    for size in dims[:d]:
        outer *= size
    half = (dims[d] // 2) * inner
    elem_bytes = DTYPE_BYTES.get(_norm(dtype), 0)
    return {
        "rank": ndim,
        "dim": d,
        "outer": outer,
        "inner": inner,
        "half": half,
        "half_is_1": half == 1,
        "half_aligned_32b": bool(elem_bytes) and (half * elem_bytes) % 32 == 0,
        "outer_bucket": _outer_bucket(outer),
        "prime_outer": _is_prime(outer) if outer <= 10_000_000 else None,
    }


def scope_of(features: dict) -> dict:
    features = normalize_features(features)
    scope = {}
    for key in ("op_family", "op", "soc", "form", "task"):
        values = _as_list(features.get(key))
        if values:
            scope[key] = values
    for key in ("cann", "bisheng", "triton"):
        value = features.get(key)
        if value:
            scope[key] = [str(value)]
    dtypes = _as_list(features.get("dtype"))
    if dtypes:
        scope["dtype"] = dtypes
    sig = features.get("shape_signature") or {}
    shape_scope = {}
    for key in ("rank", "half_is_1", "half_aligned_32b", "outer_bucket", "prime_outer"):
        if key in sig and sig[key] is not None:
            shape_scope[key] = sig[key]
    if shape_scope:
        scope["shape"] = shape_scope
    return scope


def normalize_features(features: dict) -> dict:
    f = dict(features or {})
    out = {
        "op": f.get("op") or f.get("name") or "",
        "op_family": f.get("op_family") or f.get("op_class") or "",
        "soc": f.get("soc") or f.get("hardware") or "",
        "cann": f.get("cann") or "",
        "bisheng": f.get("bisheng") or "",
        "triton": f.get("triton") or "",
        "form": _as_list(f.get("form")),
        "dtype": _as_list(f.get("dtype") or f.get("dtypes")),
        "task": f.get("task") or f.get("name") or "",
    }
    if f.get("shape_signature"):
        out["shape_signature"] = dict(f["shape_signature"])
    elif f.get("shape") is not None:
        sig = shape_signature(f.get("shape"), f.get("dim", -1),
                              (out["dtype"] or [""])[0])
        if sig:
            out["shape_signature"] = sig
    return out


def scope_match(scope: dict, features: dict) -> tuple:
    """Return (matched, reasons, score). Strict by default: missing scope facts reject."""
    features = normalize_features(features)
    reasons = []
    for key in ("op_family", "op", "soc", "form", "task"):
        wanted = _as_list((scope or {}).get(key))
        if not wanted:
            continue
        actual = _as_list(features.get(key))
        if not actual:
            return False, ["missing %s" % key], 0.0
        if not any(_contains(w, a) for w in wanted for a in actual):
            return False, ["%s mismatch: %s vs %s" % (key, actual, wanted)], 0.0
        reasons.append("%s=%s" % (key, ",".join(actual)))
    for key in ("cann", "bisheng", "triton"):
        wanted = _as_list((scope or {}).get(key))
        if not wanted:
            continue
        actual = features.get(key) or ""
        if not actual:
            return False, ["missing %s" % key], 0.0
        if not any(version_ok(str(actual), str(w)) for w in wanted):
            return False, ["%s %s not in %s" % (key, actual, wanted)], 0.0
        reasons.append("%s=%s" % (key, actual))
    dtypes = _as_list((scope or {}).get("dtype"))
    if dtypes:
        actual = _as_list(features.get("dtype"))
        actual_set = {_canonical_dtype(x) for x in actual}
        wanted_set = {_canonical_dtype(x) for x in dtypes}
        if not actual or not (actual_set & wanted_set):
            return False, ["dtype mismatch: %s vs %s" % (actual, dtypes)], 0.0
        reasons.append("dtype=%s" % ",".join(actual))
    shape_scope = (scope or {}).get("shape") or {}
    sig = features.get("shape_signature") or {}
    for key, wanted in shape_scope.items():
        actual = sig.get(key)
        if isinstance(wanted, list):
            if actual not in wanted:
                return False, ["shape.%s=%s not in %s" % (key, actual, wanted)], 0.0
        elif actual != wanted:
            return False, ["shape.%s=%s != %s" % (key, actual, wanted)], 0.0
        reasons.append("shape.%s=%s" % (key, actual))
    return True, reasons, min(1.0, 0.55 + 0.08 * len(reasons))


def _age_days(stamp: str) -> float:
    if not stamp:
        return 365.0
    try:
        value = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 86400.0)
    except ValueError:
        return 365.0


def scope_overlap(a: dict, b: dict) -> bool:
    """Do two scopes potentially describe the same run? (used for conflict detection)"""
    a, b = a or {}, b or {}
    for key in ("op_family", "op", "soc", "form", "task"):
        va, vb = _as_list(a.get(key)), _as_list(b.get(key))
        if not va or not vb:
            continue
        if not any(_contains(x, y) for x in va for y in vb):
            return False
    da = {_canonical_dtype(x) for x in _as_list(a.get("dtype"))}
    db = {_canonical_dtype(x) for x in _as_list(b.get("dtype"))}
    if da and db and not (da & db):
        return False
    for key in ("cann", "bisheng", "triton"):
        va, vb = _as_list(a.get(key)), _as_list(b.get(key))
        if va and vb and not any(version_ok(x, y) or version_ok(y, x) for x in va for y in vb):
            return False
    sa, sb = a.get("shape") or {}, b.get("shape") or {}
    for key in set(sa) & set(sb):
        if sa[key] != sb[key]:
            return False
    return True


# --------------------------------------------------------------------------- store
class MemoryStore:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = os.path.abspath(root or DEFAULT_ROOT)
        self.facts_path = os.path.join(self.root, FACTS_FILE)
        self.failures_path = os.path.join(self.root, FAILURES_FILE)
        self.skills_path = os.path.join(self.root, SKILLS_FILE)
        self.strategies_path = os.path.join(self.root, STRATEGIES_FILE)
        self._tx_depth = 0
        self._load_files()

    def _load_files(self) -> None:
        self.facts = _read_jsonl(self.facts_path)
        self.failures = _read_jsonl(self.failures_path)
        self.skills = _read_jsonl(self.skills_path)

    @contextlib.contextmanager
    def transaction(self):
        """Cross-process read-modify-write.

        Re-entrant: nested add_* calls inside a transaction reuse the outer lock
        and are persisted by the outer save().
        """
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1
            return
        with _FileLock(os.path.join(self.root, ".memory.lock")):
            self._load_files()
            self._tx_depth = 1
            try:
                yield
                self.save()
            finally:
                self._tx_depth = 0

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        _write_jsonl(self.facts_path, sorted(self.facts, key=lambda r: r.get("id", "")))
        _write_jsonl(self.failures_path, sorted(self.failures, key=lambda r: r.get("id", "")))
        _write_jsonl(self.skills_path, sorted(self.skills, key=lambda r: r.get("id", "")))

    @staticmethod
    def _next_id(rows: list, prefix: str) -> str:
        top = 0
        for row in rows:
            match = re.match(r"^%s-(\d+)$" % prefix, str(row.get("id") or ""))
            if match:
                top = max(top, int(match.group(1)))
        return "%s-%04d" % (prefix, top + 1)

    @staticmethod
    def _scope_key(scope: dict) -> str:
        return _hash(json.dumps(scope or {}, sort_keys=True, ensure_ascii=False))

    # ------------------------------------------------------------------ facts
    def add_fact(self, record: dict) -> dict:
        statement = str(record.get("statement") or "").strip()
        if not statement:
            raise ValueError("fact.statement is required")
        scope = dict(record.get("scope") or {})
        key = "fact:" + _hash(_norm(statement) + self._scope_key(scope))
        with self.transaction():
            return self._add_fact_locked(record, statement, scope, key)

    def _add_fact_locked(self, record: dict, statement: str, scope: dict, key: str) -> dict:
        existing = next((r for r in self.facts if r.get("key") == key), None)
        evidence = list(record.get("evidence") or [])
        if existing:
            existing["evidence"] = (existing.get("evidence") or []) + evidence
            existing["evidence"] = existing["evidence"][-20:]
            existing["updated"] = utc_now()
            for field in ("confidence", "authority", "status", "invalidators"):
                if record.get(field):
                    existing[field] = record[field]
            return existing
        row = {
            "id": record.get("id") or self._next_id(self.facts, "fact"),
            "key": key,
            "type": str(record.get("type") or "hardware_constraint"),
            "statement": statement,
            "scope": scope,
            "authority": str(record.get("authority") or "experiment"),
            "confidence": str(record.get("confidence") or "candidate"),
            "status": str(record.get("status") or "active"),
            "evidence": evidence,
            "invalidators": list(record.get("invalidators") or []),
            "created": utc_now(),
            "updated": utc_now(),
        }
        self.facts.append(row)
        return row

    # ------------------------------------------------------------------ failures
    def add_failure(self, record: dict) -> dict:
        stage = str(record.get("stage") or "").strip()
        signature = str(record.get("signature") or "").strip()
        if not stage or not signature:
            raise ValueError("failure.stage and failure.signature are required")
        scope = dict(record.get("scope") or {})
        sig_hash = _hash(_norm(signature))
        key = "fail:" + _hash(stage + sig_hash + self._scope_key(scope))
        with self.transaction():
            return self._add_failure_locked(record, stage, signature, sig_hash, scope, key)

    def _add_failure_locked(self, record: dict, stage: str, signature: str,
                            sig_hash: str, scope: dict, key: str) -> dict:
        existing = next((r for r in self.failures if r.get("key") == key), None)
        evidence = list(record.get("evidence") or [])
        if existing:
            existing["count"] = int(existing.get("count") or 1) + 1
            existing["last_seen"] = utc_now()
            existing["evidence"] = (existing.get("evidence") or []) + evidence
            existing["evidence"] = existing["evidence"][-20:]
            if record.get("workaround"):
                existing["workaround"] = record["workaround"]
            if record.get("root_cause"):
                existing["root_cause"] = record["root_cause"]
            if record.get("status"):
                existing["status"] = record["status"]
            return existing
        row = {
            "id": record.get("id") or self._next_id(self.failures, "fail"),
            "key": key,
            "stage": stage,
            "signature": signature,
            "signature_hash": sig_hash,
            "minimal_repro": dict(record.get("minimal_repro") or {}),
            "scope": scope,
            "root_cause": str(record.get("root_cause") or ""),
            "workaround": str(record.get("workaround") or ""),
            "status": str(record.get("status") or "open"),
            "count": 1,
            "evidence": evidence,
            "first_seen": utc_now(),
            "last_seen": utc_now(),
        }
        self.failures.append(row)
        return row

    def resolve_failures(self, features: dict, run_id: str, note: str = "") -> int:
        features = normalize_features(features)
        with self.transaction():
            resolved = 0
            for row in self.failures:
                if row.get("status") != "open":
                    continue
                matched, _reasons, _score = scope_match(row.get("scope") or {}, features)
                if not matched:
                    continue
                row["status"] = "resolved"
                row["resolved_by"] = run_id
                row["resolved_at"] = utc_now()
                if note:
                    row["resolution"] = note
                resolved += 1
            return resolved

    # ------------------------------------------------------------------ skills
    def add_skill(self, record: dict) -> dict:
        name = str(record.get("name") or "").strip()
        if not name:
            raise ValueError("skill.name is required")
        scope = dict(record.get("scope") or {})
        key = "skill:" + _hash(_norm(name) + self._scope_key(scope))
        with self.transaction():
            return self._add_skill_locked(record, name, scope, key)

    def _add_skill_locked(self, record: dict, name: str, scope: dict, key: str) -> dict:
        existing = next((r for r in self.skills if r.get("key") == key), None)
        if existing:
            for field in ("template", "template_ref", "parameters", "validation",
                          "measured", "confidence", "invalidators", "preconditions"):
                if record.get(field):
                    existing[field] = record[field]
            existing["updated"] = utc_now()
            return existing
        row = {
            "id": record.get("id") or self._next_id(self.skills, "sk"),
            "key": key,
            "name": name,
            "language": str(record.get("language") or "python"),
            "scope": scope,
            "preconditions": list(record.get("preconditions") or []),
            "parameters": list(record.get("parameters") or []),
            "template": str(record.get("template") or ""),
            "template_ref": str(record.get("template_ref") or ""),
            "validation": dict(record.get("validation") or {}),
            "measured": dict(record.get("measured") or {}),
            "confidence": str(record.get("confidence") or "candidate"),
            "invalidators": list(record.get("invalidators") or []),
            "created": utc_now(),
            "updated": utc_now(),
        }
        self.skills.append(row)
        return row

    # ------------------------------------------------------------------ lifecycle
    def decay(self, days: dict | None = None, apply: bool = False) -> dict:
        """Mark records that have not been verified recently.

        Dry-run by default.  Raw episodes are never touched; only the derived
        memory records get a stale/expired status, so they rank lower but remain
        retrievable for inspection.
        """
        days = dict({"facts": 180, "failures": 90, "skills": 60}, **(days or {}))
        report = {"dry_run": not apply, "would_mark": [], "marked": []}
        with _FileLock(os.path.join(self.root, ".memory.lock")):
            self._load_files()
            for kind, rows in (("facts", self.facts), ("failures", self.failures),
                               ("skills", self.skills)):
                threshold = float(days[kind])
                for row in rows:
                    if row.get("status") in ("stale", "expired", "obsolete", "resolved"):
                        continue
                    stamp = row.get("updated") or row.get("last_seen") or row.get("created") or ""
                    age = _age_days(stamp)
                    if age <= threshold:
                        continue
                    item = {"kind": kind, "id": row.get("id"),
                            "age_days": round(age, 1), "threshold": threshold}
                    if apply:
                        row["status"] = "expired" if kind == "failures" else "stale"
                        row["decayed_at"] = utc_now()
                        row["decay_reason"] = "not verified in %.0f days" % age
                        report["marked"].append(item)
                    else:
                        report["would_mark"].append(item)
            if apply:
                self.save()
        return report

    def verify(self, record_id: str, note: str = "", confidence: str = "") -> dict | None:
        """Re-validate a record after an experiment/doc check; clears its stale status."""
        with _FileLock(os.path.join(self.root, ".memory.lock")):
            self._load_files()
            for rows, kind in ((self.facts, "fact"), (self.failures, "failure"),
                               (self.skills, "skill")):
                for row in rows:
                    if row.get("id") != record_id:
                        continue
                    row["last_verified"] = utc_now()
                    row["updated"] = utc_now()
                    row["status"] = "open" if kind == "failure" else "active"
                    row.pop("decayed_at", None)
                    row.pop("decay_reason", None)
                    if note:
                        row["verify_note"] = note
                    if confidence:
                        row["confidence"] = confidence
                    self.save()
                    return row
        return None

    def promote_skill_from_run(self, run_dir: str, name: str, template_ref: str,
                               features: dict, preconditions=None, parameters=None,
                               validation=None, measured=None, language: str = "") -> dict:
        """Promote a passing run into procedural memory.

        Requires (a) best status pass, (b) real-device provenance, (c) template
        exists.  First pass creates a candidate; the second independent pass
        promotes it to verified.
        """
        meta_path = os.path.join(run_dir, "meta.json")
        if not os.path.isfile(meta_path):
            return {"status": "skipped", "reason": "meta.json not found"}
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        best = meta.get("best") or {}
        if best.get("status") != "pass":
            return {"status": "skipped", "reason": "best status is not pass"}
        from .provenance import provenance
        prov = provenance({"device": meta.get("device"), "hardware": meta.get("hardware")})
        if not prov.get("valid"):
            return {"status": "skipped", "reason": "provenance: %s" % prov.get("reason", "")}
        if not name or not template_ref:
            return {"status": "skipped", "reason": "name and template_ref are required"}
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = template_ref if os.path.isabs(template_ref) else os.path.join(project, template_ref)
        if not os.path.isfile(path):
            return {"status": "skipped", "reason": "template_ref not found: %s" % template_ref}

        run_id = meta.get("run_id") or os.path.basename(os.path.normpath(run_dir))
        scope = scope_of(features)
        # skills should transfer across tasks with the same op/soc/form/dtype; keep
        # the task id out of the skill scope (facts/failures may keep it).
        scope.pop("task", None)
        with self.transaction():
            existing = next((r for r in self.skills
                             if r.get("name") == name and r.get("scope") == scope), None)
            runs = list((((existing or {}).get("validation") or {}).get("runs")) or [])
            if run_id not in runs:
                runs.append(run_id)
            passes = len(runs)
            skill = {
                "name": name,
                "language": language or ("ascendc" if "ascendc" in scope.get("form", []) else "python"),
                "scope": scope,
                "preconditions": list(preconditions or []),
                "parameters": list(parameters or []),
                "template": "",
                "template_ref": template_ref,
                "validation": dict(validation or {}, runs=runs, passes=passes, last_run=run_id),
                "measured": dict(measured or ({"speedup": best.get("speedup")}
                                               if best.get("speedup") is not None else {})),
                "confidence": "verified" if passes >= 2 else "candidate",
                "invalidators": [],
            }
            key = "skill:" + _hash(_norm(name) + self._scope_key(scope))
            row = self._add_skill_locked(skill, name, scope, key)
        return {"status": "upserted", "skill": row, "passes": passes,
                "confidence": row.get("confidence")}

    # ------------------------------------------------------------------ extraction
    def record_failure_from_result(self, result: dict, features: dict,
                                   run_id: str, task_id: str = "") -> dict | None:
        status = str(result.get("status") or "")
        if status == "pass":
            return None
        stage, signature = "runtime", status
        minimal = {"task": task_id or (normalize_features(features).get("task") or "")}
        if status == "build_error":
            stage = "build"
            signature = _first_error_line((result.get("build") or {}).get("tail") or "") or "build failed"
        elif status == "fail_correctness":
            stage = "correctness"
            failed = [c for c in (result.get("cases") or []) if not c.get("passed")]
            summary = result.get("case_summary") or {}
            signature = "fail_correctness %s/%s cases" % (
                summary.get("failed", len(failed)), summary.get("total", len(result.get("cases") or [])))
            minimal["failed_cases"] = [
                {"shape": c.get("shape"), "dtype": c.get("dtype"), "dim": c.get("dim")}
                for c in failed[:5]
            ]
        elif status == "crash":
            stage = "runtime"
            signature = _first_error_line((result.get("test") or {}).get("tail") or "") or "test crash"
        elif status == "timeout":
            stage = "timeout"
            signature = "timeout"
        elif status == "bench_error":
            stage = "bench"
            signature = _first_error_line((result.get("bench") or {}).get("tail") or "") or "bench failed"
        return self.add_failure({
            "stage": stage,
            "signature": signature,
            "minimal_repro": minimal,
            "scope": scope_of(features),
            "status": "open",
            "evidence": [{
                "run_id": run_id,
                "task": task_id or normalize_features(features).get("task"),
                "log": _log_path(result, stage),
                "at": utc_now(),
            }],
        })

    # ------------------------------------------------------------------ retrieval
    def _strategy_rows(self) -> list:
        rows = []
        for row in _read_jsonl(self.strategies_path):
            cond = row.get("conditions") or {}
            scope = {}
            if cond.get("op_class"):
                scope["op_family"] = cond["op_class"]
            if cond.get("hardware"):
                scope["soc"] = cond["hardware"]
            if cond.get("dtypes"):
                scope["dtype"] = cond["dtypes"]
            rows.append({
                "id": row.get("id"),
                "kind": "strategy",
                "record": row,
                "scope": scope,
                "confidence": row.get("confidence") or "seed",
                "updated": row.get("updated") or row.get("created") or "",
            })
        return rows

    def retrieve(self, features: dict, top_k: int = 3, kinds=None) -> list:
        features = normalize_features(features)
        kinds = set(kinds) if kinds else None
        hits = []

        def add(row, kind, confidence, updated, status="", measured=None, authority="", record=None):
            record = record or row
            matched, reasons, match_score = scope_match(row.get("scope") or {}, features)
            if not matched:
                return
            score = TYPE_WEIGHT[kind] * (0.55 + 0.45 * match_score)
            score *= CONFIDENCE_WEIGHT.get(confidence, 0.4)
            score *= math.exp(-_age_days(updated) / 365.0)
            if kind == "fact":
                score *= AUTHORITY_WEIGHT.get(authority, 0.5)
                if status and status != "active":
                    score *= 0.3
            elif kind == "failure":
                score *= 1.0 if status == "open" else (0.55 if status == "resolved" else 0.1)
                score *= 1.0 + min(0.4, math.log1p(int(record.get("count") or 1)) / 5.0)
            elif kind == "skill":
                if measured:
                    score *= 1.0 + min(0.5, max(0.0, float(measured) - 1.0) / 2.0)
            elif kind == "strategy":
                stats = (record.get("stats") or {})
                applied = int(stats.get("applied") or 0)
                if applied:
                    score *= 1.0 + 0.2 * (int(stats.get("succeeded") or 0) / applied)
            hits.append({
                "type": kind,
                "id": record.get("id") or row.get("id"),
                "record": record,
                "score": round(score, 4),
                "match": round(match_score, 3),
                "why": reasons,
            })

        for row in self.facts:
            add(row, "fact", row.get("confidence") or "candidate",
                row.get("updated") or row.get("created") or "",
                status=row.get("status") or "", authority=row.get("authority") or "")
        for row in self.failures:
            add(row, "failure", "verified" if int(row.get("count") or 1) >= 2 else "candidate",
                row.get("last_seen") or row.get("first_seen") or "",
                status=row.get("status") or "")
        for row in self.skills:
            add(row, "skill", row.get("confidence") or "candidate",
                row.get("updated") or row.get("created") or "",
                measured=(row.get("measured") or {}).get("speedup"))
        for row in self._strategy_rows():
            add(row, "strategy", row.get("confidence") or "seed",
                row.get("updated") or "", measured=None, record=row.get("record"))

        hits.sort(key=lambda h: -h["score"])
        per_type = {}
        limits = {"fact": 16, "failure": 4, "skill": 3, "strategy": 4}
        out = []
        for hit in hits:
            if kinds and hit["type"] not in kinds:
                continue
            limit = limits.get(hit["type"], top_k)
            per_type.setdefault(hit["type"], 0)
            if per_type[hit["type"]] >= limit:
                continue
            per_type[hit["type"]] += 1
            out.append(hit)
        return out

    def search_text(self, query: str, top_k: int = 5, kinds=None) -> list:
        """Free-text search over memory, e.g. an agent's on-demand memory query.

        Results are NOT scope-filtered: the caller must check the record's scope
        before applying them.  This is deliberately separate from ``retrieve`` so
        a text hit can never silently count as condition-matched priors.
        """
        query = str(query or "").strip()
        if not query:
            return []
        tokens = _text_tokens(query)
        if not tokens:
            return []
        kinds = set(kinds) if kinds else None
        hits = []

        def add(row, kind, text, status="", updated=""):
            if kinds and kind not in kinds:
                return
            hay = _text_tokens(text)
            if not hay:
                return
            shared = tokens & hay
            if not shared:
                return
            score = len(shared) / len(tokens | hay)
            if query.lower() in str(text).lower():
                score += 0.25
            hits.append({
                "type": kind,
                "id": row.get("id"),
                "record": row,
                "score": round(score, 4),
                "status": status,
                "why": ["text: " + ", ".join(sorted(shared)[:8])],
            })

        for row in self.facts:
            add(row, "fact", " ".join(str(row.get(k) or "") for k in ("statement", "key", "authority")),
                status=row.get("status") or "")
        for row in self.failures:
            add(row, "failure", " ".join(str(row.get(k) or "") for k in
                                         ("signature", "stage", "workaround", "minimal_repro")),
                status=row.get("status") or "")
        for row in self.skills:
            add(row, "skill", " ".join(str(row.get(k) or "") for k in ("name", "preconditions", "language")),
                status=row.get("confidence") or "")
        for row in self._strategy_rows():
            rec = row.get("record") or {}
            add(rec, "strategy", " ".join(str(rec.get(k) or "") for k in ("statement", "key")),
                status=rec.get("confidence") or "")
        hits.sort(key=lambda h: -h["score"])
        return hits[:max(1, int(top_k))]

    def context(self, features: dict, top_k: int = 3, max_chars: int = 10000, kinds=None) -> str:
        hits = self.retrieve(features, top_k=top_k, kinds=kinds)
        features = normalize_features(features)
        if not hits:
            return "## Operator memory\n(no fact/failure/skill/strategy matched this scope)"
        lines = ["## Operator memory (scope-matched; apply only what matches the current run)"]
        labels = {"fact": "FACT", "failure": "FAILURE", "skill": "SKILL", "strategy": "STRATEGY"}
        for hit in hits:
            record = hit["record"]
            kind = hit["type"]
            what = (record.get("statement") or record.get("signature")
                    or record.get("name") or record.get("statement") or "")
            if kind == "strategy":
                what = record.get("statement") or ""
            if len(what) > 500:
                what = what[:500] + "…"
            lines.append("- [%s %s] (%s) %s" % (
                labels[kind], hit["id"],
                record.get("confidence") or record.get("status") or "candidate", what))
            lines.append("    why: %s" % ", ".join(hit["why"]))
            if kind == "fact" and record.get("authority"):
                lines.append("    authority: %s" % record["authority"])
            if kind == "failure":
                repro = record.get("minimal_repro") or {}
                if repro:
                    lines.append("    repro: %s" % json.dumps(repro, ensure_ascii=False)[:240])
                if record.get("workaround"):
                    lines.append("    workaround: %s" % str(record["workaround"])[:240])
            if kind == "skill":
                measured = record.get("measured") or {}
                if measured:
                    lines.append("    measured: %s" % json.dumps(measured, ensure_ascii=False)[:200])
                if record.get("template_ref"):
                    lines.append("    template: `%s`" % record["template_ref"])
            text = "\n".join(lines)
            if len(text) > max_chars:
                break
        return "\n".join(lines)

    # ------------------------------------------------------------------ audit
    def conflicts(self) -> dict:
        """Find evidence that disagrees under overlapping scopes.

        This is a report, not an auto-merge: the caller should split the scope
        (dtype/shape/form) or run an A/B ablation to resolve it.
        """
        strategy_conflicts = []
        for row in _read_jsonl(self.strategies_path):
            groups = {}
            for ev in row.get("evidence") or []:
                cond = (str(ev.get("hardware") or ""), str(ev.get("dtype") or ""))
                g = groups.setdefault(cond, {"success": 0, "failure": 0, "neutral": 0})
                outcome = str(ev.get("outcome") or "neutral")
                g[outcome] = g.get(outcome, 0) + 1
            bad = [{"hardware": k[0], "dtype": k[1], **v}
                   for k, v in groups.items() if v.get("success") and v.get("failure")]
            if bad:
                strategy_conflicts.append({
                    "id": row.get("id"), "key": row.get("key"),
                    "statement": row.get("statement"), "groups": bad,
                })

        skill_conflicts = []
        by_name = {}
        for skill in self.skills:
            by_name.setdefault(skill.get("name"), []).append(skill)
        for name, rows in by_name.items():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if not scope_overlap(rows[i].get("scope"), rows[j].get("scope")):
                        continue
                    conf = {rows[i].get("confidence"), rows[j].get("confidence")}
                    if "verified" in conf and "refuted" in conf:
                        skill_conflicts.append({
                            "name": name, "a": rows[i].get("id"), "b": rows[j].get("id"),
                        })

        failure_vs_skill = []
        for failure in self.failures:
            if failure.get("status") != "open":
                continue
            for skill in self.skills:
                if skill.get("confidence") != "verified":
                    continue
                if scope_overlap(failure.get("scope"), skill.get("scope")):
                    failure_vs_skill.append({
                        "failure": failure.get("id"),
                        "signature": failure.get("signature"),
                        "skill": skill.get("id"),
                        "skill_name": skill.get("name"),
                    })
        return {
            "strategies": strategy_conflicts,
            "skills": skill_conflicts,
            "failure_vs_skill": failure_vs_skill,
            "total": len(strategy_conflicts) + len(skill_conflicts) + len(failure_vs_skill),
        }

    def audit(self) -> dict:
        open_failures = [f for f in self.failures if f.get("status") == "open"]
        verified_skills = [s for s in self.skills if s.get("confidence") == "verified"]
        stale = [r for r in self.facts + self.failures + self.skills
                 if _age_days(r.get("updated") or r.get("last_seen")
                              or r.get("created") or "") > 90]
        disputed = [r for r in self.facts + self.skills
                    if r.get("status") in ("disputed", "refuted")
                    or r.get("confidence") == "refuted"]
        return {
            "root": self.root,
            "facts": len(self.facts),
            "facts_active": sum(1 for r in self.facts if r.get("status", "active") == "active"),
            "failures": len(self.failures),
            "failures_open": len(open_failures),
            "skills": len(self.skills),
            "skills_verified": len(verified_skills),
            "strategies": len(_read_jsonl(self.strategies_path)),
            "stale": len(stale),
            "disputed_or_refuted": len(disputed),
            "conflicts": self.conflicts()["total"],
            "open_failure_ids": [r.get("id") for r in open_failures[:10]],
        }


def _first_error_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"error|Error|ERROR|FAILED|fatal|undefined reference|abort|Exception", stripped):
            return stripped[:300]
    return ""


def _log_path(result: dict, stage: str) -> str:
    key = {"build": "build", "correctness": "test", "runtime": "test",
           "timeout": "test", "bench": "bench"}.get(stage, "test")
    return str((result.get(key) or {}).get("log_path") or "")


# --------------------------------------------------------------------------- cli
def _features_from_args(args) -> dict:
    return {
        "op": args.op, "op_family": args.op_family, "soc": args.soc,
        "cann": args.cann, "bisheng": args.bisheng, "triton": args.triton,
        "dtype": [x.strip() for x in args.dtype.split(",") if x.strip()],
        "form": [x.strip() for x in args.form.split(",") if x.strip()],
        "task": args.task,
        "shape": args.shape, "dim": args.dim,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.memory", description="Operator memory: facts/failures/skills.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--json", action="store_true")

    c = sub.add_parser("context")
    c.add_argument("--op", default="")
    c.add_argument("--op-family", default="")
    c.add_argument("--soc", default="")
    c.add_argument("--cann", default="")
    c.add_argument("--bisheng", default="")
    c.add_argument("--triton", default="")
    c.add_argument("--dtype", default="")
    c.add_argument("--form", default="")
    c.add_argument("--task", default="")
    c.add_argument("--shape", default="")
    c.add_argument("--dim", type=int, default=-1)
    c.add_argument("--json", action="store_true")

    for name in ("add-fact", "add-failure", "add-skill"):
        p = sub.add_parser(name)
        p.add_argument("--file", required=True)

    d = sub.add_parser("distill-run")
    d.add_argument("--run-dir", required=True)
    d.add_argument("--op", default="")
    d.add_argument("--op-family", default="")
    d.add_argument("--soc", default="")
    d.add_argument("--cann", default="")
    d.add_argument("--bisheng", default="")
    d.add_argument("--triton", default="")
    d.add_argument("--dtype", default="")
    d.add_argument("--form", default="")
    d.add_argument("--task", default="")
    d.add_argument("--shape", default="")
    d.add_argument("--dim", type=int, default=-1)

    dec = sub.add_parser("decay")
    dec.add_argument("--apply", action="store_true")
    dec.add_argument("--facts-days", type=float, default=180)
    dec.add_argument("--failures-days", type=float, default=90)
    dec.add_argument("--skills-days", type=float, default=60)

    ver = sub.add_parser("verify")
    ver.add_argument("--id", required=True)
    ver.add_argument("--note", default="")
    ver.add_argument("--confidence", default="")

    conf = sub.add_parser("conflicts")
    conf.add_argument("--json", action="store_true")

    prom = sub.add_parser("promote-skill")
    prom.add_argument("--run-dir", required=True)
    prom.add_argument("--name", required=True)
    prom.add_argument("--template-ref", required=True)
    prom.add_argument("--language", default="")
    prom.add_argument("--preconditions-json", default="")
    prom.add_argument("--parameters-json", default="")
    prom.add_argument("--validation-json", default="")
    prom.add_argument("--measured-json", default="")
    prom.add_argument("--op", default="")
    prom.add_argument("--op-family", default="")
    prom.add_argument("--soc", default="")
    prom.add_argument("--cann", default="")
    prom.add_argument("--bisheng", default="")
    prom.add_argument("--triton", default="")
    prom.add_argument("--dtype", default="")
    prom.add_argument("--form", default="")
    prom.add_argument("--task", default="")
    prom.add_argument("--shape", default="")
    prom.add_argument("--dim", type=int, default=-1)

    args = ap.parse_args(argv)
    store = MemoryStore(args.root)

    if args.cmd == "audit":
        result = store.audit()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("# Operator memory audit")
            for key, value in result.items():
                if key == "open_failure_ids":
                    print("open failures: %s" % (", ".join(value) or "-"))
                else:
                    print("%-22s %s" % (key, value))
        return 0

    if args.cmd == "context":
        features = _features_from_args(args)
        if args.json:
            print(json.dumps({"features": normalize_features(features),
                              "hits": store.retrieve(features)}, ensure_ascii=False, indent=2))
        else:
            print(store.context(features))
        return 0

    if args.cmd.startswith("add-"):
        with open(args.file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if args.cmd == "add-fact":
                record = store.add_fact(row)
            elif args.cmd == "add-failure":
                record = store.add_failure(row)
            else:
                record = store.add_skill(row)
            print("upserted %s" % record.get("id"))
        return 0

    if args.cmd == "distill-run":
        meta_path = os.path.join(args.run_dir, "meta.json")
        attempts_path = os.path.join(args.run_dir, "attempts.jsonl")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        features = _features_from_args(args)
        if not features.get("op") and meta.get("task"):
            features["op"] = meta["task"]
        if not features.get("soc") and meta.get("hardware"):
            features["soc"] = meta["hardware"]
        if not features.get("form") and meta.get("soc", "").startswith("ascend"):
            features["form"] = ["ascendc"]
        attempts = _read_jsonl(attempts_path)
        count = 0
        for attempt in attempts:
            status = str(attempt.get("status") or "")
            if status == "pass" or not status:
                continue
            pseudo = {
                "status": status,
                "case_summary": attempt.get("case_summary") or {},
                "cases": attempt.get("cases") or [],
                "build": {"tail": attempt.get("error") or ""},
                "test": {"tail": attempt.get("error") or ""},
            }
            row = store.record_failure_from_result(
                pseudo, features, str(attempt.get("task") or meta.get("task") or ""),
                str(meta.get("task") or ""))
            count += 1 if row else 0
        print("distilled failures: %d (attempts=%d)" % (count, len(attempts)))
        return 0

    if args.cmd == "decay":
        report = store.decay({"facts": args.facts_days, "failures": args.failures_days,
                              "skills": args.skills_days}, apply=args.apply)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "verify":
        row = store.verify(args.id, note=args.note, confidence=args.confidence)
        if not row:
            print("record not found: %s" % args.id)
            return 1
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "conflicts":
        report = store.conflicts()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["total"] == 0 else 1

    if args.cmd == "promote-skill":
        features = _features_from_args(args)
        def _json_arg(text, default):
            return json.loads(text) if text else default
        result = store.promote_skill_from_run(
            args.run_dir, args.name, args.template_ref, features,
            preconditions=_json_arg(args.preconditions_json, []),
            parameters=_json_arg(args.parameters_json, []),
            validation=_json_arg(args.validation_json, {}),
            measured=_json_arg(args.measured_json, {}),
            language=args.language,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "upserted" else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())