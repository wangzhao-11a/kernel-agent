"""CannBench AscendC backend: build a package, run official cases, return structured results.

This is the bridge the Python/triton loop was missing:

    scaffold package + edited files
        -> build.sh --soc=<soc> (or any build command)
        -> official case runner (prints per-case PASS/FAIL)
        -> optional benchmark (prints CANNBENCH_METRICS {...})
        -> JSON result consumed by agent.cannbench_loop and rsi.evolve

The backend is deliberately command-driven.  It does not know how AscendC is
compiled or how a wheel is installed; the spec authorizes the commands, exactly
like the Python harness authorizes `run_eval`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from .policy import cpp_policy_findings, normalize_policy

DEFAULT_TIMEOUT = 900
CASE_RE = re.compile(
    r"^\s*(?P<shape>\([^)]*\))\s+(?P<dtype>[A-Za-z0-9_]+)\s+dim=(?P<dim>-?\d+)\s+"
    r"MERE=(?P<mere>[0-9.eE+\-]+)\s+MARE=(?P<mare>[0-9.eE+\-]+)\s+(?P<status>PASS|FAIL)\s*$"
)
VERDICT_RE = re.compile(r"^[A-Za-z0-9_]*TEST\s+(?P<status>PASS|FAIL)\s*$", re.MULTILINE)
METRICS_RE = re.compile(r"CANNBENCH_METRICS\s+(?P<json>\{.*\})\s*$", re.MULTILINE)
PROFILE_RE = re.compile(r"CANNBENCH_PROFILE\s+(?P<json>\{.*\})\s*$", re.MULTILINE)
SPEEDUP_RE = re.compile(r"^\s*SPEEDUP\s+(?P<value>[0-9.]+)\s*$", re.MULTILINE)


class CannBenchError(ValueError):
    pass


@dataclass
class CannBenchSpec:
    id: str
    package: str
    edit_files: list
    build_command: object
    test_command: object
    bench_command: object = None
    profile_command: object = None
    timeout_s: float = DEFAULT_TIMEOUT
    soc: str = ""
    hardware: str = ""
    baseline: str = ""
    description: str = ""
    metric_key: str = "speedup"
    expected_speedup: float = 1.0
    metadata: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "package": self.package, "edit_files": self.edit_files,
            "build_command": self.build_command, "test_command": self.test_command,
            "bench_command": self.bench_command, "profile_command": self.profile_command,
            "timeout_s": self.timeout_s,
            "soc": self.soc, "hardware": self.hardware, "baseline": self.baseline,
            "description": self.description, "metric_key": self.metric_key,
            "expected_speedup": self.expected_speedup, "metadata": self.metadata,
            "policy": self.policy, "path": self.path,
        }


def _cmd_text(command) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(x) for x in command)
    return str(command or "")


def load_spec(path: str) -> CannBenchSpec:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise CannBenchError("cannot read spec %s: %s" % (path, exc))
    except ValueError as exc:
        raise CannBenchError("spec %s is not valid JSON: %s" % (path, exc))
    if not isinstance(data, dict):
        raise CannBenchError("spec must be a JSON object")
    for key in ("id", "package", "build_command", "test_command"):
        if not data.get(key):
            raise CannBenchError("spec.%s is required" % key)
    if not data.get("edit_files"):
        raise CannBenchError("spec.edit_files must list the files the agent may edit")

    base_dir = os.path.dirname(os.path.abspath(path))
    package = os.path.expandvars(os.path.expanduser(str(data["package"])))
    if not os.path.isabs(package):
        package = os.path.abspath(os.path.join(base_dir, package))
    if not os.path.isdir(package):
        raise CannBenchError("package directory does not exist: %s" % package)

    edit_files = []
    for rel in data["edit_files"]:
        rel = str(rel).replace("\\", "/").strip("/")
        if not rel or rel.startswith("../") or "/../" in rel:
            raise CannBenchError("edit_files must stay inside the package: %r" % rel)
        if not os.path.isfile(os.path.join(package, rel)):
            raise CannBenchError("edit file does not exist in package: %s" % rel)
        edit_files.append(rel)

    try:
        timeout_s = float(data.get("timeout_s", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        raise CannBenchError("timeout_s must be numeric")
    if timeout_s <= 0:
        raise CannBenchError("timeout_s must be > 0")

    raw_policy = data.get("policy")
    if raw_policy is None:
        raw_policy = {
            "forbid_native_compute": True, "require_kernel": True,
            "torch_fused_exempt": False,
            "reason": "CannBench AscendC package: compute must be in the kernel",
        }
    policy = normalize_policy({"policy": raw_policy})

    return CannBenchSpec(
        id=str(data["id"]),
        package=package,
        edit_files=edit_files,
        build_command=data["build_command"],
        test_command=data["test_command"],
        bench_command=data.get("bench_command"),
        profile_command=data.get("profile_command"),
        timeout_s=timeout_s,
        soc=str(data.get("soc") or ""),
        hardware=str(data.get("hardware") or ""),
        baseline=str(data.get("baseline") or ""),
        description=str(data.get("description") or ""),
        metric_key=str(data.get("metric_key") or "speedup"),
        expected_speedup=float(data.get("expected_speedup", 1.0) or 1.0),
        metadata=dict(data.get("metadata") or {}),
        policy=policy,
        path=os.path.abspath(path),
    )


# --------------------------------------------------------------------------- process
def _split(command):
    if isinstance(command, (list, tuple)):
        return [str(x) for x in command]
    return shlex.split(str(command), posix=True)


def _kill_tree(process) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=20)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _run(command, cwd: str, timeout_s: float, log_path: str, env=None) -> dict:
    args = _split(command)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    started = time.time()
    timed_out = False
    returncode = None
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            process = subprocess.Popen(
                args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True,
                start_new_session=(os.name != "nt"),
                env={**os.environ, **env} if env else None,
            )
        except OSError as exc:
            log.write("failed to start: %s\n" % exc)
            return {
                "args": args, "returncode": 127, "timed_out": False,
                "duration_s": 0.0, "log_path": log_path,
                "tail": "failed to start: %s" % exc,
            }
        try:
            returncode = process.wait(timeout=max(1.0, float(timeout_s)))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(process)
            try:
                process.wait(timeout=10)
            except Exception:
                pass
            returncode = -9
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        text = ""
    return {
        "args": args,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": round(time.time() - started, 3),
        "log_path": log_path,
        "tail": text[-4000:],
    }


# --------------------------------------------------------------------------- parse
def parse_cases(text: str) -> list:
    cases = []
    for line in (text or "").splitlines():
        match = CASE_RE.match(line)
        if not match:
            continue
        cases.append({
            "shape": match.group("shape"),
            "dtype": match.group("dtype"),
            "dim": int(match.group("dim")),
            "mere": float(match.group("mere")),
            "mare": float(match.group("mare")),
            "passed": match.group("status") == "PASS",
        })
    return cases


def parse_verdict(text: str) -> str:
    matches = VERDICT_RE.findall(text or "")
    return matches[-1] if matches else ""


def parse_metrics(text: str, nonce: str = None) -> dict:
    """Read the last metrics line, optionally requiring the run nonce.

    The bench script runs in the same process that loads the candidate plugin, so
    anything the candidate prints also lands in the log.  A per-run nonce (passed
    to the bench via the environment) makes a forged line easy to spot: only lines
    carrying this run's nonce are accepted, and the *last* one wins so the real
    bench output supersedes earlier noise.
    """
    body = text or ""
    matches = list(METRICS_RE.finditer(body))
    if nonce:
        matches = [m for m in matches if nonce in m.group("json")]
    for match in reversed(matches):
        try:
            payload = json.loads(match.group("json"))
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    if nonce:
        return {}
    match = SPEEDUP_RE.search(body)
    if match:
        return {"speedup": float(match.group("value"))}
    return {}


def read_metrics_file(path: str) -> dict:
    """Metrics written by the bench to the harness-owned file."""
    if not path or not os.path.isfile(path):
        return {}
    text = _read_text(path)
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        return payload
    return parse_metrics(text)


# --------------------------------------------------------------------------- workspace
def _ignore(_dir, names):
    return [n for n in names if n in (".git", "__pycache__", "dist", "build", ".pytest_cache")
            or n.endswith(".egg-info")]


def prepare_workspace(spec: CannBenchSpec, work_root: str, files: dict | None = None) -> str:
    workspace = os.path.join(work_root, "workspace")
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(work_root, exist_ok=True)
    shutil.copytree(spec.package, workspace, ignore=_ignore)
    for rel, content in (files or {}).items():
        rel = str(rel).replace("\\", "/").strip("/")
        if rel not in spec.edit_files:
            raise CannBenchError("agent may only edit %s; got %r" % (spec.edit_files, rel))
        target = os.path.join(workspace, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(str(content))
    return workspace


# --------------------------------------------------------------------------- run
def run_cannbench(spec: CannBenchSpec, files: dict | None = None,
                  work_root: str | None = None, run_id: str = "",
                  bench_rounds: int = 1) -> dict:
    run_root = work_root or tempfile.mkdtemp(prefix="cannbench-")
    logs = os.path.join(run_root, "logs")
    started = time.time()
    workspace = prepare_workspace(spec, run_root, files)
    result = {
        "spec_id": spec.id,
        "run_id": run_id or os.path.basename(run_root),
        "soc": spec.soc,
        "hardware": spec.hardware,
        "package": spec.package,
        "workspace": workspace,
        "files": sorted((files or {}).keys()),
        "policy": spec.policy,
        "policy_findings": [],
        "build": None,
        "test": None,
        "bench": None,
        "cases": [],
        "case_summary": {"passed": 0, "failed": 0, "total": 0},
        "timing": {},
        "status": "",
        "error": "",
    }
    if spec.policy.get("forbid_native_compute"):
        findings = cpp_policy_findings(files or {})
        result["policy_findings"] = findings
        errors = [f for f in findings if f.get("severity") == "error"]
        if errors:
            result["status"] = "fail_policy"
            result["error"] = "; ".join(f.get("message", "") for f in errors)
            return _finish_run(result, started)
    build = _run(spec.build_command, workspace, spec.timeout_s,
                 os.path.join(logs, "build.log"))
    result["build"] = build
    if build["timed_out"]:
        result["status"] = "timeout"
        result["error"] = "build timed out after %.0fs" % spec.timeout_s
        return _finish_run(result, started)
    if build["returncode"] != 0:
        result["status"] = "build_error"
        result["error"] = "build exited %s" % build["returncode"]
        return _finish_run(result, started)

    test = _run(spec.test_command, workspace, spec.timeout_s,
                os.path.join(logs, "test.log"))
    result["test"] = test
    test_text = _read_text(test["log_path"])
    cases = parse_cases(test_text)
    verdict = parse_verdict(test_text)
    result["cases"] = cases
    result["case_summary"] = {
        "passed": sum(1 for c in cases if c["passed"]),
        "failed": sum(1 for c in cases if not c["passed"]),
        "total": len(cases),
    }
    if test["timed_out"]:
        result["status"] = "timeout"
        result["error"] = "test timed out after %.0fs" % spec.timeout_s
        return _finish_run(result, started)
    if cases and result["case_summary"]["failed"]:
        result["status"] = "fail_correctness"
        result["error"] = "%d/%d cases failed" % (
            result["case_summary"]["failed"], result["case_summary"]["total"])
        return _finish_run(result, started)
    if verdict == "FAIL":
        result["status"] = "fail_correctness"
        result["error"] = "official test verdict FAIL"
        return _finish_run(result, started)
    if test["returncode"] != 0:
        result["status"] = "crash"
        result["error"] = "test exited %s without a passing verdict" % test["returncode"]
        return _finish_run(result, started)
    if not cases and verdict != "PASS":
        result["status"] = "crash"
        result["error"] = "test produced no case table and no PASS verdict"
        return _finish_run(result, started)

    if spec.bench_command:
        nonce = os.urandom(8).hex()
        metrics_path = os.path.join(logs, "bench_metrics.json")
        bench = _run(spec.bench_command, workspace, spec.timeout_s,
                     os.path.join(logs, "bench.log"),
                     env={"CANNBENCH_NONCE": nonce,
                          "CANNBENCH_METRICS_FILE": metrics_path})
        result["bench"] = bench
        result["bench_nonce"] = nonce
        if bench["timed_out"]:
            result["status"] = "bench_error"
            result["error"] = "bench timed out"
            return _finish_run(result, started)
        # Read the payload even when the bench exited non-zero: a bench that also
        # checks correctness returns 1 for "wrong result", which is a correctness
        # failure for the candidate, not an infrastructure error.  The agent needs
        # the failing shapes, not "bench exited 1".
        metrics = read_metrics_file(metrics_path) or parse_metrics(
            _read_text(bench["log_path"]), nonce=nonce)
        bad_cases = [row for row in (metrics.get("cases") or [])
                     if row.get("correct") is False or row.get("passed") is False]
        result["bench_cases_failed"] = len(bad_cases)
        if bad_cases:
            names = ["%s/%s" % (row.get("shape"), row.get("dtype")) for row in bad_cases[:5]]
            result["status"] = "fail_correctness"
            result["error"] = ("bench: %d/%d cases wrong (%s%s)"
                               % (len(bad_cases), len(metrics.get("cases") or []),
                                  ", ".join(names),
                                  " ..." if len(bad_cases) > 5 else ""))
            return _finish_run(result, started)
        if bench["returncode"] != 0:
            result["status"] = "bench_error"
            result["error"] = "bench exited %s" % bench["returncode"]
            return _finish_run(result, started)
        rounds = [{"round": 1, "speedup": metrics.get(spec.metric_key),
                   "duration_s": bench.get("duration_s")}]
        suspect = _metrics_suspect(metrics, bench.get("duration_s") or 0.0, spec.metric_key)
        # Repeated bench invocations: the ABAB/ε gate needs a distribution, and a
        # single command run has no interval at all.
        for index in range(2, max(1, int(bench_rounds)) + 1):
            extra_nonce = os.urandom(8).hex()
            extra_path = os.path.join(logs, "bench_metrics_%02d.json" % index)
            extra = _run(spec.bench_command, workspace, spec.timeout_s,
                         os.path.join(logs, "bench_%02d.log" % index),
                         env={"CANNBENCH_NONCE": extra_nonce,
                              "CANNBENCH_METRICS_FILE": extra_path})
            extra_metrics = read_metrics_file(extra_path) or parse_metrics(
                _read_text(extra["log_path"]), nonce=extra_nonce)
            rounds.append({"round": index, "speedup": extra_metrics.get(spec.metric_key),
                           "duration_s": extra.get("duration_s")})
            suspect = suspect or _metrics_suspect(extra_metrics,
                                                  extra.get("duration_s") or 0.0,
                                                  spec.metric_key)
        values = [row["speedup"] for row in rounds if row.get("speedup") is not None]
        speedups = []
        for value in values:
            try:
                speedups.append(float(value))
            except (TypeError, ValueError):
                continue
        ci = None
        if len(speedups) >= 2:
            from . import stats as stats_mod
            ci = [round(x, 4) for x in stats_mod.bootstrap_ci(speedups)]
        result["timing"] = {
            "speedup": (round(float(statistics.median(speedups)), 4) if speedups
                        else metrics.get(spec.metric_key)),
            "speedup_ci": ci,
            "n_rounds": len(rounds),
            "rounds": rounds,
            "metric_key": spec.metric_key,
            "raw": metrics,
            "measured_by": "cannbench_command",
        }
        result["metrics_suspect"] = suspect
    result["status"] = "pass"
    return _finish_run(result, started)


def scan_native_symbols(workspace: str, limit: int = 12) -> list:
    """Find vendor compute entry points linked into the built extension.

    Text-level scanning misses macro glue and function pointers; the compiled
    shared objects still carry the imported symbol names.
    """
    findings = []
    markers = (b"aclnn", b"torch::matmul", b"torch::softmax", b"at::matmul")
    scanned = 0
    for root, _dirs, names in os.walk(workspace):
        for name in names:
            if not name.endswith((".so", ".pyd")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as fh:
                    blob = fh.read()
            except OSError:
                continue
            scanned += 1
            for marker in markers:
                if marker in blob:
                    findings.append({
                        "severity": "error", "code": "native_compute",
                        "message": "%s links a native compute symbol (%s)"
                                   % (os.path.relpath(path, workspace),
                                      marker.decode("ascii", "replace")),
                    })
                    break
            if scanned >= limit:
                return findings
    return findings


def _metrics_suspect(metrics: dict, bench_wall_s: float, metric_key: str) -> str:
    """Flag a speedup that cannot come from the wall time we actually observed."""
    if not metrics or not bench_wall_s:
        return ""
    duration_ms = metrics.get("duration_ms")
    speedup = metrics.get(metric_key)
    try:
        speedup = float(speedup) if speedup is not None else None
    except (TypeError, ValueError):
        speedup = None
    if speedup is not None and (speedup <= 0 or speedup > 1e4):
        return "speedup %r is out of any plausible range" % speedup
    if duration_ms is not None:
        try:
            duration_ms = float(duration_ms)
        except (TypeError, ValueError):
            return ""
        if duration_ms > 1.5 * bench_wall_s * 1000.0:
            return "reported duration %.1fms exceeds the %.1fs bench wall time" % (
                duration_ms, bench_wall_s)
    return ""


def _finish_run(result: dict, started: float) -> dict:
    result["duration_s"] = round(time.time() - started, 3)
    return result


def parse_profile(text: str) -> dict:
    match = PROFILE_RE.search(text or "")
    if not match:
        return {}
    try:
        payload = json.loads(match.group("json"))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_profile(spec: CannBenchSpec, files: dict | None = None,
                work_root: str | None = None, run_id: str = "",
                env: dict | None = None) -> dict:
    """Build the current files and run the spec's profile_command.

    The command must print one `CANNBENCH_PROFILE {...}` line.  The harness does
    not know which profiler is used (torch_npu/msprof), only the JSON contract.
    """
    run_root = work_root or tempfile.mkdtemp(prefix="cannbench-profile-")
    logs = os.path.join(run_root, "logs")
    started = time.time()
    workspace = prepare_workspace(spec, run_root, files)
    result = {
        "spec_id": spec.id, "run_id": run_id or os.path.basename(run_root),
        "soc": spec.soc, "hardware": spec.hardware, "workspace": workspace,
        "files": sorted((files or {}).keys()), "build": None, "profile": None,
        "metrics": {}, "status": "", "error": "",
    }
    build = _run(spec.build_command, workspace, spec.timeout_s,
                 os.path.join(logs, "build.log"))
    result["build"] = build
    if build["timed_out"]:
        result["status"] = "timeout"
        result["error"] = "build timed out"
        return result
    if build["returncode"] != 0:
        result["status"] = "build_error"
        result["error"] = "build exited %s" % build["returncode"]
        return result
    if not spec.profile_command:
        result["status"] = "profile_error"
        result["error"] = "spec has no profile_command"
        return result
    profile = _run(spec.profile_command, workspace, spec.timeout_s,
                   os.path.join(logs, "profile.log"), env=env)
    result["profile"] = profile
    if profile["timed_out"]:
        result["status"] = "timeout"
        result["error"] = "profile timed out"
        return result
    if profile["returncode"] != 0:
        result["status"] = "profile_error"
        result["error"] = "profile exited %s" % profile["returncode"]
        return result
    metrics = parse_profile(_read_text(profile["log_path"]))
    result["metrics"] = metrics
    if not metrics:
        result["status"] = "profile_error"
        result["error"] = "profile produced no CANNBENCH_PROFILE line"
        return result
    result["status"] = "pass"
    result["duration_s"] = round(time.time() - started, 3)
    return result


def format_profile(result: dict) -> str:
    metrics = result.get("metrics") or {}
    lines = ["Profile: %s | %s" % (result.get("status"), result.get("error") or "ok")]
    if result.get("workspace"):
        lines.append("workspace: %s" % result["workspace"])
    for key in ("shape", "dtype", "dim", "kernel", "count", "duration_us",
                "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
                "aiv_mte3_ratio", "bottleneck"):
        if key in metrics:
            lines.append("%s: %s" % (key, metrics[key]))
    if result.get("error"):
        tail = (result.get("profile") or {}).get("tail") or ""
        if tail:
            lines.append("profile log tail:\n%s" % tail[-1200:])
    return "\n".join(lines)


def format_summary(result: dict) -> str:
    lines = [
        "CannBench result: %s | %s" % (result.get("status"), result.get("error") or "ok"),
        "build: rc=%s %.1fs" % (
            (result.get("build") or {}).get("returncode"),
            (result.get("build") or {}).get("duration_s") or 0.0),
    ]
    test = result.get("test") or {}
    summary = result.get("case_summary") or {}
    if test:
        lines.append("test: rc=%s %.1fs | cases %d/%d pass" % (
            test.get("returncode"), test.get("duration_s") or 0.0,
            summary.get("passed", 0), summary.get("total", 0)))
    cases = result.get("cases") or []
    if cases:
        lines.append("cases:")
        for case in cases:
            lines.append("  %-22s %-9s dim=%-3d %s" % (
                case["shape"], case["dtype"], case["dim"],
                "PASS" if case["passed"] else "FAIL"))
    timing = result.get("timing") or {}
    if timing.get("speedup") is not None:
        lines.append("speedup: %.3fx (%s)" % (
            float(timing["speedup"]), timing.get("metric_key") or "speedup"))
    raw = timing.get("raw") or {}
    for key, note in (("total_speedup", "sum baseline / sum candidate"),
                      ("worst_speedup", "worst single case")):
        if raw.get(key) is not None:
            lines.append("%s: %.3fx (%s)" % (key, float(raw[key]), note))
    bench_cases = raw.get("cases") or []
    if bench_cases:
        lines.append("bench cases (baseline vs candidate):")
        for case in bench_cases:
            shape = tuple(case.get("shape") or ())
            lines.append("  %-22s %-9s dim=%-3d base=%8.4fms cand=%8.4fms speedup=%6.3fx %s" % (
                shape, case.get("dtype"), int(case.get("dim", 0)),
                float(case.get("baseline_ms") or 0.0), float(case.get("candidate_ms") or 0.0),
                float(case.get("speedup") or 0.0),
                "OK" if case.get("correct") else "BAD"))
        worst_case = min(bench_cases, key=lambda c: float(c.get("speedup") or 0.0))
        lines.append("worst bench case: %s %s dim=%s speedup=%.3fx" % (
            tuple(worst_case.get("shape") or ()), worst_case.get("dtype"),
            worst_case.get("dim"), float(worst_case.get("speedup") or 0.0)))
    for name in ("build", "test", "bench"):
        stage = result.get(name)
        if stage and stage.get("returncode") not in (0, None):
            lines.append("%s log tail:\n%s" % (name, (stage.get("tail") or "")[-1500:]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli
def _load_solution_files(solution_dir: str, solution_json: str) -> dict:
    files = {}
    if solution_json:
        with open(solution_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        files.update(payload.get("files") or payload)
    if solution_dir:
        for root, _dirs, names in os.walk(solution_dir):
            for name in names:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, solution_dir).replace("\\", "/")
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    files[rel] = fh.read()
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.cannbench",
                                 description="Build a CannBench package and run its official cases.")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--solution-dir", default="")
    ap.add_argument("--solution-json", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--run-id", default="")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    files = _load_solution_files(args.solution_dir, args.solution_json)
    run_root = args.out or tempfile.mkdtemp(prefix="cannbench-%s-" % spec.id)
    result = run_cannbench(spec, files=files, work_root=run_root, run_id=args.run_id)
    print(format_summary(result))
    if args.out:
        with open(os.path.join(run_root, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print("result: %s" % os.path.join(run_root, "result.json"))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())