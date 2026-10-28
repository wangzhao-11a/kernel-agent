"""Self-test: long-trajectory compression, checkpoint/resume and token accounting."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.context import compress_messages, context_chars  # noqa: E402

TASK = '''
TASK = {
    "schema_version": 1,
    "name": "ctx_probe",
    "level": 1,
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 1,
    "warmup": 2,
    "rep": 5,
    "input_specs": [{"kind": "randn", "shape": [1024], "dtype": "float32"}],
    "baseline": "eager",
}

def reference(x):
    return x * 2

def run(x):
    return x * 2
'''

SOLUTION = "def run(x):\n    return x * 2\n"


def check_compression():
    messages = [{"role": "user", "content": "TASK PROMPT"}]
    for i in range(12):
        messages.append({"role": "assistant", "content": "assistant %d " % i + "x" * 400})
        messages.append({"role": "user", "content": "feedback %d " % i + "y" * 400})
    before = context_chars(messages)
    out = compress_messages(messages, "- evaluations: pass=1 fail=2", keep_recent=4)
    after = context_chars(out)
    assert after < before, (before, after)
    assert out[0]["content"].startswith("TASK PROMPT")
    assert "compressed trajectory" in out[0]["content"]
    assert "evaluations: pass=1 fail=2" in out[0]["content"]
    assert out[-1]["content"] == messages[-1]["content"]
    roles = [m["role"] for m in out]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), roles
    assert compress_messages(messages[:3], "s", keep_recent=4) == messages[:3]


def check_resume(tmp):
    task = os.path.join(tmp, "ctx_task.py")
    replay = os.path.join(tmp, "ctx_replay.json")
    with open(task, "w", encoding="utf-8") as fh:
        fh.write(TASK)
    with open(replay, "w", encoding="utf-8") as fh:
        json.dump({"responses": [
            {"action": "write_solution", "content": SOLUTION, "expected_speedup": 1.0},
            {"action": "finish", "summary": "resumed and finished"},
        ]}, fh)
    out_dir = os.path.join(tmp, "runs")
    base = [sys.executable, "-m", "agent.loop", "--task", task, "--provider", "replay",
            "--replay-file", replay, "--device", "cpu", "--no-rsi",
            "--out-dir", out_dir, "--target-speedup", "99", "--timeout", "120"]
    proc = subprocess.run(base + ["--max-iters", "1"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    metas = glob.glob(os.path.join(out_dir, "*_ctx_probe", "meta.json"))
    assert len(metas) == 1, metas
    run_dir = os.path.dirname(metas[0])
    checkpoint = os.path.join(run_dir, "checkpoint.json")
    assert os.path.isfile(checkpoint), checkpoint
    with open(checkpoint, encoding="utf-8") as fh:
        cp = json.load(fh)
    assert cp["iteration"] == 1 and cp["llm_calls"] == 1, cp
    assert cp["best"]["status"] == "pass", cp["best"]

    proc = subprocess.run(base + ["--max-iters", "2", "--resume", run_dir],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["iterations"] == 2, meta
    assert meta["resumed_from"] == os.path.abspath(run_dir), meta
    assert len(meta["token_usage_by_iteration"]) == 2, meta["token_usage_by_iteration"]
    assert meta["best"]["status"] == "pass", meta["best"]
    with open(os.path.join(run_dir, "transcript.md"), encoding="utf-8") as fh:
        transcript = fh.read()
    assert "resumed from checkpoint" in transcript, transcript[-1500:]
    with open(os.path.join(run_dir, "transcript.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assistants = [r for r in rows if r.get("role") == "assistant"]
    assert assistants and all("usage" in r and "context_chars" in r for r in assistants), assistants


def main():
    check_compression()
    tmp = tempfile.mkdtemp(prefix="ka_ctx_")
    check_resume(tmp)
    print("passed: 2 | failed: 0")


if __name__ == "__main__":
    main()