"""Self-test: roofline estimate math + worker integration (CPU mechanism only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TASK = '''
TASK = {
    "schema_version": 1,
    "name": "roof_probe",
    "level": 1,
    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
    "num_trials": 1,
    "warmup": 2,
    "rep": 5,
    "input_specs": [{"kind": "randn", "shape": [4096], "dtype": "float32"}],
    "workload": {"bytes": 32768, "flops": 0},
    "peak_bw_gbps": 100.0,
}

def reference(x):
    return x * 2

def run(x):
    return x * 2
'''

SOLUTION = '''
def run(x):
    return x * 2
'''


def check_estimate():
    from harness.roofline import estimate

    roof = estimate({"bytes": 1e9, "flops": 0}, 1.0,
                    {"peak_bw_gbps": 1000, "peak_tflops": 100})
    assert roof["achieved_bw_gbps"] == 1000.0, roof
    assert roof["bw_utilization"] == 1.0 and roof["bottleneck"] == "memory", roof
    roof = estimate({"bytes": 0, "flops": 1e12}, 2000.0,
                    {"peak_bw_gbps": 1000, "peak_tflops": 500})
    assert abs(roof["achieved_tflops"] - 0.5) < 1e-9, roof
    assert roof["bottleneck"] == "compute", roof
    assert estimate({"bytes": 0, "flops": 0}, 1.0) == {}
    assert estimate({"bytes": 1}, 0.0) == {}


def check_worker(tmp):
    task = os.path.join(tmp, "roof_task.py")
    sol = os.path.join(tmp, "roof_sol.py")
    with open(task, "w", encoding="utf-8") as fh:
        fh.write(TASK)
    with open(sol, "w", encoding="utf-8") as fh:
        fh.write(SOLUTION)
    out_dir = os.path.join(tmp, "out")
    cmd = [sys.executable, "-m", "harness.runner", "--task", task, "--solution", sol,
           "--device", "cpu", "--out-dir", out_dir, "--trials", "1", "--warmup", "2",
           "--rep", "5", "--timeout", "120"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    import glob
    paths = glob.glob(os.path.join(out_dir, "result_*.json"))
    assert len(paths) == 1, paths
    with open(paths[0], encoding="utf-8") as fh:
        result = json.load(fh)
    roof = (result.get("timing") or {}).get("roofline") or {}
    assert roof.get("bottleneck") == "memory", result.get("timing")
    assert roof.get("achieved_bw_gbps", 0) > 0, roof
    assert roof.get("bw_utilization") is not None, roof


def check_memory_search():
    from rsi.memory import MemoryStore

    store = MemoryStore(os.path.join(ROOT, "rsi", "knowledge"))
    hits = store.search_text("softmax online", top_k=5)
    assert hits, "free-text memory search returned no hits"
    assert any(h["type"] == "strategy" for h in hits), hits
    assert store.search_text("qqzzxx", top_k=5) == []


def main():
    check_estimate()
    tmp = tempfile.mkdtemp(prefix="ka_roof_")
    check_worker(tmp)
    check_memory_search()
    print("passed: 3 | failed: 0")


if __name__ == "__main__":
    main()