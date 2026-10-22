#!/usr/bin/env bash
# Profile one solution with Nsight Compute (NVIDIA) and print the parsed summary.
#
#   scripts/profile_ncu.sh tasks/level1/softmax.py solutions/examples/softmax_manual.py [out_dir]
#
# Requires: ncu in PATH, NVIDIA GPU, torch CUDA build.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:?usage: profile_ncu.sh <task.py> <solution.py> [out_dir]}"
SOLUTION="${2:?usage: profile_ncu.sh <task.py> <solution.py> [out_dir]}"
OUT="${3:-runs/profiling/ncu}"
mkdir -p "$OUT"

ncu --csv --target-processes all --launch-count 3 -f -o "$OUT/profile" \
    python -m harness.profile_target --task "$TASK" --solution "$SOLUTION" --device cuda

ncu --import "$OUT/profile.ncu-rep" --csv > "$OUT/profile.csv"
python -m harness.profilers.ncu --file "$OUT/profile.csv"