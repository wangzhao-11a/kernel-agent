#!/usr/bin/env bash
# Profile one solution on Ascend (910B/910B2) with msprof and print the op summary.
#
#   scripts/profile_msprof.sh tasks/level1/softmax.py my_solution.py [out_dir]
#
# Requires: CANN environment sourced (msprof in PATH), torch_npu installed.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:?usage: profile_msprof.sh <task.py> <solution.py> [out_dir]}"
SOLUTION="${2:?usage: profile_msprof.sh <task.py> <solution.py> [out_dir]}"
OUT="${3:-runs/profiling/msprof}"
mkdir -p "$OUT"

msprof --output="$OUT" --type=text --summary-format=csv --export=on \
    --application="python -m harness.profile_target --task $TASK --solution $SOLUTION --device npu"

python -m harness.profilers.msprof --dir "$OUT"