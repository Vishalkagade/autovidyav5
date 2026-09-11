#!/bin/bash
# exp009: tiny-overfit gate, then the pipeline. Idempotent (state files).
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -f trajectory/scratch/exp009/state/tinyprobe.json ]; then
  echo "[$(date +%H:%M:%S)] running exp009 tinyprobe gate"
  ../.venv/bin/python trajectory/scratch/exp009/driver.py tinyprobe > trajectory/scratch/exp009/state/log_tinyprobe.txt 2>&1 || { echo "[$(date +%H:%M:%S)] EXP009_GATE_FAIL tinyprobe — NOT launching"; exit 1; }
fi
../.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('trajectory/scratch/exp009/state/tinyprobe.json'))['pass'] else 1)" || { echo "[$(date +%H:%M:%S)] EXP009_GATE_FAIL tinyprobe"; exit 1; }
echo "[$(date +%H:%M:%S)] TINYPROBE PASS — launching exp009 pipeline"
bash scripts/pipeline_local.sh exp009
