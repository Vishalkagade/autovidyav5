#!/bin/bash
# exp010: wait for the exp009 chain, run the tiny-overfit gate, then the pipeline. Idempotent (state files).
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -qE "EXP009_ALL_DONE|EXP009_KILLED" trajectory/scratch/exp009/local.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] exp009 finished — running exp010 tinyprobe gate"
if [ ! -f trajectory/scratch/exp010/state/tinyprobe.json ]; then
  ../.venv/bin/python trajectory/scratch/exp010/driver.py tinyprobe > trajectory/scratch/exp010/state/log_tinyprobe.txt 2>&1 || { echo "[$(date +%H:%M:%S)] EXP010_GATE_FAIL tinyprobe — NOT launching"; exit 1; }
fi
../.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('trajectory/scratch/exp010/state/tinyprobe.json'))['pass'] else 1)" || { echo "[$(date +%H:%M:%S)] EXP010_GATE_FAIL tinyprobe"; exit 1; }
echo "[$(date +%H:%M:%S)] TINYPROBE PASS — launching exp010 pipeline"
bash scripts/pipeline_local.sh exp010 > trajectory/scratch/exp010/local.log 2>&1
