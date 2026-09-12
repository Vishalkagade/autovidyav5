#!/bin/bash
# exp011: wait for the exp010 chain (including its controls if the gate opened), tiny-overfit gate, then the pipeline. Idempotent.
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -qE "EXP010_ALL_DONE|EXP010_KILLED" trajectory/scratch/exp010/local.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] exp010 finished — running exp011 tinyprobe gate"
if [ ! -f trajectory/scratch/exp011/state/tinyprobe.json ]; then
  ../.venv/bin/python trajectory/scratch/exp011/driver.py tinyprobe > trajectory/scratch/exp011/state/log_tinyprobe.txt 2>&1 || { echo "[$(date +%H:%M:%S)] EXP011_GATE_FAIL tinyprobe — NOT launching"; exit 1; }
fi
../.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('trajectory/scratch/exp011/state/tinyprobe.json'))['pass'] else 1)" || { echo "[$(date +%H:%M:%S)] EXP011_GATE_FAIL tinyprobe"; exit 1; }
echo "[$(date +%H:%M:%S)] TINYPROBE PASS — launching exp011 pipeline"
bash scripts/pipeline_local.sh exp011 > trajectory/scratch/exp011/local.log 2>&1
