#!/bin/bash
# Waits for the VOC confirmation chain, runs the exp008 tiny-overfit gate on a free GPU, then the exp008 pipeline.
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
until grep -q "VOC_ALL_DONE" trajectory/scratch/confirm_voc/local.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] VOC finished — running exp008 tinyprobe gate"
if ../.venv/bin/python trajectory/scratch/exp008/driver.py tinyprobe > trajectory/scratch/exp008/state/log_tinyprobe.txt 2>&1; then
  echo "[$(date +%H:%M:%S)] TINYPROBE PASS — launching exp008 pipeline"
  bash scripts/pipeline_local.sh exp008
else
  echo "[$(date +%H:%M:%S)] EXP008_GATE_FAIL tinyprobe — NOT launching"
fi
