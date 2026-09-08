#!/bin/bash
# Waits for exp001's P11 chain, reruns the tiny-overfit gate on a free GPU, then runs exp002.
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
until grep -q "EXP001_P11_ALL_DONE" trajectory/scratch/exp001/p11.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] P11 finished — running exp002 tinyprobe gate"
if ../.venv/bin/python trajectory/scratch/exp002/driver.py tinyprobe > trajectory/scratch/exp002/state/log_tinyprobe.txt 2>&1; then
  echo "[$(date +%H:%M:%S)] TINYPROBE PASS — launching exp002 pipeline"
  bash scripts/pipeline_local.sh exp002
else
  echo "[$(date +%H:%M:%S)] EXP002_GATE_FAIL tinyprobe — NOT launching"
fi
