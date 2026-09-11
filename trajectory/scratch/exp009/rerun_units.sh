#!/bin/bash
# Re-run the exp009 per-unit step (failed on a module attribute; fixed). Idempotent, sequential, runs beside exp010 training.
cd "$(cd "$(dirname "$0")/../../.." && pwd)"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 42 123 7; do
  [ -f trajectory/scratch/exp009/state/per_unit_S2_mech_seed$s.json ] && continue
  echo "[$(date +%H:%M:%S)] START units seed $s"
  ../.venv/bin/python trajectory/scratch/exp009/driver.py units --variant mech --seed $s --weights runs_visdrone/exp009_mech_S2_seed$s/weights/last.pt > trajectory/scratch/exp009/state/log_units_rerun_seed$s.txt 2>&1 && echo "[$(date +%H:%M:%S)] DONE units seed $s" || echo "[$(date +%H:%M:%S)] FAILED units seed $s"
done
echo "[$(date +%H:%M:%S)] UNITS_RERUN_DONE"
