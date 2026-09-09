#!/bin/bash
# VOC no-regression leg: probe -> baseline seeds (two at a time) -> P2 seeds (alone) -> units -> assemble. Idempotent.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D=trajectory/scratch/confirm_voc/driver.py; ST=trajectory/scratch/confirm_voc/state; mkdir -p $ST
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $D "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
stage() { [ -f $ST/${1}_S2_seed$2.json ] || run stage --variant $1 --seed $2; }
units() { [ -f $ST/per_unit_${1}_seed$2.json ] && return; W=$($PY -c "import json;print(json.load(open('$ST/${1}_S2_seed$2.json'))['save_dir'])")/weights/last.pt; run units --variant $1 --seed $2 --weights "$W"; }
echo "[$(date +%H:%M:%S)] VOC_START"
[ -f $ST/probe.json ] || run probe
echo "[$(date +%H:%M:%S)] PROBE_DONE $($PY -c "import json;print(json.load(open('$ST/probe.json'))['budgets'])")"
stage baseline 42 & stage baseline 123 & wait; stage baseline 7
for s in 42 123 7; do stage mech $s; done
units baseline 42 & units baseline 123 & wait; units baseline 7 & units mech 42 & wait; units mech 123 & units mech 7 & wait
[ -f $ST/assembly.json ] || run assemble
echo "[$(date +%H:%M:%S)] VOC_ALL_DONE"
