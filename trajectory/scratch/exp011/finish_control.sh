#!/bin/bash
# Finish exp011: control seed 123 (OOM'd when two P2-topology controls ran in parallel), its units, then classify. Idempotent.
cd "$(cd "$(dirname "$0")/../../.." && pwd)"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python; DRV=trajectory/scratch/exp011/driver.py; ST=trajectory/scratch/exp011/state
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $DRV "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_').txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
[ -f $ST/control_S2_seed123.json ] || run stage --variant control --seed 123
[ -f $ST/per_unit_S2_control_seed123.json ] || run units --variant control --seed 123 --weights "$($PY -c "import json;print(json.load(open('$ST/control_S2_seed123.json'))['save_dir'])")/weights/last.pt"
echo "[$(date +%H:%M:%S)] EXP011_CONTROL123_DONE"
