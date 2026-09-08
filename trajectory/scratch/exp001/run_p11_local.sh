#!/bin/bash
# exp001 P11 sealed-seed replication (seeds 1000, 2000). Baseline legs were
# deferred at exp000 -> run them here via the exp000 driver (writes exp000
# state files, as the sbatch chain would have). Mechanism legs alone (20 GB).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D0=trajectory/scratch/exp000/driver.py; D1=trajectory/scratch/exp001/driver.py
S0=trajectory/scratch/exp000/state; S1=trajectory/scratch/exp001/state
run() { echo "[$(date +%H:%M:%S)] START $*"; "$@" > "$S1/log_p11_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; echo "[$(date +%H:%M:%S)] DONE  $*"; }
echo "[$(date +%H:%M:%S)] EXP001_P11_START"
( [ -f $S0/S2_seed1000.json ] || run $PY $D0 stage --stage S2 --seed 1000 ) & ( [ -f $S0/S2_seed2000.json ] || run $PY $D0 stage --stage S2 --seed 2000 ) & wait
for s in 1000 2000; do [ -f $S1/mech_S2_seed$s.json ] || run $PY $D1 stage --variant mech --seed $s; done
for s in 1000 2000; do
  [ -f $S0/per_unit_S2_seed$s.json ] || { D=$($PY -c "import json;print(json.load(open('$S0/S2_seed$s.json'))['save_dir'])"); run $PY $D0 units --seed $s --weights "$D/weights/last.pt"; }
  [ -f $S1/per_unit_S2_mech_seed$s.json ] || { D=$($PY -c "import json;print(json.load(open('$S1/mech_S2_seed$s.json'))['save_dir'])"); run $PY $D1 units --variant mech --seed $s --weights "$D/weights/last.pt"; }
done
echo "[$(date +%H:%M:%S)] EXP001_P11_ALL_DONE"
