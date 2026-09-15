#!/bin/bash
# SKU-110K diagnostic: exp001's ungated P2 level (variant mech) x3 seeds, units, assemble vs the same baselines. Sequential, idempotent.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D=trajectory/scratch/confirm_sku/driver.py; ST=trajectory/scratch/confirm_sku/state
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $D "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
echo "[$(date +%H:%M:%S)] SKU_DIAG_START"
for s in 42 123 7; do [ -f $ST/mech_S2_seed$s.json ] || run stage --variant mech --seed $s; done
for s in 42 123 7; do [ -f $ST/per_unit_mech_seed$s.json ] || run units --variant mech --seed $s --weights "$($PY -c "import json;print(json.load(open('$ST/mech_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
[ -f $ST/assembly.json ] || run assemble --mech-variant mech
echo "[$(date +%H:%M:%S)] SKU_DIAG_ALL_DONE $($PY -c "import json;a=json.load(open('$ST/assembly.json'));print('WIN_PASS',a['WIN_PASS'],'NO_REGRESSION',a['NO_REGRESSION_PASS'],'delta',round(a['seedavg_delta_map50'],4),'ci',[round(x,4) for x in a['per_unit']['ci95']])")"
