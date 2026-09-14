#!/bin/bash
# VOC no-regression leg for exp011 (bandgate variant); baseline legs reused. Sequential, idempotent.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D=trajectory/scratch/confirm_voc/driver.py; ST=trajectory/scratch/confirm_voc/state
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $D "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
echo "[$(date +%H:%M:%S)] VOC_EXP011_START"
for s in 42 123 7; do [ -f $ST/bandgate_S2_seed$s.json ] || run stage --variant bandgate --seed $s; done
for s in 42 123 7; do [ -f $ST/per_unit_bandgate_seed$s.json ] || run units --variant bandgate --seed $s --weights "$($PY -c "import json;print(json.load(open('$ST/bandgate_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
[ -f $ST/assembly_bandgate.json ] || run assemble --mech-variant bandgate
echo "[$(date +%H:%M:%S)] VOC_EXP011_ALL_DONE $($PY -c "import json;a=json.load(open('$ST/assembly_bandgate.json'));print('NO_REGRESSION_PASS',a['NO_REGRESSION_PASS'],'delta',round(a['seedavg_delta_map50'],4),'ci',[round(x,4) for x in a['per_unit']['ci95']])")"
