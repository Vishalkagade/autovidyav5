#!/bin/bash
# exp015 follow-on legs after the SKU-110K chain: VisDrone (vs anchor exp011; 3 seeds, units) then VOC (negband variant, 3 seeds, units, assemble). Sequential, idempotent.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; DV=trajectory/scratch/exp015/driver_visdrone.py; SV=trajectory/scratch/exp015/state; DO=trajectory/scratch/confirm_voc/driver.py; SO=trajectory/scratch/confirm_voc/state
run() { d=$1; st=$2; shift 2; echo "[$(date +%H:%M:%S)] START $*"; if $PY $d "$@" > "$st/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
until grep -qE "EXP015_SKU_ALL_DONE|EXP015_KILLED" trajectory/scratch/exp015/local.log 2>/dev/null; do sleep 60; done
grep -q "EXP015_KILLED" trajectory/scratch/exp015/local.log && { echo "[$(date +%H:%M:%S)] SKU leg killed — follow-on legs not run"; exit 0; }
echo "[$(date +%H:%M:%S)] EXP015_VISDRONE_START"
for s in 42 123 7; do [ -f $SV/mech_S2_seed$s.json ] || run $DV $SV stage --variant mech --seed $s; done
for s in 42 123 7; do [ -f $SV/per_unit_S2_mech_seed$s.json ] || run $DV $SV units --variant mech --seed $s --weights "$($PY -c "import json;print(json.load(open('$SV/mech_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
echo "[$(date +%H:%M:%S)] EXP015_VISDRONE_ALL_DONE"
echo "[$(date +%H:%M:%S)] EXP015_VOC_START"
for s in 42 123 7; do [ -f $SO/negband_S2_seed$s.json ] || run $DO $SO stage --variant negband --seed $s; done
for s in 42 123 7; do [ -f $SO/per_unit_negband_seed$s.json ] || run $DO $SO units --variant negband --seed $s --weights "$($PY -c "import json;print(json.load(open('$SO/negband_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
[ -f $SO/assembly_negband.json ] || run $DO $SO assemble --mech-variant negband
echo "[$(date +%H:%M:%S)] EXP015_VOC_ALL_DONE $($PY -c "import json;a=json.load(open('$SO/assembly_negband.json'));print('NO_REGRESSION_PASS',a['NO_REGRESSION_PASS'],'delta',round(a['seedavg_delta_map50'],4),'ci',[round(x,4) for x in a['per_unit']['ci95']])")"
echo "[$(date +%H:%M:%S)] EXP015_FOLLOWON_ALL_DONE"
