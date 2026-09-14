#!/bin/bash
# SKU-110K WIN leg for exp011: waits for the VOC leg, then probe -> baseline x3 -> bandgate x3 -> units -> assemble. Sequential, idempotent.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D=trajectory/scratch/confirm_sku/driver.py; ST=trajectory/scratch/confirm_sku/state; mkdir -p $ST
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $D "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
until grep -qE "VOC_EXP011_ALL_DONE|VOC_EXP011_FAILED" trajectory/scratch/confirm_voc/local.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] SKU_START"
[ -f $ST/probe.json ] || run probe
echo "[$(date +%H:%M:%S)] PROBE_DONE $($PY -c "import json;print(json.load(open('$ST/probe.json'))['budgets'])")"
for s in 42 123 7; do [ -f $ST/baseline_S2_seed$s.json ] || run stage --variant baseline --seed $s; done
for s in 42 123 7; do [ -f $ST/bandgate_S2_seed$s.json ] || run stage --variant bandgate --seed $s; done
for v in baseline bandgate; do for s in 42 123 7; do [ -f $ST/per_unit_${v}_seed$s.json ] || run units --variant $v --seed $s --weights "$($PY -c "import json;print(json.load(open('$ST/${v}_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done; done
[ -f $ST/assembly_bandgate.json ] || run assemble --mech-variant bandgate
echo "[$(date +%H:%M:%S)] SKU_ALL_DONE $($PY -c "import json;a=json.load(open('$ST/assembly_bandgate.json'));print('WIN_PASS',a['WIN_PASS'],'delta',round(a['seedavg_delta_map50'],4),'floor',round(a['s2_noise_floor'],4),'ci',[round(x,4) for x in a['per_unit']['ci95']])")"
