#!/bin/bash
# exp012 on SKU-110K: seed-42 screen (vs baseline, prereg bar) -> seeds 123/7 -> units -> assemble vs baseline and vs the anchor (bandgate). Sequential, idempotent.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D=trajectory/scratch/confirm_sku/driver.py; ST=trajectory/scratch/confirm_sku/state; E=trajectory/scratch/exp012/state
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $D "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
echo "[$(date +%H:%M:%S)] EXP012_SKU_START"
[ -f $ST/sizeband_S2_seed42.json ] || run stage --variant sizeband --seed 42
DEC=$($PY - <<'PYEOF'
import json; L=lambda v: json.load(open(f"trajectory/scratch/confirm_sku/state/{v}_S2_seed42.json"))["primary"]
b=[json.load(open(f"trajectory/scratch/confirm_sku/state/baseline_S2_seed{s}.json"))["primary"] for s in (42,123,7)]; floor=max(b)-min(b); d=L("sizeband")-L("baseline")
dec="PROMOTE" if d >= -2*floor else "KILL"; json.dump({"seed":42,"sizeband":L("sizeband"),"baseline":L("baseline"),"anchor":L("bandgate"),"delta_vs_baseline":d,"bar":-2*floor,"decision":dec}, open("trajectory/scratch/exp012/state/screen_sku_seed42.json","w"), indent=2); print(dec, round(d,4), "bar", round(-2*floor,4))
PYEOF
)
echo "[$(date +%H:%M:%S)] SCREEN_DECISION=$DEC"
[ "${DEC%% *}" = "KILL" ] && { run units --variant sizeband --seed 42 --weights "$($PY -c "import json;print(json.load(open('$ST/sizeband_S2_seed42.json'))['save_dir'])")/weights/last.pt"; echo "[$(date +%H:%M:%S)] EXP012_KILLED_AT_SEED42"; exit 0; }
for s in 123 7; do [ -f $ST/sizeband_S2_seed$s.json ] || run stage --variant sizeband --seed $s; done
for s in 42 123 7; do [ -f $ST/per_unit_sizeband_seed$s.json ] || run units --variant sizeband --seed $s --weights "$($PY -c "import json;print(json.load(open('$ST/sizeband_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
[ -f $ST/assembly_sizeband.json ] || run assemble --mech-variant sizeband
[ -f $ST/assembly_sizeband_vs_bandgate.json ] || run assemble --mech-variant sizeband --ref bandgate
echo "[$(date +%H:%M:%S)] EXP012_SKU_ALL_DONE $($PY -c "import json;a=json.load(open('$ST/assembly_sizeband.json'));c=json.load(open('$ST/assembly_sizeband_vs_bandgate.json'));print('vs baseline: NO_REGRESSION',a['NO_REGRESSION_PASS'],'WIN',a['WIN_PASS'],'delta',round(a['seedavg_delta_map50'],4),'ci',[round(x,4) for x in a['per_unit']['ci95']],'| vs anchor: delta',round(c['seedavg_delta_map50'],4),'ci',[round(x,4) for x in c['per_unit']['ci95']],'per_seed',{k:round(v,4) for k,v in c['per_unit']['per_seed_means'].items()})")"
