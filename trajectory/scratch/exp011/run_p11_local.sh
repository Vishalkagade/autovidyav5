#!/bin/bash
# exp011: after the control seed-123 leg, classify; if Provisional Winner run P11 (sealed seeds 1000/2000; baseline legs exist
# in exp000 state since exp001's P11). Control legs on the sealed seeds only if classify set control_must_replicate_at_p11.
# Sequential (one P2-topology run at a time). Idempotent.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python; D=trajectory/scratch/exp011/driver.py; ST=trajectory/scratch/exp011/state; LOG=trajectory/scratch/exp011/local.log
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $D "$@" > "$ST/log_p11_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-80).txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
until grep -q "EXP011_CONTROL123_DONE" $LOG 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] EXP011_P11_START"
[ -f trajectory/experiments/exp011.json ] || $PY trajectory/scratch/exp011/classify.py --write --notes "$(cat trajectory/scratch/exp011/classify_notes.txt 2>/dev/null)" > $ST/log_classify.txt 2>&1 || { echo "[$(date +%H:%M:%S)] FAILED classify"; exit 1; }
CLS=$($PY -c "import json;print(json.load(open('trajectory/experiments/exp011.json'))['classification'])"); echo "[$(date +%H:%M:%S)] CLASSIFICATION=$CLS"
[ "$CLS" = "Provisional Winner" ] || { echo "[$(date +%H:%M:%S)] EXP011_P11_ALL_DONE (not run: $CLS)"; exit 0; }
CTL=$($PY -c "import json;print(json.load(open('trajectory/experiments/exp011.json'))['p9']['control_must_replicate_at_p11'])"); echo "[$(date +%H:%M:%S)] CONTROL_MUST_REPLICATE=$CTL"
for s in 1000 2000; do [ -f $ST/mech_S2_seed$s.json ] || run stage --variant mech --seed $s; done
for s in 1000 2000; do [ -f $ST/per_unit_S2_mech_seed$s.json ] || run units --variant mech --seed $s --weights "$($PY -c "import json;print(json.load(open('$ST/mech_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
if [ "$CTL" = "True" ]; then
  for s in 1000 2000; do [ -f $ST/control_S2_seed$s.json ] || run stage --variant control --seed $s; done
  for s in 1000 2000; do [ -f $ST/per_unit_S2_control_seed$s.json ] || run units --variant control --seed $s --weights "$($PY -c "import json;print(json.load(open('$ST/control_S2_seed$s.json'))['save_dir'])")/weights/last.pt"; done
fi
$PY trajectory/scratch/exp011/p11_classify.py --write > $ST/log_p11_classify.txt 2>&1 && echo "[$(date +%H:%M:%S)] P11_VERDICT=$($PY -c "import json;print(json.load(open('trajectory/experiments/exp011.json'))['classification'])")" || echo "[$(date +%H:%M:%S)] FAILED p11_classify"
echo "[$(date +%H:%M:%S)] EXP011_P11_ALL_DONE"
