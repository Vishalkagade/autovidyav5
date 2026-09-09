#!/bin/bash
# Local-GPU twin of scripts/pipeline.sh (same contract, same state files, same
# gates) — plain bash, no SLURM; control legs run two at a time.
#   nohup bash scripts/pipeline_local.sh exp001 > trajectory/scratch/exp001/local.log 2>&1 &
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "$0")/.." && pwd)"
EXP=${1:?usage: pipeline_local.sh expNNN}
PY=../.venv/bin/python; DRV=trajectory/scratch/$EXP/driver.py; ST=trajectory/scratch/$EXP/state; SEEDS="42 123 7"
mkdir -p "$ST"; [ -f "$DRV" ] || { echo "FATAL: $DRV missing"; exit 1; }
[ -f "trajectory/scratch/$EXP/prereg.json" ] || { echo "FATAL: prereg.json missing"; exit 1; }
run() { echo "[$(date +%H:%M:%S)] START $*"; if $PY $DRV "$@" > "$ST/log_$(echo "$*" | tr -c 'A-Za-z0-9\n' '_').txt" 2>&1; then echo "[$(date +%H:%M:%S)] DONE  $*"; else echo "[$(date +%H:%M:%S)] FAILED $* (see state log)"; exit 1; fi; }
stage() { [ -f $ST/${1}_S2_seed$2.json ] || run stage --variant $1 --seed $2; }
units() { [ -f $ST/per_unit_S2_${1}_seed$2.json ] && return; D=$($PY -c "import json;print(json.load(open('$ST/${1}_S2_seed$2.json'))['save_dir'])"); run units --variant $1 --seed $2 --weights "$D/weights/last.pt"; }
echo "[$(date +%H:%M:%S)] ${EXP^^}_START"
# 1. first-S2-seed screen
stage mech 42
[ -f $ST/screen_seed42.json ] || run screen --seed 42
DEC=$($PY -c "import json;print(json.load(open('$ST/screen_seed42.json'))['decision'])")
echo "[$(date +%H:%M:%S)] SCREEN_DECISION=$DEC"
[ "$DEC" = "KILL" ] && { units mech 42; echo "[$(date +%H:%M:%S)] ${EXP^^}_KILLED_AT_SEED42"; exit 0; }
# 2. second seed + futility stop
stage mech 123
FUT=$($PY - "$EXP" <<'PYEOF'
import json, sys
exp = sys.argv[1]
floor = json.load(open("trajectory/profiles/calibration.json"))["noise_floor_s2_map50"]
bar = json.load(open(f"trajectory/scratch/{exp}/prereg.json")).get("futility_two_seed_mean_delta", -floor)
b = [json.load(open(f"trajectory/scratch/exp000/state/S2_seed{s}.json"))["primary"] for s in (42, 123)]
m = [json.load(open(f"trajectory/scratch/{exp}/state/mech_S2_seed{s}.json"))["primary"] for s in (42, 123)]
d = sum(m) / 2 - sum(b) / 2
print(("STOP" if d <= bar else "GO"), round(d, 5), "bar", round(bar, 5))
PYEOF
)
echo "[$(date +%H:%M:%S)] FUTILITY_CHECK=$FUT"
if [ "${FUT%% *}" = "STOP" ]; then units mech 42 & units mech 123 & wait; echo "[$(date +%H:%M:%S)] ${EXP^^}_FUTILITY_REJECT"; echo "[$(date +%H:%M:%S)] ${EXP^^}_ALL_DONE gate=SHUT(futility)"; exit 0; fi
stage mech 7
# 3. per-unit on all mechanism seeds
units mech 42 & units mech 123 & wait; units mech 7
# 4. P9 control-deferral gate
GATE=$($PY - "$EXP" <<'PYEOF'
import json, sys
exp = sys.argv[1]
floor = json.load(open("trajectory/profiles/calibration.json"))["noise_floor_s2_map50"]
b = [json.load(open(f"trajectory/scratch/exp000/state/S2_seed{s}.json"))["primary"] for s in (42, 123, 7)]
m = [json.load(open(f"trajectory/scratch/{exp}/state/mech_S2_seed{s}.json"))["primary"] for s in (42, 123, 7)]
d = sum(m) / 3 - sum(b) / 3
print(("OPEN" if d >= floor else "SHUT"), round(d, 5), "floor", round(floor, 5))
PYEOF
)
echo "[$(date +%H:%M:%S)] P9_CONTROL_GATE=$GATE"
# 5. matched control (gate open only) — two at a time
if [ "${GATE%% *}" = "OPEN" ]; then
  stage control 42 & stage control 123 & wait; stage control 7
  units control 42 & units control 123 & wait; units control 7
fi
echo "[$(date +%H:%M:%S)] ${EXP^^}_ALL_DONE gate=$GATE"
