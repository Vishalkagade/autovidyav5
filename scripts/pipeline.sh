#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-user=vishal.kagade@hs-ansbach.de
#SBATCH --mail-type=BEGIN,END,FAIL
#
# THE reusable self-starting experiment pipeline (exp005-008 pattern from
# autovidya_v5). One experiment, one sbatch job, zero interaction:
#
#   first-S2-seed screen -> remaining seeds -> per-unit -> P9 control-deferral
#   gate -> control (only if the gate opens)
#
# Idempotent: every step is guarded by its state file, so a requeued or
# resubmitted job resumes where it left off. Headless only — never pty-attach.
#
# Submit (from the repo root; job name and log follow the experiment):
#   sbatch --job-name=avd_exp001 \
#          --output=trajectory/scratch/exp001/job_%j.log \
#          --export=ALL,EXP=exp001 scripts/pipeline.sh
#
# Chain onto a running job (start when its log prints ALL_DONE/KILLED):
#   sbatch ... --export=ALL,EXP=exp002,WAIT_LOG=trajectory/scratch/exp001/job_NNN.log \
#          scripts/pipeline.sh
#
# Per-experiment contract — trajectory/scratch/$EXP/ must contain:
#   prereg.json   pre-registered discriminator, kill bar, P9 gate bar (before
#                 any Stage-2 number exists)
#   driver.py     CLI:  stage --variant mech|control --seed S
#                       units --variant mech|control --seed S --weights W
#                       screen --seed 42   (writes state/screen_seed42.json
#                                           with {"decision": "KILL"|"PROMOTE"})
#                 Site_wrap mechanisms MUST use the SurgeryTrainer pattern
#                 inside driver.py (Model.train drops in-place surgery).
#   state/        stage/units JSONs land here; stage JSONs carry "primary"
#                 and "save_dir".
#
# Baseline reference: trajectory/scratch/exp000/state/S2_seed{42,123,7}.json.
# P9 gate bar: calibration.noise_floor_s2_map50 (trajectory/profiles/
# calibration.json) — the control run is spent only when seed-averaged
# mech-baseline >= 1x that floor (program.md, control-deferral gate).
set -e
export CUDA_HOME=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-12.8.1-atsh4meheappo7elpgzr4smo64epwmh6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${EXP:?set EXP=expNNN via --export}"
PY=../.venv/bin/python
DRV=trajectory/scratch/$EXP/driver.py
ST=trajectory/scratch/$EXP/state
SEEDS="42 123 7"
mkdir -p "$ST"
[ -f "$DRV" ] || { echo "FATAL: $DRV missing"; exit 1; }
[ -f "trajectory/scratch/$EXP/prereg.json" ] || { echo "FATAL: prereg.json missing — pre-register before spending GPU"; exit 1; }

# Optional chain: wait for a predecessor's log to declare it finished.
if [ -n "$WAIT_LOG" ]; then
  echo "[$(date +%H:%M:%S)] CHAIN: waiting on $WAIT_LOG"
  until grep -qE "ALL_DONE|KILLED" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M:%S)] CHAIN: predecessor finished"
fi

echo "[$(date +%H:%M:%S)] ${EXP^^}_START"

# 1. First-S2-seed screen (seed 42). S1-as-a-stage is retired:
#    ins_s1_screen_has_no_predictive_value (inherited from v5).
[ -f $ST/mech_S2_seed42.json ] || $PY $DRV stage --variant mech --seed 42
[ -f $ST/screen_seed42.json ]  || $PY $DRV screen --seed 42
DEC=$($PY -c "import json;print(json.load(open('$ST/screen_seed42.json'))['decision'])")
echo "[$(date +%H:%M:%S)] SCREEN_DECISION=$DEC"
[ "$DEC" = "KILL" ] && { echo "[$(date +%H:%M:%S)] ${EXP^^}_KILLED_AT_SEED42"; exit 0; }

# 2. Second seed, then the pre-registered futility stop: if the two-seed
#    mean delta sits at/below the bar (prereg futility_two_seed_mean_delta,
#    default -1x S2 noise floor), seed 7 cannot rescue a classification —
#    stop as an early Reject and save the run.
[ -f $ST/mech_S2_seed123.json ] || $PY $DRV stage --variant mech --seed 123
FUT=$($PY - "$EXP" <<'PYEOF'
import json, sys
exp = sys.argv[1]
floor = json.load(open("trajectory/profiles/calibration.json"))["noise_floor_s2_map50"]
bar = json.load(open(f"trajectory/scratch/{exp}/prereg.json")).get(
    "futility_two_seed_mean_delta", -floor)
b = [json.load(open(f"trajectory/scratch/exp000/state/S2_seed{s}.json"))["primary"] for s in (42, 123)]
m = [json.load(open(f"trajectory/scratch/{exp}/state/mech_S2_seed{s}.json"))["primary"] for s in (42, 123)]
d = sum(m) / 2 - sum(b) / 2
print(("STOP" if d <= bar else "GO"), round(d, 5), "bar", round(bar, 5))
PYEOF
)
echo "[$(date +%H:%M:%S)] FUTILITY_CHECK=$FUT"
if [ "${FUT%% *}" = "STOP" ]; then
  for s in 42 123; do
    [ -f $ST/per_unit_S2_mech_seed$s.json ] && continue
    D=$($PY -c "import json;print(json.load(open('$ST/mech_S2_seed$s.json'))['save_dir'])")
    $PY $DRV units --variant mech --seed $s --weights "$D/weights/last.pt"
  done
  echo "[$(date +%H:%M:%S)] ${EXP^^}_FUTILITY_REJECT"
  echo "[$(date +%H:%M:%S)] ${EXP^^}_ALL_DONE gate=SHUT(futility)"
  exit 0
fi
[ -f $ST/mech_S2_seed7.json ] || $PY $DRV stage --variant mech --seed 7

# 3. Per-unit metric on every mechanism seed (frozen eval, 2,158 units).
#    ALWAYS last.pt — the Tier-3 eval-checkpoint lock (best.pt selection
#    sees part of the frozen eval; see adapter program.md).
for s in $SEEDS; do
  [ -f $ST/per_unit_S2_mech_seed$s.json ] && continue
  D=$($PY -c "import json;print(json.load(open('$ST/mech_S2_seed$s.json'))['save_dir'])")
  $PY $DRV units --variant mech --seed $s --weights "$D/weights/last.pt"
done

# 4. P9 control-deferral gate: spend the control only if the mechanism could
#    still rise above Hold (seed-avg delta >= 1x S2 noise floor).
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

# 5. Matched control, same stages and seeds (gate open only).
if [ "${GATE%% *}" = "OPEN" ]; then
  for s in $SEEDS; do
    [ -f $ST/control_S2_seed$s.json ] || $PY $DRV stage --variant control --seed $s
  done
  for s in $SEEDS; do
    [ -f $ST/per_unit_S2_control_seed$s.json ] && continue
    D=$($PY -c "import json;print(json.load(open('$ST/control_S2_seed$s.json'))['save_dir'])")
    $PY $DRV units --variant control --seed $s --weights "$D/weights/last.pt"
  done
fi

echo "[$(date +%H:%M:%S)] ${EXP^^}_ALL_DONE gate=$GATE"
