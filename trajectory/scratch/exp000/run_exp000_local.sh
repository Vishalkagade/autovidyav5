#!/bin/bash
# exp000 on the local RTX 4090 (user decision 2026-09-07). Same steps and
# state files as run_exp000.sh, so either script resumes the other. Differences:
#   - plain bash, no SLURM; run detached:  nohup bash trajectory/scratch/exp000/run_exp000_local.sh > trajectory/scratch/exp000/local.log 2>&1 &
#   - at most TWO trainings at once (program.md, 2026-09-07 amendment)
#   - sealed seeds 1000/2000 deferred until a Provisional Winner needs P11
set -e
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
PY=../.venv/bin/python
DRV=trajectory/scratch/exp000/driver.py
ST=trajectory/scratch/exp000/state
mkdir -p "$ST"
run() { echo "[$(date +%H:%M:%S)] START $*"; $PY $DRV "$@" > "$ST/log_$(echo "$*" | tr ' -' '__').txt" 2>&1; echo "[$(date +%H:%M:%S)] DONE  $*"; }
stage() { [ -f $ST/${1}_seed$2.json ] || run stage --stage $1 --seed $2; }
units() { [ -f $ST/per_unit_S2_seed$1.json ] && return; D=$($PY -c "import json;print(json.load(open('$ST/S2_seed$1.json'))['save_dir'])"); run units --seed $1 --weights "$D/weights/last.pt"; }

echo "[$(date +%H:%M:%S)] EXP000_START"
# 1. probe alone (budgets gate everything after it)
[ -f $ST/probe.json ] || run probe
echo "[$(date +%H:%M:%S)] PROBE_DONE $($PY -c "import json;b=json.load(open('$ST/probe.json'))['budgets'];print('s1=',b['s1_epochs'],' s2=',b['s2_epochs'])")"
# 2. S2 42 || S2 123
stage S2 42 & stage S2 123 & wait
# 3. S2 7 || (S1 42 -> S1 123 -> S1 7)
stage S2 7 & ( stage S1 42; stage S1 123; stage S1 7 ) & wait
# 4. per-unit F1 (predict is light; two at once)
units 42 & units 123 & wait
units 7
# 5. diagnostics on best working seed
if [ ! -f $ST/diagnostics.json ]; then
  BEST=$($PY -c "
import json
p={s: json.load(open(f'$ST/S2_seed{s}.json'))['primary'] for s in (42,123,7)}
print(max(p, key=p.get))")
  D=$($PY -c "import json;print(json.load(open('$ST/S2_seed$BEST.json'))['save_dir'])")
  run diagnose --seed $BEST --weights "$D/weights/last.pt"
fi
# 6. assemble (sealed seeds absent -> tolerated)
[ -f $ST/assembly.json ] || run assemble
echo "[$(date +%H:%M:%S)] EXP000_ALL_DONE"
