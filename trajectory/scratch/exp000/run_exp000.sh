#!/bin/bash
#SBATCH --job-name=avd_exp000
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=trajectory/scratch/exp000/job_%j.log
#SBATCH --mail-user=vishal.kagade@hs-ansbach.de
#SBATCH --mail-type=BEGIN,END,FAIL
#
# exp000 — vanilla baseline calibration + diagnosis (Phase-0). Self-starting,
# idempotent: every step guarded by its state file; if the 24h walltime cuts
# it off, resubmit the SAME script and it resumes where it stopped.
#
# Submit from the repo root:   sbatch trajectory/scratch/exp000/run_exp000.sh
#
# Order puts the S2 working seeds right after the probe so the most valuable
# data (noise floor + per-unit units) exists earliest if walltime hits.
# Estimated total: ~18-22 GPU-h at ~1.1 min/epoch (probe measures the real
# number; the whole chain still fits one 24h allocation if s2 <= 150).
set -e
export CUDA_HOME=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-12.8.1-atsh4meheappo7elpgzr4smo64epwmh6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd "${SLURM_SUBMIT_DIR:-/home/hpc/v134ce/v134ce15/vishal/autovidya/autovidya_visdrone}"

PY=../.venv/bin/python
DRV=trajectory/scratch/exp000/driver.py
ST=trajectory/scratch/exp000/state
mkdir -p "$ST"

echo "[$(date +%H:%M:%S)] EXP000_START"

# 1. Budget probe (200-epoch cap, seed 42) -> pre-registered budget rule.
[ -f $ST/probe.json ] || $PY $DRV probe
echo "[$(date +%H:%M:%S)] PROBE_DONE $($PY -c "import json;b=json.load(open('$ST/probe.json'))['budgets'];print('s1=',b['s1_epochs'],' s2=',b['s2_epochs'])")"

# 2. S2 legs: working seeds first (noise floor), then sealed baseline legs.
for s in 42 123 7 1000 2000; do
  [ -f $ST/S2_seed$s.json ] || $PY $DRV stage --stage S2 --seed $s
done

# 3. Per-unit F1 on the frozen eval for every S2 leg (last.pt — Tier-3 lock).
for s in 42 123 7 1000 2000; do
  [ -f $ST/per_unit_S2_seed$s.json ] && continue
  D=$($PY -c "import json;print(json.load(open('$ST/S2_seed$s.json'))['save_dir'])")
  $PY $DRV units --seed $s --weights "$D/weights/last.pt"
done

# 4. S1 legs (probe-budget noise floor; S1 is probes-only, not a screen).
for s in 42 123 7; do
  [ -f $ST/S1_seed$s.json ] || $PY $DRV stage --stage S1 --seed $s
done

# 5. Diagnostics on the best WORKING seed by S2 primary.
if [ ! -f $ST/diagnostics.json ]; then
  BEST=$($PY -c "
import json
p={s: json.load(open(f'$ST/S2_seed{s}.json'))['primary'] for s in (42,123,7)}
print(max(p, key=p.get))")
  D=$($PY -c "import json;print(json.load(open('$ST/S2_seed$BEST.json'))['save_dir'])")
  $PY $DRV diagnose --seed $BEST --weights "$D/weights/last.pt"
fi

# 6. Assemble: calibration.json (write-once), instrument check, cost log.
[ -f $ST/assembly.json ] || $PY $DRV assemble

echo "[$(date +%H:%M:%S)] EXP000_ALL_DONE"
