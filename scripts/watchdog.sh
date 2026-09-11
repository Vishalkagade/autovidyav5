#!/bin/bash
# Deterministic health check for the running experiment chain. Prints ONE line: OK ... or PROBLEM ...
# Checks: (1) a pipeline/chain process is alive, (2) GPU is busy, (3) no FAILED/GATE_FAIL/Traceback in the newest logs,
# (4) the newest per-step log grew in the last 15 min. Exit 1 on PROBLEM.
cd "$(cd "$(dirname "$0")/.." && pwd)"
now=$(date +%s); problems=(); info=()
LOG=$(ls -t trajectory/scratch/*/local.log trajectory/scratch/*/p11.log 2>/dev/null | head -1)
EXP=$(basename "$(dirname "$LOG")")
if grep -qE "ALL_DONE|KILLED|FUTILITY_REJECT" "$LOG" 2>/dev/null; then echo "OK $EXP finished ($(grep -oE 'ALL_DONE.*|KILLED.*|FUTILITY_REJECT.*' "$LOG" | tail -1)) — nothing running, next step is on the agent"; exit 0; fi
if grep -qE "FAILED|GATE_FAIL|FATAL" "$LOG" 2>/dev/null; then problems+=("$EXP log has $(grep -oE 'FAILED.*|GATE_FAIL.*|FATAL.*' "$LOG" | tail -1 | cut -c1-80)"); fi
pgrep -f "pipeline_local.sh|run_exp000_local.sh|run_p11_local.sh|run_voc.sh|launch_after" >/dev/null || problems+=("no chain process alive")
started=$(( now - $(stat -c %Y "$LOG") )); read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ","); [ "$started" -gt 180 ] && [ "${util:-0}" -lt 5 ] && [ "${mem:-0}" -lt 2000 ] && problems+=("GPU idle util ${util}% mem ${mem}MiB")  # util alone is bursty (assigner CPU fallback); an idle GPU also holds no memory; a stage caches the dataset for ~1 min before touching the GPU
NEWEST=$(ls -t trajectory/scratch/$EXP/state/log_*.txt 2>/dev/null | head -1)
if [ -n "$NEWEST" ]; then age=$(( now - $(stat -c %Y "$NEWEST") )); [ $age -gt 900 ] && problems+=("$(basename "$NEWEST" | cut -c1-40) not updated for $((age/60)) min"); tr "\r" "\n" < "$NEWEST" | grep -av "OutOfMemoryError in TaskAlignedAssigner" | grep -aqE "Traceback|Error" && problems+=("Traceback/Error in $(basename "$NEWEST" | cut -c1-40)"); fi
step=$(tail -1 "$LOG" 2>/dev/null | cut -c1-70)
if [ ${#problems[@]} -eq 0 ]; then echo "OK $EXP | gpu ${util}% | $step"; exit 0; else echo "PROBLEM $EXP | ${problems[*]} | last: $step"; exit 1; fi
