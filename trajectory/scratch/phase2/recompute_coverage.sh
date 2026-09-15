#!/bin/bash
# Recompute band coverage (square-grid fix) for the exp011 legs on VOC and SKU-110K, then re-assemble. Inference only.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$(cd "$(dirname "$0")/../../.." && pwd)"; PY=../.venv/bin/python
for ds in voc sku; do
  for s in 42 123 7; do $PY trajectory/scratch/confirm_$ds/driver.py coverage --variant bandgate --seed $s 2>&1 | grep "\[coverage\]"; done
  rm -f trajectory/scratch/confirm_$ds/state/assembly_bandgate.json; $PY trajectory/scratch/confirm_$ds/driver.py assemble --mech-variant bandgate 2>&1 | grep -E "^\[(voc|sku)\]"
done
echo COVERAGE_DONE
