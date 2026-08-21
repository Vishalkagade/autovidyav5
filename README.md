# autovidya_visdrone

**AutoVidya research harness, instantiated for YOLO26n + VisDrone-DET
from-scratch (bold-regime cross-domain architectural synthesis).**

Successor to [`../autovidya_v5`](../autovidya_v5) (COCO-2k, closed as a
structured null). Trajectory hypothesis: YOLO26+COCO is a co-evolved local
optimum — mechanisms that were nulls there may work on tiny-object aerial
imagery, where that equilibrium does not hold. A VisDrone Winner must then
replicate on 2 more datasets (candidates: SKU-110K, TT100K) before being
claimed as a model improvement.

This is a standalone repo. It contains the harness (`core/`, model-agnostic,
copied frozen from v5), a YOLO26n VisDrone adapter
(`adapters/yolo26n_visdrone_scratch/`), and protocol docs (`program.md`)
that drive an LLM agent through calibrate/diagnose → explore phases under a
strict discipline layer.

## What is different from v5

1. **Dataset**: full VisDrone-DET train split (6,471 images, 10 classes,
   YOLO labels) from scratch; **frozen eval set = val + test-dev (2,158
   images)**, locked by `visdrone_manifest.json`.
2. **Screen = first S2 seed.** v5's P12 audit showed the short Stage-1
   screen had no predictive value; this trajectory screens at the full
   Stage-2 budget on seed 42 and only then spends seeds 123/7.
3. **One reusable pipeline runner** (`scripts/pipeline.sh`): self-starting
   sbatch, idempotent, chainable — first-S2-seed screen → remaining seeds →
   per-unit → P9 control-deferral gate → control. Headless only.
4. **v5's paid-for lessons are baked in**: SurgeryTrainer mandatory for
   site_wrap, aux_objective is driver-side only, per-image predict for the
   unit metric, noise-floor caveats. See the inherited entries in
   `trajectory/insights.md`.

## Quick start

```bash
# 1. Render the dataset locks (lists, yamls, nc=10 model yaml, manifest):
python setup/create_visdrone_lists.py

# 2. Smoke-test the harness wiring (no GPU needed):
../.venv/bin/python -m scripts.smoke_test

# 3. Submit exp000 (calibration + diagnosis) as a self-starting job:
sbatch --job-name=avd_exp000 \
       --output=trajectory/scratch/exp000/job_%j.log \
       --export=ALL,EXP=exp000 scripts/pipeline.sh
#    (exp000 uses its own driver — the pipeline's mech/control split does
#     not apply to the baseline; see the exp000 scratch dir once written.)

# 4. Budgets and noise floors in the adapter are placeholders until exp000
#    writes trajectory/profiles/calibration.json.
```

## Layout

```
autovidya_visdrone/
├── program.md                              # agent protocol — guard rails + methodology
├── README.md                               # this file
├── CLAUDE.md                               # session standing orders
│
├── core/                                   # HARNESS (model-agnostic, frozen — copied from v5)
│   ├── adapter.py                          #   Adapter Protocol (+ control contract)
│   ├── discipline/                         #   P0 / P9 / P10 / P11 gates
│   └── schemas/                            #   JSON schemas for all trajectory records
│
├── adapters/
│   └── yolo26n_visdrone_scratch/           # ADAPTER (model-specific)
│       ├── adapter.py                      #   implements core.adapter.Adapter
│       ├── sites.py                        #   YOLO26n named sites (unchanged from v5)
│       ├── generic_control.py              #   P9 param-matched control blocks
│       ├── configs/yolo26n-visdrone.yaml   #   nc=10 model yaml (rendered, Tier-3)
│       └── program.md                      #   adapter-specific protocol + hazards
│
├── setup/create_visdrone_lists.py          # dataset locks (lists + yamls + manifest)
├── scripts/
│   ├── pipeline.sh                         # THE reusable self-starting sbatch runner
│   └── smoke_test.py                       # harness wiring check
├── visdrone_manifest.json                  # tracked sha256 fingerprint of all locks
└── trajectory/                             # ALL agent-writable state
    ├── experiments/                        #   per-experiment JSONs
    ├── profiles/                           #   baseline + calibration profiles
    ├── insights.md                         #   insight pool (inherited ops/methodology entries)
    ├── inherited_v5_scientific.md          #   COCO-2k scientific results — SEALED until P0 done

    ├── index.json                          #   CWM log
    └── scratch/                            #   per-experiment scratch space
```

## Status

Scaffold instantiated 2026-08-21 from autovidya_v5. Open items before the
trajectory can start:

- [x] Dataset converted + verified (train 6,471 / val 548 / test-dev 1,610)
- [x] Frozen eval set decided (val + test-dev, 2,158 — user, 2026-08-21)
- [ ] `setup/create_visdrone_lists.py` run on this machine (lists + manifest)
- [ ] `ultralytics_src` verified clean, commit recorded
- [ ] exp000 driver written; exp000 run (budgets, noise floors, sealed-seed
      baseline legs, diagnostics, instrument check) →
      `trajectory/profiles/calibration.json` + `baseline.json`
- [ ] First mechanism prereg (P2-evidence-fusion family, with the regime
      argument) — written WITH the user
```
