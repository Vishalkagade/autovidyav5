# autovidya_visdrone

VisDrone trajectory. Successor to `../autovidya_v5` (COCO-2k, closed as structured null).

## Hypothesis

YOLO26+COCO is a co-evolved local optimum (both directions of P3-fusion modification hurt on COCO).
Mechanisms may work where that equilibrium does not hold: tiny-object aerial imagery.

## Dataset (LOCKED facts)

- Path: `/home/atuin/v134ce/v134ce15/datasets/VisDrone` (images/ + labels/, YOLO format, 10 classes)
- Splits: train 6,471 / val 548 / test 1,610 (dir name is `test`, VisDrone name is test-dev)
- **Frozen eval set: val + test-dev = 2,158 images** (user decision 2026-08-21)
- Confirmation strategy: a Winner must replicate on 2 more datasets: one small-object set (SKU-110K; win required) and one normal-object set (Pascal VOC; no-regression bar, user decision 2026-09-07)

## Standing orders (carried over from autovidya_v5)

- **Autonomous operation (user order 2026-09-08).** Consider all information, decide, and keep experimenting — tournament -> prereg -> driver -> launch -> classify -> commit -> next. Do not stop to ask. Stop only for the program.md STOP conditions or a destructive action.
- **Headless sbatch only.** Never pty-attach for GPU work (SSH drop kills training). Self-starting job chains: first-S2-seed screen -> remaining seeds -> per-unit -> P9 control-deferral gate -> control.
- **Fairshare is depleted (~0.09).** Expect 1-2 day queue waits. Plan self-starting jobs; never park a live allocation on a question. Between experiments: decide and continue.
- **Surgery trainer pattern is MANDATORY** for site_wrap training. `Model.train` rebuilds from yaml and drops in-place surgery.
- **Model problem, not data problem.** Mechanisms must fix missing computation. Pre-register a discriminator before each experiment.
- **Quota:** home ~95/100G soft. Watch for output truncation from full quota.
- **Commits:** Conventional Commits, NO co-author trailers (user preference).
- Claude Code is the orchestrator. No frameworks.
