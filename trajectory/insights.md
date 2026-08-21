# Insights — YOLO26n / VisDrone from-scratch trajectory

## Inherited from prior trajectories

All entries in this section: `status: inherited` — advisory only, exempt
from the exp-id evidence rule (their evidence lives in `../autovidya_v5` and
earlier repos), never citable as evidence for any classification, never
counted by P2 or P3. IMPORTANT for this trajectory: COCO-2k falsifications
are NOT death sentences here — re-testing them on VisDrone is the founding
hypothesis. What inherits unconditionally is the OPERATIONAL knowledge and
the methodology lessons; the scientific nulls inherit only as "this is what
COCO said", to be argued against via P8's regime argument.

### From the bounded-gate regime (U-Net/Kvasir, pre-v5)

- `ins_bounded_gates_capped` — Bounded element-wise amplitude gates (Rayleigh
  CDF and 4 generic controls): no per-unit-significant effect in either
  direction; whole class capped in the bounded regime.
- `ins_bounded_spatial_coupling_fragile` — Spatial coupling in gate
  mechanisms: multi-seed fragility (Huber gate).
- `ins_bounded_asym_boundary_loss` — Asymmetric boundary-only auxiliary
  losses: one-sided error inflation.
- `ins_aggregate_metrics_lie` — Batch-weighted aggregation once produced a
  false +0.005 headline. Hence P10.

### From the COCO-2k from-scratch trajectory (v5, closed 2026-08 as structured null)

Scientific results (COCO-regime — the re-test targets):

- `ins_v5_p3_fusion_locally_optimal` (v5 exp004+exp007) — On COCO-2k the P3
  fusion's evidence mix was locally optimal against BOTH directions of
  modification: suppressing the coarse stream hurt (-0.0045); supplying an
  equal-width P2 fine stream hurt far worse (-0.0222, the largest harm of
  any v5 mechanism, with topology tripwires proving the pathway was real).
  THIS is the trajectory hypothesis: if P2-evidence fusion also fails on
  VisDrone, the mechanism is dead; if it works, the co-evolved-optimum
  reading of COCO gains direct support.
- `ins_v5_capacity_scaling_falsified` (v5 exp003+exp005) — Scaling the
  cooperative-cascade family 6x (19.7k → 118k params) ERASED its small
  positive margin instead of amplifying it. "Thin wrappers are merely
  capacity-starved" is not a sufficient account of nulls; more capacity can
  make a mechanism worse.
- `ins_v5_cls_gap_epiphenomenon` (v5 exp006) — An ECOC auxiliary loss
  verifiably closed ~10% of the cls generalization gap with an exact-null
  mAP effect (COCO-2k): there, the gap was a symptom, not a cause. If
  VisDrone's exp000 surfaces a cls-gap failure mode, cite this before
  targeting it.
- `ins_v5_tail_not_reachable_from_train` (v5 exp008) — Explicitly optimizing
  the per-image loss tail (CVaR_0.25, verifiably applied) made the val F1=0
  mass WORSE on COCO-2k. Whole-image failure mass there was a property of
  what the train set cannot teach, not of optimizer pressure.
- `ins_v5_scale_not_frequency` (v5 exp000) — COCO-2k baseline failure was
  structured by object scale and clutter, NOT class frequency (per-class AP
  uncorrelated with train instance count). Check whether VisDrone's exp000
  reproduces this structure before assuming it.

Methodology lessons (inherit with full force):

- `ins_v5_s1_screen_not_predictive` (v5 P12, after exp004) — A short
  screening stage did not predict full-budget outcomes (3 promotions, 2 into
  negative S2). Consequence baked into THIS trajectory's design: the screen
  is the first full-budget S2 seed, not a shorter stage (program.md
  "Screening", scripts/pipeline.sh).
- `ins_v5_training_deterministic_within_seed` (v5 exp004) — Identical pinned
  args reproduced mAP50 bit-for-bit and all per-unit values exactly. Paired
  comparisons have zero measurement error; the "noise floor" is entirely
  BETWEEN-seed variation. Verify once on VisDrone (cheap: rerun one short
  probe) before leaning on it.
- `ins_v5_seed_noise_wider_than_calibrated` (v5, 7 baseline seeds) — A
  3-seed max-min noise floor came from a lucky narrow cluster; 7 seeds
  showed ~1.6x that spread (SD 0.0032 vs floor 0.0063 max-min). Read exp000's
  floor as a lower bound; judge Winner margins against per-seed SD, and
  record per-seed baseline values verbatim in the experiment JSONs.
- `ins_v5_regression_to_mean_at_floor` (v5 exp002/exp003) — Sub-floor
  effects move seeds back toward the seed-average; never read a sub-floor
  delta as real without per-unit paired evidence.

Operational facts (inherit as standing hazards — see adapter program.md):

- `ins_v5_surgery_dropped_by_train` (v5 exp001) — `Model.train` rebuilds
  from yaml and silently drops in-place site_wrap surgery; a full stage
  trained vanilla before this was caught. SurgeryTrainer pattern MANDATORY.
- `ins_v5_aux_objective_dead` (v5 exp002) — The `aux_objective` mechanism
  kind is metadata with no plumbing; objective mechanisms are implemented
  driver-side with an applied-batches tripwire.
- `ins_v5_zero_params_not_zero_cost` (v5 exp004) — A parameter-free
  mechanism ran 2.3-3.1x baseline wall-clock (large-activation ops). Budget
  mechanisms by estimated compute, not param count.
- `ins_v5_list_predict_ooms` (v5 exp000) — ultralytics 8.4.x loads a list
  source as ONE batch; per-unit metrics must predict image-by-image.

## Findings (populate as trajectory runs)
