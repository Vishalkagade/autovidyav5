# Insights — YOLO26n / VisDrone from-scratch trajectory

## Inherited from prior trajectories — operational + methodology ONLY

All entries in this section: `status: inherited` — advisory only, exempt
from the exp-id evidence rule (their evidence lives in `../autovidya_v5` and
earlier repos), never citable as evidence for any classification, never
counted by P2 or P3. These are facts about the harness, the framework, and
the statistics — they inherit with full force and carry no claims about
what VisDrone's baseline is bad at.

**The inherited SCIENTIFIC results (COCO-2k mechanism outcomes, failure-mode
structure, bounded-regime falsifications) live in
[`inherited_v5_scientific.md`](inherited_v5_scientific.md) and are SEALED
until `profiles/baseline.json` has committed candidate failure modes (P0
anti-anchoring — see that file's header and program.md). Do not open that
file during Phase-0.**

### Methodology (statistics + screening)

- `ins_aggregate_metrics_lie` (pre-v5) — Batch-weighted aggregation once
  produced a false +0.005 headline. Hence P10.
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

### Operational (harness + framework hazards — see also adapter program.md)

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

- `ins_001` — claim: ultralytics 8.4.23 `compute_ap` interpolates linearly to
  (recall 1, precision 0) beyond the last reached recall, so mAP50 on this
  eval RISES monotonically with the val conf threshold (0.273 @0.001, 0.300
  @0.05, 0.322 @0.1, 0.375 @0.25) — a phantom-area term, not better
  detection. evidence: exp000. status: active. Consequence: never compare
  runs at different conf thresholds; a mechanism that changes tail precision
  moves this term, so P10's per-unit F1 (conf .25, threshold-fixed) is the
  arbiter, never a headline mAP delta alone.
- `ins_002` — claim: at 640 px from scratch, VisDrone's instrument is viable:
  per-unit F1 degeneracy 1.5-1.7% (COCO-2k was 41-44%) and relative S2
  noise floor 0.018 — ~8x more headroom than the pre-registered 0.15 bar.
  evidence: exp000. status: active.
- `ins_003` — claim: P2 evidence fusion (stride-4 feature routed into the
  head), the mechanism COCO-2k falsified at -0.022 (`ins_v5_p3_fusion_
  locally_optimal`), gains +0.0151 mAP50 (3.0x floor; seeds +0.019/+0.011/
  +0.015) on VisDrone from scratch, while a param-matched stride-8 control
  sits at baseline (-0.0015; per-unit p=0.68). The gain is the finer evidence
  path, not capacity. Same-sign per-unit effect on every seed (P10 vs
  baseline CI [+0.0077,+0.0122], vs control [+0.0076,+0.0121]).
  evidence: exp001. status: active — PROVISIONAL until P11 resolves; if P11
  passes this is the first direct support for the co-evolved-optimum reading
  of COCO (a COCO-null mechanism that works where object scale breaks the
  equilibrium).
- `ins_004` — claim: recall-at-conf-0.25 by scale is a threshold-bound
  discriminator that misses AP-visible gains: exp001 raised small-class AP50
  by ~+0.03 (pedestrian, people, motor) but small_recall@.25 by only
  +0.008-0.013, failing a +0.05 bar. Pre-register threshold-free
  discriminators (per-scale or per-class AP deltas), never recall at a fixed
  confidence. evidence: exp001. status: active.
- `ins_005` — claim: "params added" mis-prices topology mechanisms by an
  order of magnitude here: the P2 route adds 0.4% params, 1.5x forward
  FLOPs, and 3.0x wall-clock (4x anchors -> assigner memory -> CPU fallback
  on dense batches). Budget by measured wall-clock on a real batch, and
  report the mechanism-vs-control compute asymmetry in every P9 record
  (extends `ins_v5_zero_params_not_zero_cost`). evidence: exp001. status:
  active.
