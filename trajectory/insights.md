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
  evidence: exp001. status: stale — superseded by ins_006 (exp001 reclassified
  Provisional Winner -> Winner at P11).
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
- `ins_006` — claim: P2 evidence fusion is a VisDrone WINNER: sealed seeds
  1000/2000 replicate at +0.0199/+0.0160 mAP50 (4.0x/3.2x floor; five-seed
  range +0.011..+0.020, all positive), combined five-seed per-unit test vs
  baseline CI [+0.0094,+0.0131] (p=7e-41), vs control (working seeds; control
  margin 3.3x floor so it did not need to replicate) CI [+0.0076,+0.0121].
  The same computation COCO-2k falsified at -0.022 (`ins_v5_p3_fusion_
  locally_optimal`) is a certified gain where the median object is under one
  stride-8 cell. This is direct support for the co-evolved-optimum reading of
  YOLO26+COCO: the head's stride set is tuned to COCO's object sizes, not
  optimal in general. Claim discipline: "a VisDrone Winner", not "a model
  improvement" — the cross-dataset track (SKU-110K win + Pascal VOC
  no-regression) is still open. evidence: exp001. status: active.
- `ins_007` — claim: a log-domain normalisation with sign-unconstrained
  surround weights diverges to NaN mid-training under AMP (exp002: epoch ~33
  of 80, after tracking the baseline exactly to epoch 30). Source-domain
  constraints (non-negative pooled energy) must be enforced by
  parameterisation, and modules must self-check adversarial parameter values;
  200-iteration overfit probes cannot catch mid-training divergence.
  evidence: exp002. status: active (operational).
- `ins_008` — claim: on VisDrone the NMS-free head's residual duplicates
  (8.4-8.7% of predictions at conf .25) are not what limits dense-scene
  performance: a learned divisive-normalization field reduced duplicates on
  every seed (-0.003 to -0.006 absolute) and pure false positives (~-0.01)
  with a null effect on mAP50 (mean -0.00004) and per-unit F1 (CI
  [-0.0031,+0.0010]) and no gain in >=100-box scenes. Proposals citing
  fm_dense_scene_f1_collapse must target recall under crowding (missed
  neighbours), not redundancy. evidence: exp003. status: active.
- `ins_009` — claim: re-allocating gradient toward small objects is zero-sum
  on from-scratch YOLO26n/VisDrone: Horvitz-Thompson 1/p(scale) loss
  weighting (4.4x small, verified applied, weight-area corr -0.37) raised
  small recall +0.026 and small-class AP50 +0.007..+0.009 while medium/large
  recall fell -0.066/-0.054 and vehicle classes lost 0.012-0.033 AP50, net
  -0.0114 mAP50 (Kill at seed 42). Emphasis moves detections between scales;
  only added evidence (exp001, stride-4 path) adds them. Objective-side
  small-object proposals need a mechanism that does not borrow capacity from
  the other scales. evidence: exp004 (+ exp001 contrast). status: active.
- `ins_010` — claim: decision slots without evidence do not recover exp001's
  gain: four sub-cell hypotheses per stride-8 cell (virtual stride-4 anchor
  grid, stride-8 features; +4.3k params) gave +0.0030 mAP50 (0.6x floor),
  per-unit F1 -0.0033 (CI [-0.0054,-0.0013], p=0.001, worse on every seed),
  small recall@.25 down on every seed, and only 0.26x of exp001's small-class
  AP50 gain (pre-registered reading: undetermined at the 0.25 boundary; the
  point estimate sits with the evidence account). Extra hypotheses re-rank
  small objects rather than find them. Combined with exp004 (emphasis is
  zero-sum) the ledger says the P2 head's value is the stride-4 EVIDENCE
  path. evidence: exp005 (+ exp001, exp004). status: active.
- `ins_011` — claim: the P2 head's gain (exp001, +0.015) is the CONJUNCTION
  of stride-4 evidence and a stride-4 decision grid, not either alone:
  stride-4 evidence carried losslessly (polyphase / space-to-depth route,
  +10.3k params) into the stride-8 P3 fusion gives +0.0002 mAP50 and 0.08x
  of exp001's small-class gain (exp006, evidence account refuted at the
  pre-registered 0.25x bar); stride-4 decision slots with stride-8 evidence
  give 0.26x and negative per-unit F1 (exp005). A finer decision grid only
  pays when the features under it are finer too, and finer features only
  pay when a decision is made at their resolution. Cheap halves of the P2
  head do not exist at this scale; the 3x wall-clock is the price.
  evidence: exp006, exp005, exp001. status: active.
- `ins_012` — claim: sibling-class confusion on VisDrone (van->car 0.6,
  bicycle->motor 0.56 of located objects) is not a decision-boundary
  problem: a verified ECOC-style logit margin on the five measured sibling
  pairs (lambda 0.5, margin 2.0) left the confusion rate unchanged (0.1104
  vs 0.1100), never reached its margin (mean term 0.85 after 80 epochs), and
  cost -0.0072 mAP50 with AP down on 8/10 classes (Kill). The classifier
  cannot be pushed to separate what the features do not resolve; sibling
  confusion belongs to fm_small_gt_missed's resolution story, not to the
  objective. Objective-side proposals citing fm_sibling_class_confusion are
  not worth GPU without new evidence. evidence: exp007 (+ exp001). status:
  active.
- `ins_013` — claim: the P2 evidence-fusion head (exp001's VisDrone Winner)
  REGRESSES on Pascal VOC (normal-object data, from scratch, same protocol):
  seed-avg mAP50 -0.0063 (VOC floor 0.0085; seeds +0.005/-0.017/-0.007),
  per-image F1 -0.0050 with bootstrap CI [-0.009, -0.001] (Wilcoxon
  p=0.056), small-object recall down on every seed (0.357 -> 0.332 avg).
  The pre-registered no-regression bar FAILED on its per-unit clause.
  Together with v5's COCO-2k result (-0.022) this places the mechanism as a
  small-object specialisation, not a model improvement: the claim stays "a
  VisDrone Winner". The cross-dataset track's SKU-110K leg is now moot for
  the model-improvement claim and is not run. evidence: confirm_voc (state/
  assembly.json), exp001, v5 exp007. status: active.
