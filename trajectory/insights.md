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
- `ins_014` — claim: a budgeted second sample of stride-4 evidence on the
  undecided band (SPRT cascade, q=0.25 of cells, +122.7k params, exp008)
  recovers 0.61x of the P2 head's small-class AP50 gain (+0.014 seed-avg;
  pre-registered reading: the sparse cascade DOES recover the evidence
  effect) and raises mAP50 on every seed (+0.0061/+0.0018/+0.0059, seed-avg
  +0.0046 = 0.92x floor) and mAP50-95 on every seed, yet LOWERS the
  per-image F1 at the working threshold on every seed (-0.0071, CI
  [-0.0094,-0.0047], p<1e-4) with recall@.25 down on every scale (medium
  -0.017 on all seeds). Ranking improves, the operating point degrades —
  the exp005 signature (ins_010) reappears with real stride-4 evidence
  under the slots. Measured cause (state/dup_mech_S2.json, threshold
  sweep): duplicate fraction 0.112/0.112/0.113 vs baseline 0.084/0.087/
  0.087 while NON-duplicate false positives fell (0.137/0.126/0.132 vs
  0.164/0.154/0.156) and per-image F1 sits below the baseline at every
  threshold >= 0.15 including its peak — the selected cell's stride-8
  parent keeps emitting a decision while its sub-anchor inherits the
  parent logit, so one object gets two boxes. The evidence stage works
  (fewer real FPs, higher AP); the decision stage double-counts. Reject
  by the per-unit rule; the P9 gate never opened, so the selection rule
  (SPRT band vs random cells) is untested. P4 retry = exp009 (parent
  masked). evidence: exp008 (+ exp005, exp001). status:
  active.
- `ins_015` — claim: in a sparse stride-4 second stage, the duplicates come
  from the ADDITIVE accumulation rule, not from the parent anchor: masking
  the stride-8 parent of every selected cell (exp009, exp008's one P4
  retry) RAISED the duplicate fraction (0.129/0.127/0.123 vs exp008's
  0.112/0.112/0.113, baseline 0.085) while non-duplicate FPs stayed at the
  exp008 level, so the four sub-cells of a selected cell co-fire on the
  parent logit they all inherit (logit = z1 + residual; the negatives must
  cancel a shared positive prior and do not). Removing the parent also
  removed its real work: medium recall@.25 -0.042, small-class AP50 gain
  0.61x -> 0.26x of the P2 head's, mAP50 -0.0011 seed-avg (Reject). The
  stage-1 decision is needed in band cells; what must change is that
  sub-cell decisions be their own logits (as in exp001's real stride-4
  level, where per-unit F1 rose). evidence: exp009 (+ exp008, exp001).
  status: active.
- `ins_016` — claim: the SPRT band is an informative selector for small
  objects: the top-q (q=0.25) undecided stride-8 cells hold 85% (exp010) /
  88% (exp008) of small GT centres on the frozen eval, 3.4x the 25% a
  random budget covers; what the sparse cascade lacks is decision capacity,
  not selection — its thin second stage loses small recall@.25 (-0.015)
  where exp001's full stride-4 level gains (+0.009), and its residual
  duplicate rate (0.102) already matches exp001's ungated stride-4 level
  (0.098; baseline 0.085), so duplicates are a property of 4x anchors, not
  the blocker. evidence: exp010/state/coverage_seed42.json, exp001/state/
  dup_mech_seed42.json, exp008, exp001. status: active (exp010 chain
  incomplete when written).
- `ins_017` — claim: the sparse stride-4 cascade's failure is at the
  DECISION stage and is not the duplicate rate: three decision rules
  (additive accumulation, parent masked, own logits with the stage-1
  posterior as a feature: exp008/009/010) all raise mAP50-95 on every
  seed and all lose per-image F1 at the working point on every seed
  (-0.0071 / -0.009 / -0.0084); cutting duplicates from 0.112 to 0.102
  (exp001's ungated level: 0.098) moved per-unit F1 the wrong way, while
  small recall@.25 fell ~-0.014 on every exp010 seed. The band selects the
  right cells (85% of small GTs, ins_016) but a 4x4-window trunk on raw
  backbone P2 plus one P3 cell cannot produce confident sub-cell decisions;
  exp001's neck-fused P2 level can. P2 saturation: (sequential-evidence,
  3-level head.detect) is closed; the family's remaining question — does
  band gating of a full stride-4 level beat random gating — is exp011.
  evidence: exp008, exp009, exp010 (+ exp001, ins_016). status: active.
- `ins_018` — claim: WHERE a stride-4 decision level is allowed to decide
  is the mechanism, not the level itself. exp001's yolo26-p2 topology with
  the P2 level gated by a sequential-test band (stride-8 cells whose
  one-to-one objectness is below B_hi=0.5, top q=0.25 by objectness; the
  rest of the level masked) is a VisDrone WINNER: mAP50 +0.0155 on the
  working seeds (3.1x floor) and +0.0155/+0.0143 on the sealed seeds; per-
  image F1 vs baseline +0.0106 over all five seeds (CI [+0.0088,+0.0125],
  p 2e-37, every seed positive); vs the RANDOM-gated control (identical
  topology, params and compute) +0.0104 (CI [+0.0082,+0.0127], p 4e-19),
  while random gating itself is null vs baseline (+0.0011, ns). The band
  holds 88% of small GT centres at a 25% cell budget (random: 25%) and the
  gated level keeps 1.04x of the ungated level's small-class AP50 gain with
  duplicates at the ungated level's rate. Source domain: Wald's SPRT
  (decide when the posterior leaves the band, otherwise sample again).
  Cross-domain in the P8 sense; the trajectory's first. Confirmation track
  (user strategy) pending: VOC no-regression (prediction: the band removes
  exp001's VOC regression, ins_013) and SKU-110K win. evidence: exp011 (+
  exp001, ins_016, ins_017). status: active.
- `ins_019` — claim: band gating REMOVES the P2 level's cross-dataset
  regression. On Pascal VOC (normal objects, from scratch, same protocol
  and the same baseline seeds as exp001's leg) the band-gated stride-4
  level meets the pre-registered no-regression bar: mAP50 -0.0037 seed-avg
  (floor 0.0085; seeds -0.0002/-0.0079/-0.0030) and per-image F1 -0.0021
  with CI [-0.0059,+0.0017] (p 0.46), where exp001's ungated level failed
  it (-0.0063; F1 -0.0050, CI [-0.0090,-0.0010], ins_013). Mechanism of
  the difference, measured (CORRECTED 2026-09-15: the first write-up used
  predict's rectangular grid to index cells and reported ~21% coverage;
  the square-grid recomputation is in the per-unit files): on VOC the
  band does land on the objects — 62-70% of medium and 42-46% of large GT
  centres (n=2,002 / 9,947), with only ~0.003% of cells "decided" at B_hi
  — yet the gated stride-4 level emits <1% of the detections at conf .25
  (anchor_analysis.json), i.e. it is inert because the one-to-one
  assignment gives large objects to the stride-8/16/32 anchors, not
  because the band withholds it. On VisDrone the same rule covers 88% of
  small GT centres and the level emits 44% of the detections. The band
  is a data-adaptive switch in effect, not by gating alone: it spends the
  stride-4 decision where the first stage is uncertain, and where objects
  are large the level has nothing to win in the assignment.
  Confirmation strategy status: VOC no-regression PASS; SKU-110K win leg
  running. evidence: confirm_voc/state/assembly_bandgate.json (+ ins_013,
  exp011). status: active.
- `ins_020` — claim: thMeasured (square-grid coverage, corrected 2026-09-15;
  the first write-up's "21%, below random" figure came from a rectangular-
  grid indexing error): on SKU the coarse grid is already DECIDED (o >=
  B_hi) on 52% of medium and 29% of small product centres — the only
  dataset where the decided clause is active at all (VisDrone 37% of
  medium, VOC 17%) — so those cells are excluded from the band by design;
  the band holds 20-29% of medium and 45-66% of small centres, i.e. ~44% of
  the products the first stage has NOT decided. The gated level is not
  inert there (10% of detections at conf .25), and it hurts: medium-object
  recall@.25 falls -0.04/-0.07/-0.07, precision and recall both drop, and
  the training-time val curve sits below the baseline from the first
  epochs (0.883/0.874/0.866 vs 0.888 at the end), so the loss is in
  training, not at the operating point. Reading: on dense medium-object
  shelves a decided product's neighbouring cells are in the band, and
  their live stride-4 anchors compete with the decided stride-8 anchor
  for the same product in the one-to-one assignment; the selection changes
  as the posterior trains, so the target flips between levels. Whether the
  cost is the 4-level topology itself or the gating is the question the
  SKU diagnostic (ungated exp001 level, prereg_diag.json) answers. operating point. The switch is therefore not free: where the first
  stage is already decided, the live quarter of the stride-4 level still
  competes in the one-to-one assignment, and on shelves with ~141 medium
  objects per image that competition costs recall. Whether the cost is
  the 4-level topology itself or the gating is the open question the SKU
  diagnostic (ungated exp001 level, prereg_diag.json) answers. evidence:
  confirm_sku/state/assembly_bandgate.json (+ ins_019, exp011). status:
  active.
- `ins_021` — claim: on SKU-110K the harm is the GATING, not the stride-4
  level: exp001's ungated level (same yolo26-p2 topology, same seeds and
  baselines, pre-registered diagnostic) is neutral-to-positive — mAP50
  +0.0016 seed-avg (floor 0.0016; seeds +0.0002/+0.0035/+0.0012),
  per-image F1 +0.0018 with CI [+0.0010,+0.0025] (every seed positive)
  — where the band-gated level lost -0.013 / -0.0144 (ins_020). So the
  4-level topology is harmless on dense medium-object shelves and the
  switch is what costs: selecting a quarter of the cells by the stride-8
  posterior, when the posterior has already decided half the products,
  puts live stride-4 anchors next to decided stride-8 anchors and the
  one-to-one targets flip between levels as the selection moves during
  training. Together with ins_019 (VOC: level inert, band harmless) and
  ins_018 (VisDrone: band = the mechanism), the band is a switch whose
  rule is right where coarse-grid uncertainty tracks the small objects
  (sparse aerial scenes) and wrong where the coarse grid is confident on
  a dense scene. A refinement of the RULE (not the level) is the natural
  next step; not started (user decision pending 2026-09-15). evidence:
  confirm_sku/state/assembly.json (+ ins_018-020). status: active.
