# exp004 proposal tournament — 2026-09-09 (after exp003 Reject)

Ledger: exp001 Winner (evidence-fusion, fm_small); exp003 Reject
(lateral-inhibition @ head.detect, fm_dense) with `ins_008`: duplicates are
not the dense-scene bottleneck — missed neighbours are. Family-diversity:
exp004 must not be lateral-inhibition. Instrument lessons: threshold-free
discriminators (ins_004); budget by wall-clock (ins_005); encode source-
domain constraints in parameterisation (ins_007).

Criteria (fixed): (1) fm fit (2) P8 (3) effect vs 0.005 floor (4) cost (5) ledger distance.

## Candidates

### C12 — Horvitz–Thompson detection-probability reweighting (statistical ecology) — objective, driver-side
- Source: distance sampling / Horvitz–Thompson estimation (Buckland et al.,
  Distance Sampling; Horvitz & Thompson 1952): an estimate over a population
  with size-dependent detection probability p(size) is unbiased only when
  each detected unit is weighted 1/p. Applied to the training objective:
  each positive target's cls+box loss is weighted by 1/p(scale bucket),
  with p MEASURED on the baseline (small 0.23, medium 0.60, large 0.73 —
  exp000), weights normalised to mean 1 over positives per batch so the loss
  scale is unchanged. fm: small_gt_missed (also the crowded-neighbour half
  of fm_dense per ins_008: missed neighbours are small).
- P8: nearest CV block = focal loss / class-balanced loss (re-weight by
  predicted confidence or class frequency). Difference: the weight is a
  measured detection probability by object SCALE, fixed from Phase-0 data —
  an estimator correction, not a hardness heuristic. It is a new training
  objective (allowed class: "source-domain training objectives active from
  epoch 0"), not capacity/renaming/rearrangement.
- Regime argument: on COCO p(small) is not far below p(large) for a tuned
  detector, so the correction is ~flat; here p_small/p_large = 0.31 and 68%
  of GT is small — the correction is 3x and covers most targets.
- Effect story: the optimiser currently spends gradient in proportion to
  what it already detects; HT re-centres it on what it misses. Prior: 1-2x
  floor. Honest risk: up-weighting may trade large-object precision.
- Cost: zero params, zero extra compute. Control: PERMUTED-WEIGHT PLACEBO —
  identical weight multiset per batch, randomly permuted across positives
  (same loss scale and variance, no size structure). Isolates "size-
  structured emphasis" from "reweighting noise".
- Scores: 5 / 3 / 3 / 5 / 5 = 21.

### C3 — Hierarchical ECOC (coding theory), fm_sibling — 4 / 3 / 2 / 5 / 5 = 19 (unchanged)
### C15 (new) — Opponent-pair class coding (Hering opponent-process theory), fm_sibling
- Sibling logit pairs (van/car, bicycle/motor, people/pedestrian,
  tricycle/awning) pass through a learned antisymmetric opponent transform
  y_a = x_a - b*x_b. Nearest CV block: a 1x1 conv on logits (= its own
  generic control). Difference is structure (sparse antisymmetric), which
  P8 treats as thin. Confusion at 30 px is a resolution problem (exp001).
- Scores: 4 / 2 / 2 / 5 / 5 = 18.
### C5 relaxation labeling — 16; C4 predictive-coding recurrence — 15.

## Decision: exp004 = C12, family `estimator-weighting`, kind aux_objective (driver-side loss, E2ELoss(loss_fn=HTLoss) pattern, applied-batches tripwire per ins_v5_aux_objective_dead).

## Pre-registration (prereg.json)
- Discriminators (threshold-free): D1 mean AP50 over the small-dominated
  classes (pedestrian, people, bicycle, motor; >=80% small GT) must RISE on
  every seed; D2 that rise must EXCEED the AP50 change of the large-
  dominated classes (bus, truck) on every seed (scale-structured gain).
- Kill -0.005 @42; futility -0.005; P9 gate +0.005; control = permuted
  placebo, same seeds.
- Prior stated before data: Hold or small Reject most likely; a Winner here
  would say the from-scratch optimum is estimator-biased toward what it
  already sees.
