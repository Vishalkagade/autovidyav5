# exp002 proposal tournament — 2026-09-09 (after exp001 Provisional Winner)

New information since exp001's tournament: (a) the harness detects a real
3x-floor effect (positive control passed P9+P10; control A/A null, p=0.68);
(b) fm_small_gt_missed now has a Provisional Winner from the
`evidence-fusion` family — the family-diversity rule says exp002 comes from a
different family; (c) `ins_004`: discriminators must be threshold-free or
directional, never recall@conf; (d) `ins_005`: budget by wall-clock.

Criteria (fixed, program.md): (1) fm fit (2) P8 argument (3) expected effect
vs 0.005 floor (4) GPU + wall-clock (5) distance from saturated pairs.

## Candidates (bench from exp001 + one new)

### C2 — Divisive normalization on one-to-one class logits (head.detect)
- Source: visual neuroscience (Carandini & Heeger 2012). fm: dense_scene
  (8.7% duplicate predictions at conf .25; F1 0.28 in >=200-box scenes).
- Computation: y_i = x_i - n*log(sigma + sum_j w_j sigmoid(x_j)), learned
  3x3 per-class surround, per scale, inside the one2one cls head. 276 params.
- P8: nearest = NMS / DETR self-attention; difference = differentiable local
  suppression field, no attention, no feature mixing, trained from step 0.
- Effect story: fewer redundant high-score cells -> fewer duplicate
  detections -> precision, and sharper peaks in crowds. Bounded: duplicates
  are 8.7% of predictions; honest prior 0.5-1.5x floor.
- Cost: ~zero compute. SurgeryTrainer hazard (site_wrap) — mitigated by
  before/after asserts on trainer.model AND trainer.ema.
- Scores: 4 / 4 / 2 / 5 / 5 = 20.

### C3 — Hierarchical ECOC for sibling classes (aux objective)
- Source: coding theory. fm: sibling_class_confusion (van->car 58%).
- P8: v5 exp006 ECOC exact-null on COCO; regime argument = confusion is the
  eval outcome here, mAP50 is class-aware. But a code adds no pixels: a
  30-px van vs car is a resolution problem, not a decision-boundary one —
  and exp001 just showed resolution IS the lever on this dataset.
- Scores: 4 / 3 / 2 / 5 / 5 = 19.

### C12 (new) — Detection-probability reweighting (Horvitz–Thompson, statistical ecology)
- Source: distance sampling / HT estimator (Buckland et al.): correct for
  size-dependent detection probability by weighting each target 1/p(size).
  fm: small_gt_missed (p_small = 0.23 measured).
- P8: nearest = focal / class-balanced loss (weights by confidence or class
  frequency). Difference = weights by MEASURED detection probability per
  scale bucket. But this is a re-weighting of an existing loss — P8's
  exclusion of "capacity, renaming, or rearrangement" is close; it is not a
  new computation in the network.
- Scores: 5 / 2 / 3 / 5 / 5 = 20 (ties C2 on points; loses on P8 and on fm
  diversity — fm_small already has a Provisional Winner).

### C5 relaxation labeling — 16, C4 predictive-coding recurrence — 15 (unchanged).

## Decision: exp002 = C2 (divisive normalization), family `lateral-inhibition`, site `head.detect`.

Tie-break vs C12: C2 is a new computation in the head (P8-clean) and targets
an fm with no mechanism yet. C12 stays on the bench as the first candidate
if the P3 composition gate ever fires.

## Pre-registration (in prereg.json)
- Discriminators (directional, threshold-free where possible; ins_004):
  D1 duplicate fraction at conf .25 (adapter matcher) must FALL on every
  working seed; D2 mean per-image F1 on units with >=100 GT boxes (n=239)
  must RISE on every seed. Both required for the lateral-inhibition account.
- Kill bar -0.005 @42, futility -0.005 two-seed mean, P9 gate +0.005.
- Control: adapter GenericResidualConv on the same one2one_cv3[i] output,
  target 92 params per scale (1x1 10->4->10), ~1.0x mechanism params.
- Expected verdict, stated before data: Hold is the most likely outcome
  (thin wrapper). The experiment is worth its ~5 GPU-h because the fm is
  untouched and the control is exact.
