# exp007 proposal tournament — 2026-09-10 (after exp006 Reject)

Ledger: Winner exp001 (P2 head); the decomposition (exp005, exp006) says
the gain is the conjunction of stride-4 evidence and grid (ins_011); exp003
(duplicates) and exp004 (emphasis) are nulls. fm_small_gt_missed is now
well characterised. fm_sibling_class_confusion (van->car 58%, bicycle->
motor 57%, people->pedestrian 25%) is the one measured failure mode with NO
experiment yet. Family-diversity: exp006 was evidence-fusion; anything else.

Criteria (fixed): (1) fm fit (2) P8 (3) effect vs 0.005 floor (4) cost (5) ledger distance.

## Candidates

### C3' — Sibling-pair margin objective (ECOC, coding theory) — objective, driver-side
- Source: error-correcting output codes (Dietterich & Bakiri 1995). The
  five sibling pairs are at Hamming distance 1 in the one-hot code; the
  objective adds softplus(margin - (z_c - z_sibling)) over positives of
  paired classes, gain 0.5, margin 2.0 — a margin on the one separating bit.
- P8: nearest CV block = hierarchical softmax (YOLO9000) / pairwise margin
  losses. Difference: no tree, no extra outputs; a targeted margin on
  measured confusion pairs, fixed from Phase-0 data. v5 exp006 ran ECOC
  bits on COCO-2k: null (`ins_v5_cls_gap_epiphenomenon`). Regime argument:
  there the target was a train/val cls GAP; here the target is confusion
  measured ON the eval as the outcome, and mAP50 is class-aware, so a
  reduction maps directly to the metric.
- Honest prior: exp001 showed resolution is the lever; a 30-px van vs car
  may be irreducible at 640. Prior 0.5-1x floor. Control: same loss on a
  fixed random NON-sibling pairing (placebo; same form and mass).
- Cost: zero params, zero compute. Scores: 4 / 3 / 2 / 5 / 5 = 19.

### C15 opponent-pair coding — 18 (same fm, structural, thinner P8).
### C5 relaxation labeling — 16; C4 predictive-coding recurrence — 15.

## Decision: exp007 = C3', family `code-margin`, kind aux_objective (E2ELoss(loss_fn=SiblingMarginLoss) pattern, tripwire).
Discriminators: D1 sibling-pair confusion rate (share of located objects of a paired class assigned the sibling class, adapter matcher, conf .25) must FALL on every seed; D2 mean AP50 of the ten classes' paired members must rise more than that of unpaired... all classes are paired here, so D2 = mean AP50 of the five WORST-confused classes (van, bicycle, awning-tricycle, people, truck) must RISE on every seed.
