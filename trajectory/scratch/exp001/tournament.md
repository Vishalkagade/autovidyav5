# exp001 proposal tournament — DRAFT for user review (2026-09-08)

Inputs: `trajectory/profiles/baseline.json` (fm_small_gt_missed,
fm_dense_scene_f1_collapse, fm_sibling_class_confusion), calibration
(S2 = 80 ep, floor 0.0050 mAP50, baseline 0.2704), inherited insights (now
open: `inherited_v5_scientific.md`). Fixed criteria from program.md, scored
1 (weak) to 5 (strong):

  (1) fit to a measured fm_*        (2) P8 regime/novelty argument
  (3) expected effect vs 0.005 floor, stated mechanistically
  (4) GPU + wall-clock cost          (5) distance from saturated pairs (ledger empty: all 5)

## Candidates

### C1 — P2 evidence fusion (topology: stock `yolo26-p2.yaml`, nc=10)
- fm: fm_small_gt_missed (primary), fm_dense_scene_f1_collapse (secondary).
- Mechanism: route the stride-4 feature (backbone idx 2, computed but unused
  today) into the head; Detect gets a P2 output. Measured: +10,652 params
  (0.4% of baseline; cap is 250,770), forward 1.5x (4.7 -> 7.1 ms / batch 8
  at 640).
- P8: v5 exp007 tested this on COCO-2k: -0.0222, the largest harm of any v5
  mechanism (`ins_v5_p3_fusion_locally_optimal`). Regime argument: VisDrone
  median GT side is 22 px native = ~8 px at the 640 input, i.e. UNDER one
  stride-8 cell; 68% of GT is COCO-small and the model recovers 23% of it.
  On COCO-2k the median object spanned many stride-8 cells, so stride-4
  evidence was redundant noise there. The mechanism's stated computation
  (a finer-grid evidence path fused before decision) is exactly what the
  failure mode says is missing. This is the trajectory's founding hypothesis
  test — program.md names it as first-class.
- Dual role: it is also the POSITIVE CONTROL for the harness. P2 heads are
  reported to help on VisDrone in the literature. If the full gate chain
  cannot certify it, every later null is uninterpretable (harness may be
  blind). Pre-register that reading in prereg.json.
- P9 control (topology-kind, designed here): same +10.6k params spent at
  stride 8 — widen head layer 16 (C3k2 P3-out) so the added parameter count
  matches within +-10%, NO stride-4 route. Isolates "finer evidence" from
  "more head capacity". Note the control will be CHEAPER in compute than the
  mechanism (compute is part of what P2 buys; record it, do not hide it).
- Cost: mechanism 3 seeds x ~2 h + control 3 x ~1.4 h + units. ~11 GPU-h.
- Scores: (1) 5  (2) 5  (3) 4 — literature P2 gains at nano scale are
  +1-3 mAP50 points, 2-6x the floor  (4) 3  (5) 5.  TOTAL 22.

### C2 — Divisive normalization on the one-to-one Detect logits
- Source domain: visual neuroscience — canonical cortical computation
  (Carandini & Heeger, Nat. Rev. Neurosci. 2012).
- fm: fm_dense_scene_f1_collapse (8.7% duplicate predictions at conf .25;
  precision 0.38 at best-F1).
- Mechanism: per scale, logit_i <- logit_i / (sigma + sum_j w_j * relu(logit_j))^n
  over a learned local neighbourhood w (3x3 or 5x5), sigma and n learned,
  applied inside the one2one branch before assignment; trained from step 0.
- P8: nearest CV block is NMS (post-hoc, non-differentiable, absent in
  YOLO26) or self-attention over queries (DETR). Difference: pooled-energy
  division, local, differentiable, no attention weights — a suppression
  field, not a mixing operator.
- Risk: thin wrapper (~1-3k params). v5: every 8-20k wrapper landed inside
  +-0.5 floor. Expected effect: duplicates -> precision; mAP50 effect
  bounded by the 8.7% duplicate mass, plausibly ~1x floor. Weak on (3).
- Scores: (1) 4  (2) 4  (3) 2  (4) 5  (5) 5.  TOTAL 20.

### C3 — Hierarchical error-correcting output codes for sibling classes (aux objective, driver-side)
- Source domain: coding theory (ECOC; Dietterich & Bakiri 1995).
- fm: fm_sibling_class_confusion (van->car 58%, bicycle->motor 57%,
  people->pedestrian 25%).
- P8: v5 exp006 ran ECOC on COCO-2k: closed 10% of the cls gap, exact-null
  mAP (`ins_v5_cls_gap_epiphenomenon`). Regime argument: on COCO the gap was
  a train/val symptom; here the confusion is measured ON the eval as the
  outcome, and mAP50 is class-aware, so a class fix maps directly to the
  metric. Sibling structure is explicit (3 pairs), so the code can be
  designed, not learned.
- Cost: no params, no extra compute. Cheapest candidate.
- Weakness: aux objectives shape the classifier, not the evidence; a van at
  30 px may be genuinely indistinguishable from a car at 640. (3) is
  uncertain; honest prior ~1x floor.
- Scores: (1) 4  (2) 3  (3) 2  (4) 5  (5) 5.  TOTAL 19.

### C4 — Predictive-coding recurrence at head.c3k2_out_p3
- Source domain: neuroscience (Rao & Ballard 1999): iterate the P3 block
  twice, feeding back the residual between the first output's coarse
  reconstruction and the P3 input. fm_small_gt_missed.
- P8: nearest CV block is iterative refinement (RefineDet, cascade R-CNN).
  Difference: same weights re-applied with an error signal, not a second
  stage. Compute 2x at P3 (largest head activation) — `ins_v5_zero_params_
  not_zero_cost` warns this costs like a mechanism twice its size.
- Scores: (1) 3  (2) 3  (3) 2  (4) 2  (5) 5.  TOTAL 15.

### C5 — Relaxation labeling over neighbouring detections (head.detect)
- Source domain: constraint satisfaction (Hummel & Zucker 1983): iterate
  class probabilities p_i <- p_i * (1 + sum_j C(c_i, c_j) p_j) with a learned
  compatibility matrix C over spatial neighbours. fm_sibling_class_confusion.
- P8: nearest CV block is relation networks / CRF-as-RNN. Difference:
  compatibility on class labels only, no feature mixing. Borderline: a
  reviewer could call it a CRF layer. (2) weak.
- Scores: (1) 3  (2) 2  (3) 2  (4) 4  (5) 5.  TOTAL 16.

### C6 — Anti-aliased downsampling (Nyquist argument) — REJECTED at P8
- Source domain: signal processing. But the nearest CV block IS the
  mechanism (BlurPool, Zhang 2019). Standard CV primitive. Not admitted.

## Ranking

| # | candidate | fm | total |
|---|---|---|---|
| 1 | C1 P2 evidence fusion | small_gt_missed | 22 |
| 2 | C2 divisive normalization | dense_scene | 20 |
| 3 | C3 hierarchical ECOC | sibling_confusion | 19 |
| 4 | C5 relaxation labeling | sibling_confusion | 16 |
| 5 | C4 predictive-coding recurrence | small_gt_missed | 15 |

## Recommendation: exp001 = C1

Reasons beyond the score: it is the hypothesis test the trajectory exists
for; it is the positive control the harness needs before any null means
anything; its P9 control is clean (capacity at stride 8 vs evidence at
stride 4). Family for P2 bookkeeping: `evidence-fusion`. Under the family-
diversity rule exp002 must come from a different family — C2 or C3 is the
bench.

## Pre-registration sketch for C1 (to be written as prereg.json only after user approval)

- Discriminator (pre-registered): small_recall (per_scale_metric) must rise
  by >= 0.05 absolute on every working seed; if mAP50 rises but small recall
  does not, the gain is NOT the stated mechanism -> classify from P10 but
  record `discriminator_failed`.
- Kill bar at seed 42: mech - baseline(seed 42 = 0.26970) < -1x floor
  (-0.0050) -> Kill.
- Futility after seed 123: two-seed mean delta <= -0.0050 -> Reject.
- P9 gate: seed-avg delta >= 0.0050 -> run the stride-8 widened control.
- Positive-control reading: if C1 ends below Hold, log `harness_sensitivity_
  unproven` in index.json and STOP for the user before exp002.
- Wall-clock: ~2 h per S2 seed (1.5x baseline), two seeds in parallel.
