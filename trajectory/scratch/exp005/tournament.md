# exp005 proposal tournament — 2026-09-09 (after exp004 Kill)

Ledger: exp001 Winner (evidence-fusion, stride-4 route: +0.015); exp003
Reject (lateral-inhibition: duplicates irrelevant, ins_008); exp004 Kill
(estimator-weighting: emphasis is zero-sum, ins_009). The open question the
ledger poses: is exp001's gain FINER EVIDENCE (stride-4 features) or MORE
DECISION SLOTS per area (4x anchors)? Measured today: 21.5% of eval GT
centres share a stride-8 cell with another GT (7.8% at stride 4, 42% at
stride 16; train 21.0%). One prediction per cell caps that fifth.

Criteria (fixed): (1) fm fit (2) P8 (3) effect vs 0.005 floor (4) cost (5) ledger distance.

## Candidates

### C17 (new) — Multi-hypothesis decision at P3 (target tracking / mixture outputs)
- Source: multiple-hypothesis tracking (Reid 1979) and mixture density
  outputs (Bishop 1994): a resolution cell that may contain several targets
  emits K hypotheses, each with its own position and class, and assignment
  resolves them one-to-one. Here K=4 sub-cell hypotheses per stride-8 cell
  at P3, laid out as a virtual stride-4 grid (pixel-shuffle of the cls/box
  output convs), evidence unchanged (stride-8 features). fm_small_gt_missed
  + the crowded-neighbour half of fm_dense (ins_008).
- P8: nearest CV blocks = sub-pixel conv (ESPCN) and multi-anchor heads
  (YOLOv3). Difference: hypotheses are POSITION offsets on the decision
  output with no size priors, competing under one-to-one assignment; ESPCN
  upsamples features, anchors encode sizes. Honest note: computationally a
  pixel-shuffle on the head's last convs — the experiment's value is the
  discrimination it buys, stated in the prereg.
- Effect story: same decision density as the P2 head (exp001) at ~1/3 of
  its compute and 0 stride-4 evidence. Prediction registered BEFORE data:
  if slots are the mechanism, this recovers >= half of exp001's small-class
  gain; if evidence is, it recovers little. Either answer is a finding.
- Cost: +4.3k params, ~negligible FLOPs; assigner sees 4x anchors at P3
  (same as exp001, expandable segments on). Control: adapter
  GenericResidualConv on the P3 cls outputs (one2many + one2one), ~2.15k
  params each = 0.99x.
- Scores: 5 / 3 / 4 / 5 / 5 = 22.

### C3 hierarchical ECOC — 19; C15 opponent coding — 18; C5 relaxation labeling — 16; C4 predictive-coding recurrence — 15 (unchanged).

## Decision: exp005 = C17, family `multi-hypothesis-decision`, site `head.detect`, kind site_wrap (SurgeryTrainer).
