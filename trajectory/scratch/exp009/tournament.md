# exp009 — P4 targeted retry of exp008 (2026-09-11)

Not a tournament round: program.md P4 permits ONE corrected retry when a
scientific failure names a single diagnosable design flaw with a directional
prediction from the failure dynamics. exp008 (SPRT cascade, Reject) does:

- Failure dynamics: mAP50 up on every seed (+0.0061/+0.0018/+0.0059) and
  mAP50-95 up on every seed, small-class AP50 +0.014 (0.61x the P2 head),
  yet per-image F1 at the working point DOWN on every seed (-0.0071,
  CI [-0.0094,-0.0047]) and recall@.25 down on every scale. Ranking right,
  operating point wrong.
- Diagnosis (measured, `trajectory/scratch/exp008/state/dup_mech_S2.json`):
  duplicate fraction 0.112/0.112/0.113 vs baseline 0.084/0.087/0.087,
  non-duplicate FP fraction DOWN, detections per image up at every threshold,
  per-image F1 below the baseline at every threshold >= 0.15 including the peak.
  The parent stride-8 anchor of a selected cell keeps emitting its decision
  while its sub-anchor inherits the parent logit (z1 + residual): two boxes per
  object. The one-to-one assigner also labels the parent a negative whenever
  the sub-anchor wins, so the shared logit gets opposite-signed gradients.
- Correction (one change): `parent="mask"` — selected cells emit no stage-1
  decision in either branch. This is the SPRT rule itself (no decision while
  sampling continues), so it is a correction of the implementation, not a new
  mechanism. Params, compute, q, B_hi, control design unchanged.
- Prediction: dup_frac back in the baseline band (<= 0.095 every seed, D4),
  precision and per-unit F1 >= baseline, AP gains retained.
- Family diversity: consecutive `sequential-evidence` experiment, overruled
  explicitly — P4 retry of a scientific failure with a measured flaw.
- Hard cap: this is exp008's only retry. If D4 fails or the chain rejects,
  the answer is "next proposal" (bench: C20 SIC 19, C4 predictive coding 15).
