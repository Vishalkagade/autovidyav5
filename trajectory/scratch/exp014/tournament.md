# exp014 — Phase-2 refinement of exp011, axis R3-coupling (2026-09-16)

R1-formula is closed (P5: exp012, exp013). ins_022 (level-off test): the
gated models' stride-8 heads cede products to the stride-4 level wherever
the band sits (anchor -0.025 with the level off, absband -0.054), while the
ungated level is neutral; the loss is assignment ownership, not selection.

### R-C' — Scale-owned assignment (R3-coupling)  — CHOSEN
- In both assigners (one-to-many, one-to-one) stride-4 anchors are candidates
  only for GT with longer side < 32 px (input scale); larger GT is assigned
  among stride-8/16/32 anchors. Selection rule = the anchor's (top-q band).
- Prediction: on SKU (40-px products) stride 4 never owns products -> the
  stride-8 head is trained as owner -> no recall loss from band membership;
  live stride-4 anchors are trained as negatives on products -> no FP flood.
  On VisDrone (68% of GT < 32 px) ownership of small objects is unchanged,
  medium objects go to stride 8 (the anchor's medium coverage in the band
  was 45%; medium recall unchanged or better). VOC unchanged (inert).
- Cost: none. Nearest analog: per-level scale ranges (FCOS/ATSS); here
  applied only to the gated level and only as an ownership constraint.
- Scores: 5 / 3 / 5 / 5 / 5 = 23.

### R-C — Frozen (EMA) selection during training (R3-coupling) — 4/3/3/4/4 = 18.
  Stabilises band membership but does not stop stride 4 from owning
  products; ins_022 says ownership is the loss. Bench.

### R-E — Gate at inference only, train ungated (R1-position) — 3/3/3/5/4 = 18.
  Trains like the ungated level (neutral on SKU) but then the band removes
  anchors that own objects at inference (ungated + band at inference:
  0.839 vs 0.906, measured in the level-off test). Rejected by measurement.
