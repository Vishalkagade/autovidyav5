# exp012 — first Phase-2 refinement of the exp011 anchor (2026-09-15)

Anchor profile fields targeted: `sku110k.failure_account`, `sku110k.gt_by_cell_state`
(52% of medium products decided by stage 1; band lands on/next to them),
`sku110k.ungated_level_diagnostic` (the level is harmless, the gating is the
loss, ins_021), `visdrone.gt_by_cell_state` (88% of small GT in the band).

## Candidates (fixed criteria: fm fit / P8 / effect size / cost / distance from saturated axes)

### R-A — Size-aware band (axis R1-formula)
- Rule: a stride-8 cell may take the second sample only if it is undecided
  (o < B_hi) AND stage 1's own predicted box at that cell is sub-cell-scale
  (max side < 32 px at the 640 input, the COCO-small boundary); top-q among
  those, empty slots when fewer qualify. Everything else as the anchor.
- Source domain: still the SPRT, sharpened — "take another sample" is only
  worth it when the second sample carries new information; at stride 4 that
  means the object is below the stride-8 cell. Stage 1 already estimates
  the object size (its box head), so the rule reads it.
- Prediction (measured basis): SKU-110K products are ~40 px, so most band
  cells are excluded -> the level goes quiet -> SKU returns to baseline
  (anchor -0.013 -> ~0). VisDrone small GT are ~8 px and the band already
  holds 88% of them -> unchanged. VOC objects are large -> unchanged
  (already inert). Cost: none (a comparison on tensors stage 1 already
  computes). Scores: 5 / 4 / 4 / 5 / 5 = 23.

### R-B — Footprint exclusion (axis R1-formula)
- Exclude from the band every cell inside a decided cell's predicted box.
  Targets the "next to decided objects" half of the account only; on SKU the
  undecided products themselves would still admit live stride-4 anchors at
  40 px. Scores: 4 / 3 / 3 / 5 / 5 = 20.

### R-C — Frozen selection during training (axis R3-coupling)
- Select from a slowly-updated (EMA) posterior so targets stop flipping
  between levels. Addresses the training-instability half; does not stop
  the level firing on 40-px products. Cost: an EMA buffer. Scores:
  4 / 3 / 3 / 4 / 5 = 19. Bench.

### R-D — Density-adaptive budget (axis R1-formula)
- q shrinks with the decided-cell fraction. Measured: decided CELLS are
  1.7% on SKU (objects occupy few cells even when 52% of products are
  decided), so the signal is too weak to move q. Scores: 3 / 3 / 2 / 5 / 5
  = 18.

## Decision: exp012 = R-A (size-aware band), axis R1-formula.
Control = the anchor (exp011 legs on all three datasets). Order: SKU-110K
(first seed = screen), VisDrone vs anchor, VOC. If R-A fails, R-C is next
(a different axis); a second R1-formula failure would close that axis (P5).
