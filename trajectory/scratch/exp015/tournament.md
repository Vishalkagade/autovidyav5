# exp015 — Phase-2 refinement of exp011, axis R3-coupling, second and last attempt (2026-09-16)

Per-level breakdown of the five SKU-110K seed-42 checkpoints (300 eval images,
conf .25, IoU .5; trajectory/scratch/phase2/level_breakdown_sku_seed42.json):

| model    | TP    | FP   | dup  | recall | F1     |
| baseline | 37451 | 5211 |  824 | 0.8352 | 0.8420 |
| ungated  | 37476 | 5055 | 1010 | 0.8358 | 0.8438 |
| anchor   | 37053 | 5000 | 1238 | 0.8263 | 0.8380 |
| absband  | 37327 | 5055 | 1407 | 0.8324 | 0.8368 |
| ownband  | 37345 | 5317 | 1262 | 0.8329 | 0.8364 |

exp014 restored recall (ownership fixed) but F1 stayed 0.005 down: the
gated models carry 400-600 extra DUPLICATES per 300 images. A stride-4
anchor that is masked in training gets no signal at all (its outputs are
replaced by constants), so it is never trained as a negative for objects
another anchor owns; whenever its cell is in the band at inference it
fires beside the stride-8 detection. The ungated model trains every
stride-4 anchor as a negative when it loses the assignment, hence its low
duplicate count.

### R-N — Negative-trained masked anchors (R3-coupling) — CHOSEN
- Training: the stride-4 level keeps its real outputs; the live mask is
  applied inside the assigner (masked anchors can never be positives), so
  masked anchors are trained as negatives. Inference: masked exactly as the
  anchor. Selection rule and ownership: the anchor's. Cost: none.
- Prediction: SKU duplicates return to the ungated level's rate and the
  0.005 loss closes (screen PASS); VisDrone duplicates (anchor 0.100 vs
  baseline 0.085) fall too, mAP50 unchanged or up.
- Scores: 5 / 3 / 5 / 5 / 4 = 22.

### R-C — Frozen (EMA) selection — 18. Bench. If R-N fails, R3-coupling is
closed (P5) and the remaining axes are R1-position, R1-paramcount, R2-aux,
R4-cross-domain.
