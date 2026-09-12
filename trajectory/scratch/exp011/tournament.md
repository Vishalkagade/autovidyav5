# exp011 proposal tournament — 2026-09-12 (user order: continue until a cross-domain Winner)

Ledger entering this round (exp010 seeds 123/7 still training; this record
is written before its verdict so the next chain can start without a gap):
- exp008/exp009/exp010 (sparse SPRT cascade, three variants of the decision
  rule) all raise ranking metrics and small-class AP50 but stay under the
  floor on mAP50 and lose per-unit F1. exp010 seed 42: +0.0040 mAP50, dup
  0.102 (exp001's ungated stride-4 level: 0.098; baseline 0.085).
- MEASURED on exp010/exp008 seed 42 (`exp010/state/coverage_seed42.json`):
  the band selects cells holding 85% / 88% of small GT centres at q = 0.25
  (random: 25%). The selection is informative; the thin second stage is not
  a good enough decider (small recall@.25 -0.015 vs exp001's +0.009).
- exp001's P2 level IS a good decider (+0.015, Winner) but regresses on
  normal-object data (ins_013) where a stride-4 level has nothing to add.

## Candidates (fixed criteria: fm fit / P8 / effect size / cost / distance from saturated pairs)

### C24 (new) — Band-gated stride-4 decision level
- exp001's topology; the stride-4 level may decide only in the SPRT band
  (undecided stride-8 cells, top-q). Control = random gating, same
  everything. fm: small_gt_missed + dense. P8: nearest = QueryDet (sparse
  high-res head where a trained query head fires); difference = the gate is
  the first stage's own posterior band, no query head/loss, one shared
  assignment; vs exp008-010 the decider is the P2 level itself. Effect: the
  topology gives +0.015 ungated and the band keeps 85% of small GTs —
  Winner-track vs baseline is likely; the scientific content is D3 (band vs
  random) and the cross-dataset prediction (mostly-masked level on VOC).
  Cost 3x (exp001's). Distance: same family as exp008-010 (4th consecutive),
  site = the 4-level head (the pair (sequential-evidence, 3-level
  head.detect) will be saturated if exp010 fails).
- Scores: 5 / 3 / 5 / 2 / 2 = 17.

### C20 — Successive interference cancellation at the head — 4 / 4 / 2 / 4 / 5 = 19 (unchanged)
- Stride-8 evidence residual; ins_011 caps the effect prior at 2.

### C4 predictive-coding recurrence at P3 — 15. C23 CFAR threshold — 14.

## Decision: exp011 = C24.
C20 wins on paper by two points, both from cost and saturation distance.
The tournament overrules with explicit reasoning: (1) the user's standing
order is a cross-domain Winner, and C24 is the only candidate whose effect
size is backed by two measured facts on this trajectory (the P2 level's
+0.015 and the band's 85% coverage) rather than an argument; (2) family
diversity is a heuristic and its purpose — not spending a trajectory on one
idea — is served by the bound below; (3) C24's cross-dataset prediction
(the band masks the stride-4 level where stage 1 decides) is exactly the
confirmation-track question the user set. Bound: if exp011 fails, the
sequential-evidence family is closed on this trajectory regardless of site
and the next round is C20 by rule. If exp010's gate opens, exp011 waits for
exp010's controls (chain order), then runs.
