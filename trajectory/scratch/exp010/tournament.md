# exp010 proposal tournament — 2026-09-11 (user order: continue until a cross-domain Winner)

Ledger entering this round: exp008 (SPRT cascade, Reject) raised mAP50 on
every seed (+0.0046 avg, 0.92x floor), mAP50-95 on every seed and
small-class AP50 by +0.014 (0.61x the P2 head) but lost per-image F1 on every
seed; measured cause = duplicates (0.112 vs 0.085). exp009 (its one P4
retry, parent masked) did NOT remove them (0.129 on seed 42) and lost the
parent's contribution (mAP50 -0.0023, medium recall -0.042): the four
sub-cells of a selected cell co-fire on the inherited parent logit.

## Candidates (fixed criteria: fm fit / P8 / effect size / cost / distance from saturated pairs)

### C22 (new) — Sparse second sample with an INDEPENDENT stride-4 decision
- Same band selection and stage-2 evidence as exp008; the stage-1 posterior
  enters stage 2 as a detached feature and the sub-cell logit is stage 2's
  own (no additive parent term); parent kept. Mirrors exp001's decision
  structure (each stride-4 anchor its own logit, one-to-one assignment
  shared) at 1/4 of the evidence cost.
- fm: small_gt_missed (+dense). P8: nearest = QueryDet; difference = band
  selection by the first stage's posterior, posterior-conditioned second
  decision, single shared assignment (see prereg). Honest note: thinner than
  exp008's accumulation argument — the P8 case now rests on the band and the
  posterior conditioning.
- Effect: exp008's AP gains were reached WITH the duplicate drag; removing
  the measured cause is a directional prediction, not a hope. Cost known
  (2.4 h/seed). Distance: third consecutive sequential-evidence experiment at
  head.detect — a failure saturates the pair (P2) and forces rotation.
- Scores: 5 / 3 / 4 / 4 / 2 = 18.

### C20 — Successive interference cancellation at the head (wireless comms) — 4 / 4 / 2 / 4 / 5 = 19 (unchanged)
- Residual is stride-8 evidence; ins_011 says that is the limiting factor,
  so criterion 3 stays at 2. Total 19 beats C22 on paper by the saturation
  criterion alone.

### C23 (new) — CFAR adaptive detection threshold (radar constant-false-alarm-rate)
- Local clutter estimate from guard/reference cells normalises the
  objectness logit. Same class-site as exp002/003 (normalisation at
  head.detect, lateral-inhibition family; ins_007/ins_008): 3 / 3 / 1 / 5 / 2
  = 14.

### C4 predictive-coding recurrence at P3 — 15 (unchanged; stride-8 evidence).

## Decision: exp010 = C22.
C20 scores one point higher, entirely from the saturation criterion. The
tournament overrules family diversity explicitly: C22 is the only candidate
whose effect-size argument is backed by measured data on this trajectory
(exp008's ranking gains plus the measured duplicate mechanism), and the
overrule is bounded — if C22 fails, P2 fires on the pair and the next round
is C20 by rule. exp010 is NOT a retry (P4 spent on exp009); it enters from
P8 with the novelty audit in prereg.json.
