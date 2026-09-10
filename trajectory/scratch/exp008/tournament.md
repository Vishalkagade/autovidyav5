# exp008 proposal tournament — 2026-09-10 (user order: continue until a cross-domain Winner)

Ledger after seven experiments: the only certified gain is the P2 head
(exp001), and the decomposition (ins_011) says it works because evidence
AND decision are both at stride 4. Every thin cross-domain mechanism
(0-10k params; decision-side or objective-side) is a null with a
mechanistic reason (ins_008, 009, 010, 012). Untested: heavy structural
mechanisms (50-250k params of the 250k cap) that produce finer evidence
and decide on it. All candidates below live there.

Criteria (fixed): (1) fm fit (2) P8 (3) effect vs 0.005 floor (4) cost (5) ledger distance.

## Candidates

### C19 — SPRT cascade: sequential evidence accumulation to stride 4 (statistics, Wald 1945)
- Source: the sequential probability ratio test — decide when the
  accumulated log-likelihood ratio leaves an (A, B) band, otherwise take
  another sample. Applied per stride-8 cell: the P3 head produces a
  first-stage objectness LLR; cells inside the learned undecided band
  request a second sample — a stride-4 evidence patch (the P2 feature
  around the cell, 4x4 window) processed by a small conv stack that emits
  four sub-cell decisions (the exp001 conjunction, but only where the SPRT
  says the evidence is insufficient). Decided cells keep their stride-8
  prediction. Budget: top-q fraction of cells by band membership (q = 0.25
  pre-registered), so stride-4 compute is ~1/4 of the P2 head's.
- fm: small_gt_missed + dense (missed neighbours).
- P8: nearest CV blocks = QueryDet (Yang et al. 2022: sparse high-res
  queries for small objects) and cascade detectors (Viola-Jones). Difference:
  selection by a learned two-sided LLR band on the FIRST-stage score
  (accumulate-then-decide), not by a separate 'small-object' query head;
  the second stage ADDS its LLR to the first (evidence accumulation) rather
  than replacing it. Honest note: QueryDet is close; the P8 case rests on
  the accumulation rule and the band, and the control is designed to test
  exactly that.
- Control (designed): same sparse stride-4 stage on a RANDOM selection of
  the same fraction of cells (placebo gating). If SPRT selection does not
  beat random selection, the decision rule is not the mechanism.
- Effect story: recovers the conjunction where it matters. Prediction: a
  Winner-track effect (>= 2x floor) at ~1.3x wall-clock. Params ~60-120k.
- Scores: 5 / 3 / 4 / 4 / 5 = 21.

### C20 — Successive interference cancellation at the head (multi-user detection, wireless comms)
- Source: SIC (Verdu 1998): decode the strongest user, reconstruct and
  subtract its contribution, decode the next. Per stride-8 cell: after the
  first one-to-one decision, a learned 'footprint' of the detected object
  is subtracted from the P3 feature and a second decision pass runs on the
  residual, recovering neighbours the first detection masked. fm: dense
  (missed neighbours, ins_008). P8: nearest = iterative/recurrent detection
  and DETR query decoding; difference = explicit reconstruct-and-subtract
  in feature space. Effect prior: exp005 showed extra slots alone fail;
  SIC changes the evidence for the second slot, so it is not the same test
  — but the residual is still stride-8 evidence, which ins_011 says is the
  limiting factor. Cost ~1.3x. Scores: 4 / 4 / 2 / 4 / 5 = 19.

### C21 — Cortical magnification: learned retinotopic resampling of the input (neuroscience) — REJECTED at P8
- Nearest CV block IS the mechanism (learning-to-zoom, saliency-guided
  resampling). Not admitted.

### C4 predictive-coding recurrence at P3 — 15 (unchanged; stride-8 evidence).

## Decision: exp008 = C19 (SPRT cascade), family `sequential-evidence`, kind site_wrap at head.detect + head.c3k2_out_p3 (SurgeryTrainer), q = 0.25.
