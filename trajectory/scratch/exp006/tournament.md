# exp006 proposal tournament — 2026-09-09 (after exp005 Reject)

Ledger: exp001 Winner (stride-4 P2 head: +0.015, 1.5x fwd, 3x wall-clock);
exp003 Reject (duplicates irrelevant); exp004 Kill (emphasis zero-sum);
exp005 Reject (decision slots without evidence: 0.26x of exp001's small-
class gain, per-unit F1 negative). The decomposition has one leg left:
stride-4 EVIDENCE at stride-8 decision density. If that leg carries most of
exp001's gain, the mechanism is the evidence path, and it can be had at
~1.1x compute instead of 3x wall-clock. Family-diversity: exp005 was
multi-hypothesis-decision, so evidence-fusion is admissible.

Criteria (fixed): (1) fm fit (2) P8 (3) effect vs 0.005 floor (4) cost (5) ledger distance.

## Candidates

### C18 (new) — Polyphase evidence route (multirate signal processing)
- Source: polyphase decomposition in multirate filter banks (Vaidyanathan
  1993; Crochiere & Rabiner 1983): a signal at rate 1/4 is represented at
  rate 1/8 as four polyphase components with NO information loss, instead of
  being decimated. The backbone's stride-2 convs decimate; the head never
  sees the stride-4 feature. Mechanism: backbone P2 (stride 4, 64 ch) ->
  space-to-depth into its 4 polyphase components (256 ch at stride 8) ->
  1x1 conv to 32 ch -> concatenated into the P3 head fusion. Decision grid
  unchanged (stride 8). fm_small_gt_missed (+ crowded neighbours).
- P8: nearest CV blocks = SPD-Conv (Sunkara & Luo 2022) / YOLOv5 Focus —
  both are space-to-depth in the STEM replacing a strided conv. Difference:
  a lateral route from the P2 feature into the P3 fusion, alongside (not
  instead of) the existing path; purpose is to preserve P2 detail for the
  P3 decision. Honest note: the block itself is ultralytics' Focus; the
  novelty is where the evidence goes and the decomposition it completes.
- Effect story: exp001 gain minus exp005 gain ~= evidence share. Prediction
  registered before data: >= 0.5x of exp001's small-class AP50 gain.
- Cost: +10,304 params (0.4%), forward 1.1x (5.2 vs 4.7 ms / batch 8), no
  extra anchors (assigner unaffected) -> ~1.1x wall-clock. Control: same 32
  extra channels into the same concat computed from the STRIDE-8 P3 feature
  by two 1x1 convs (+11,184 params, 1.085x).
- Scores: 5 / 3 / 4 / 5 / 5 = 22.

### C3 ECOC — 19; C15 opponent coding — 18; C5 relaxation labeling — 16; C4 predictive-coding — 15.

## Decision: exp006 = C18, family `evidence-fusion`, kind topology (plain Model.train; strides + Focus-presence tripwires).
