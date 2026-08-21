# SEALED — inherited scientific results (COCO-2k / bounded regime)

**Seal rule (P0 anti-anchoring, user-authorized 2026-08-21): this file may
NOT be read until `trajectory/profiles/baseline.json` exists with committed
`analysis.candidate_failure_modes`.** VisDrone's failure modes are an
empirical output of exp000's diagnostics; a session that has read the
COCO-2k failure-mode narrative before diagnosing is primed to find the same
structure. Sessions running Phase-0 stop at this header. Once baseline.json
is committed, this file is required reading for Phase-1 proposal work — P8
re-test arguments cite these entries by id.

This file is frozen inherited record: append nothing, edit nothing.
All entries: `status: inherited` — advisory only, exempt from the exp-id
evidence rule (evidence lives in `../autovidya_v5` and earlier repos), never
citable as evidence for any classification, never counted by P2 or P3.
COCO-2k falsifications are NOT death sentences here — re-testing them on
VisDrone is the founding hypothesis; what a re-test needs is the regime
argument (P8).

## From the COCO-2k from-scratch trajectory (v5, closed 2026-08 as structured null)

- `ins_v5_p3_fusion_locally_optimal` (v5 exp004+exp007) — On COCO-2k the P3
  fusion's evidence mix was locally optimal against BOTH directions of
  modification: suppressing the coarse stream hurt (-0.0045); supplying an
  equal-width P2 fine stream hurt far worse (-0.0222, the largest harm of
  any v5 mechanism, with topology tripwires proving the pathway was real).
  THIS is the trajectory hypothesis: if P2-evidence fusion also fails on
  VisDrone, the mechanism is dead; if it works, the co-evolved-optimum
  reading of COCO gains direct support.
- `ins_v5_capacity_scaling_falsified` (v5 exp003+exp005) — Scaling the
  cooperative-cascade family 6x (19.7k → 118k params) ERASED its small
  positive margin instead of amplifying it. "Thin wrappers are merely
  capacity-starved" is not a sufficient account of nulls; more capacity can
  make a mechanism worse.
- `ins_v5_cls_gap_epiphenomenon` (v5 exp006) — An ECOC auxiliary loss
  verifiably closed ~10% of the cls generalization gap with an exact-null
  mAP effect (COCO-2k): there, the gap was a symptom, not a cause. If
  VisDrone's exp000 surfaced a cls-gap failure mode, cite this before
  targeting it.
- `ins_v5_tail_not_reachable_from_train` (v5 exp008) — Explicitly optimizing
  the per-image loss tail (CVaR_0.25, verifiably applied) made the val F1=0
  mass WORSE on COCO-2k. Whole-image failure mass there was a property of
  what the train set cannot teach, not of optimizer pressure.
- `ins_v5_scale_not_frequency` (v5 exp000) — COCO-2k baseline failure was
  structured by object scale and clutter, NOT class frequency (per-class AP
  uncorrelated with train instance count). Check whether VisDrone's exp000
  reproduced this structure — after the fact, never as a prior.

## From the bounded-gate regime (U-Net/Kvasir, pre-v5)

Falsified in the BOUNDED-GATE regime; re-proposing one here requires an
explicit argument for why the from-scratch regime revives it (on top of the
P8 regime argument for the dataset shift).

- `ins_bounded_gates_capped` — Bounded element-wise amplitude gates (Rayleigh
  CDF and 4 generic controls): no per-unit-significant effect in either
  direction; whole class capped in the bounded regime.
- `ins_bounded_spatial_coupling_fragile` — Spatial coupling in gate
  mechanisms: multi-seed fragility (Huber gate).
- `ins_bounded_asym_boundary_loss` — Asymmetric boundary-only auxiliary
  losses: one-sided error inflation.
