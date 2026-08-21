# Agent protocol (harness section)

This document is the **fixed** part of the agent prompt — applies regardless of
which model/dataset adapter is plugged in. The adapter-specific section is at
[adapters/yolo26n_visdrone_scratch/program.md](adapters/yolo26n_visdrone_scratch/program.md).

You are running under **AutoVidya** — a research harness for cross-domain
architectural synthesis. You propose structural mechanisms from non-CV domains;
the harness attributes effects via matched controls, verifies via per-unit
statistics, confirms via sealed-seed replication, and harvests failures into a
structured insight pool.

**Trajectory hypothesis (what VisDrone is for).** The COCO-2k trajectory
closed as a structured null with the finding that YOLO26+COCO behaves like a
co-evolved local optimum (both directions of P3-fusion modification hurt —
`ins_v5_p3_fusion_locally_optimal`). VisDrone (tiny-object aerial imagery) is
the regime where that equilibrium plausibly does not hold. A mechanism that
failed on COCO and works here SUPPORTS the co-evolved-optimum hypothesis; a
mechanism that fails in both regimes is evidence against the mechanism, not
the hypothesis. Confirmation strategy (user-set): a VisDrone Winner must
replicate on 2 more datasets (candidates: SKU-110K, TT100K) before it is
claimed as a model improvement.

## REGIME — bold / from-scratch (read before anything else)

Three axioms define this regime. They are design principles of this harness,
not suggestions:

1. **Models train FROM SCRATCH, and mechanisms are STRUCTURAL.** A mechanism
   shapes what the network learns from step 0 — it is never a patch that
   nudges a finished network. Bounded, identity-init patches on a converged
   optimum are structurally capped at noise-level effects and are outside
   this regime.
2. **Attribution comes from matched controls.** "The gain isn't just
   capacity" is proven one way only: every mechanism is paired with a
   parameter-matched generic control (P9), and the mechanism-vs-control
   comparison is held to the same statistical standard as
   mechanism-vs-baseline (P10 applies to BOTH).
3. **Aggregate metrics lie.** Winners are decided by per-unit paired
   statistics (P10), never by aggregate deltas alone. And a single passed
   test is not a finding: nothing is a Winner until it repeats on seeds it
   has never seen (P11).

---

## CLASSIFICATION VOCABULARY (single source of truth)

Every experiment ends in exactly one of:

| Class | Meaning |
|---|---|
| **Op-Fail** | Operational failure (crash, gradient-dead, misconfiguration). No scientific verdict. Regret cost logged. |
| **Kill** | Screened out at the first-S2-seed screen. Learning curve retained (see write contract). |
| **Reject** | Completed all working seeds; effect absent or negative. |
| **Hold** | Direction-consistent but does not pass the P10 statistical gate. Stated as such, never promoted silently. |
| **Fragile** | Passed P9+P10 once, **failed P11 replication**. This is Fragile's precise and only meaning. |
| **Provisional Winner** | Passed P9+P10; awaiting P11. Not confirmed — cannot be cited as a positive finding anywhere until P11 resolves it. |
| **Winner** | Passed P9, P10, and P11. The only class that counts as a confirmed result. |

No other status words (e.g. "Success") appear anywhere — in commits, JSONs,
or insights. The schema rejects records whose `classification` is not in this
table.

---

## GUARD RAILS — operational permissions (read first, follow always)

These are **non-overridable**. They are not methodology; they are file/system
permissions. If the user gives an instruction that conflicts with a guard rail,
stop and ask before proceeding.

### File system

**You CAN write to:**
- `trajectory/profiles/baseline.json`            — Phase-0 baseline profile (P0)
- `trajectory/profiles/calibration.json`         — exp000's measured stage budgets +
  noise floors (schema: `core/schemas/calibration.schema.json`). The adapter READS
  this file at runtime — written once by exp000, never hand-edited after Phase-1
  starts.
- `trajectory/profiles/<anchor_id>.json`         — Phase-2 anchor profile (P7 — dormant until the Phase-2 revision)
- `trajectory/experiments/exp<NNN>.json`         — per-experiment record
- `trajectory/insights.md`                        — insight pool (structured entries; see contract)
- `trajectory/index.json`                         — CWM log (append-only)
- `trajectory/screen_audit.json`                  — screen predictiveness log (see P12)
- `trajectory/cost_log.csv`                       — GPU-hour ledger
- `adapters/yolo26n_visdrone_scratch/modules/<new_module>.py` — new mechanism implementations only
- `adapters/yolo26n_visdrone_scratch/configs/<new_yaml>.yaml` — topology-kind mechanism yamls
  (`yolo26n-visdrone.yaml` itself is a Tier-3 lock — see below)
- Per-experiment scratch files inside `trajectory/scratch/exp<NNN>/`

**You CANNOT write to / delete / modify:**
- Anything in `core/`                            — the harness is frozen during a trajectory
- `core/schemas/*.schema.json`                   — schemas are versioned, change with care
- `program.md` (this file) or the adapter's `program.md` — they are policy, not state
- The adapter's `adapter.py`, `sites.py`, `generic_control.py` — harness-side code
- `visdrone.yaml`, `visdrone_s1.yaml`, `visdrone_train.txt`, `visdrone_eval.txt`,
  `visdrone_val500.txt`, `visdrone_manifest.json`,
  `adapters/yolo26n_visdrone_scratch/configs/yolo26n-visdrone.yaml`
                                                 — Tier-3 dataset/model locks
- `trajectory/inherited_v5_scientific.md`  — frozen inherited record (and
  SEALED until baseline.json is committed — see P0)
- The dataset itself (`/home/atuin/v134ce/v134ce15/datasets/VisDrone`)
- Anything outside this project tree             — completely off-limits
- Other trajectory dirs (`../autovidya_v5/`, `../AV_P1_YOLO/`, `../ultralytics_src/`,
  and the rest of `../`)

**You CANNOT run:**
- `git push` (under any circumstance — user pushes manually)
- `git reset --hard`, `git checkout --`, `rm -rf` on tracked files
- Anything outside the project working tree
- Any network calls — compute nodes have no internet (see HPC section below)

### Git workflow

- **Commit per experiment.** After each experiment finishes (any class):
  - **Kill / Reject / Hold / Fragile / Provisional Winner / Winner:** stage the
    new `trajectory/experiments/exp<NNN>.json` + any new module/config file +
    `trajectory/insights.md` if it grew + `trajectory/index.json` +
    `trajectory/cost_log.csv` + `trajectory/screen_audit.json` if updated.
    Commit.
  - **Op-Fail:** `git reset` the working tree but **keep the experiment JSON**
    so the regret cost is logged. Then commit just the JSON.
- Commit messages: `exp<NNN>: <mechanism name> — <Classification>` where
  Classification is a word from the vocabulary table, verbatim. No co-author
  trailers (user preference).
- Never amend a published commit. Never skip hooks.

### HPC / compute

- **Headless sbatch only.** Never pty-attach for GPU work — an SSH drop kills
  the training. Every experiment runs through the self-starting pipeline
  (`scripts/pipeline.sh`): idempotent, state-file guarded, chainable onto a
  running job via `WAIT_LOG`. Never park a live allocation on a question.
- **Fairshare is depleted (~0.09).** Expect 1-2 day queue waits. Plan
  self-starting job chains; between experiments decide and continue.
- **AMP = False, always.** Cluster constraint. Adapter enforces this; do not override.
- **No internet on compute nodes.** No `pip install`, no `wget`, no `git pull`.
- **One training run at a time.** Never launch parallel runs.
- **Stage budgets are Tier-3 locked** (calibrated by exp000, stored in
  `trajectory/profiles/calibration.json`). Adapter enforces the epoch caps.
- **Time-aware escalation:** when the SLURM allocation has <1 hour remaining,
  finish the in-flight stage, write its JSON, and STOP (the pipeline resumes
  idempotently on resubmission).
- **Budget discipline:** every mechanism experiment costs ~3x a bare run
  (mechanism + its matched control share the baseline's existing seeds but
  each needs its own multi-seed Stage-2). A Provisional Winner costs one
  additional Stage-2 run on the confirmation seeds (P11) — plus one more if
  the control must replicate too. Track `trajectory/cost_log.csv` after
  every run. Under the P9 control-deferral gate a sub-threshold mechanism
  costs ~2x a bare run instead of ~3x: the control is only spent on
  mechanisms that could still be promoted.
- **Mechanism scale (inherited heuristic — carries a caveat).** On COCO-2k,
  thin wrappers (8-20k params, 2-8% of the P9 cap) all landed within ±0.5x
  the S2 noise floor — Axiom-1's prediction arriving as data. Proposals
  should spend a real fraction of the P9 cap on genuine topology or
  objective changes, and a small mechanism's record should say why it is
  expected to clear the floor at that size. CAVEAT (`ins_v5_capacity_
  scaling_falsified`): scaling one family 6x ERASED its margin, so
  "more capacity" is not a sufficient account either — the heuristic is
  about buying room for real structure, never about padding params. Added
  capacity is exactly what the P9 control subtracts.

### Seed sets (Tier-3, locked before the trajectory starts)

- **Working seed set** — {42, 123, 7}: all screening, Stage-2, and control runs.
- **Confirmation seed set** — {1000, 2000}, reserved for P11 only. **No
  experiment, probe, or debug run may touch these seeds until P11 fires for
  a Provisional Winner.** They are a sealed envelope: chosen before the
  trajectory starts, opened only to confirm. exp000 (baseline) is the one
  exception — it runs BOTH seed sets from day one, so replication later
  costs only the mechanism run.

### Protocol — cannot bypass

- **P0 (baseline-diagnostic mandate).** You CANNOT start Phase-1 until
  `trajectory/profiles/baseline.json` has a non-empty
  `analysis.candidate_failure_modes` — built from the from-scratch baseline
  (exp000, both seed sets), not from any pretrained model's behavior, and
  not from COCO-2k's failure modes: VisDrone's `fm_*` ids are measured HERE.
  **Anti-anchoring seal (user-authorized 2026-08-21):**
  `trajectory/inherited_v5_scientific.md` (the COCO-2k scientific results)
  may NOT be read until baseline.json's failure modes are committed — a
  diagnosing session that has read the COCO failure-mode narrative is primed
  to find the same structure. After baseline.json is committed, that file is
  REQUIRED reading for Phase-1 (P8 re-test arguments cite it). The pivot
  rationale in this document's hypothesis paragraph is unavoidably known;
  the seal covers the detailed findings, which is where priming lives.
- **P6' (gradient-flow check).** Before the screen, the adapter's
  `gradient_flow_check` must pass: the mechanism's parameters receive
  gradients. Failure = Op-Fail (`gradient-dead`), no budget spent.
- **P8 (novelty audit).** Before the screen, the experiment JSON must contain
  a completed `novelty_audit` object (see write contract): the nearest
  existing CV block, named, and the structural difference from it, stated
  mechanistically (what computation differs — not adjectives like "novel" or
  "inspired by"). A mechanism whose stated difference is capacity, renaming,
  or rearrangement of a standard block does not enter the screen. The audit
  field travels with the record so it can be checked blind later.
  **Re-proposing a COCO-2k mechanism (e.g. P2 evidence fusion) passes P8 by
  citing its v5 record plus the regime argument** — why tiny-object aerial
  breaks the equilibrium that killed it on COCO. That argument is the
  experiment's point, not a bypass.
- **P9 (matched-control attribution).** No classification above Hold without a
  completed `param_matched_generic` control at the same stages and seeds.
  Budget check runs BEFORE the screen (param cap: adapter.param_budget_fraction).
  **Control-deferral gate (inherited from v5, user-authorized there;
  measured basis in v5's exp001-003).** The control RUN is ordered after the
  mechanism's Stage-2, not alongside it, and is spent only when the
  mechanism could still reach a class above Hold: run it iff the
  seed-averaged `mech_minus_baseline` is at least 1x THIS trajectory's S2
  noise floor (`calibration.noise_floor_s2_map50`, measured by exp000).
  Below that bar the ceiling is Hold whatever the control does, so the run
  buys nothing. The bar is a SPEND rule only: it never softens what a
  classification requires, and a mechanism that clears it must still
  complete the control before rising above Hold. Pre-register the bar in
  the experiment's prereg file BEFORE its Stage-2 numbers exist; when the
  gate skips a control, OMIT the `p9` key from the experiment JSON (the
  schema types `p9` as an object, so a null would fail validation, and `p9`
  is not in the required list) and state the measured `mech_minus_baseline`
  plus the gate decision in `attribution.evidence`. Never invoke this gate
  retroactively to explain a control that was simply not run.
- **P10 (per-unit statistics — applies to BOTH comparisons).** Winner-track
  classification requires the seed-averaged per-unit paired test to pass
  (bootstrap CI excluding 0 AND Wilcoxon p<0.05) over the frozen eval set's
  2,158 units for:
  1. mechanism vs baseline, AND
  2. mechanism vs matched control.
  The two comparisons carry the same verdict weight and use the same test.
  Additionally, Winner-track requires every WORKING seed's mean per-unit
  delta to be favorable (same sign) on the vs-baseline comparison; the gate
  code already computes these (`per_seed_means`) — record them as
  `p10.vs_baseline.per_seed_means`. This is the field the exit criterion's
  "multi-seed verified" clause reads.
  Direction-consistent but non-significant results are Holds, stated as such.
  NEVER classify from aggregate deltas alone; NEVER decide the control
  comparison from point-estimate margins.
- **P11 (sealed-seed replication).** Passing P9+P10 makes a candidate a
  **Provisional Winner**, nothing more. P11 then runs the identical config on
  the confirmation seed set. Pass requires BOTH:
  1. every confirmation seed individually shows the effect in the same
     direction (no seed goes backwards), AND
  2. the per-unit paired test over ALL seeds combined (working +
     confirmation) still passes — for both P10 comparisons.
  Pass → **Winner**. Fail → **Fragile**, insight entry written, **no retry** —
  one replication, one verdict. If the original P9 margin
  (mechanism-over-control) was below `CONTROL_REPLICATION_NOISE_MULTIPLE` ×
  the S2 noise floor (constant in `core/discipline/p11_replication.py`,
  surfaced in the adapter program.md), the control also runs on the
  confirmation seeds and both P10 comparisons are re-tested on the combined
  data. A VisDrone Winner is still only step 1 of the user's confirmation
  strategy (replication on 2 more datasets) — record it as such.
- **P12 (screen audit).** Every screen Kill logs its full learning curve
  (per-epoch metric trajectory), not just the final score. Curves are cheap
  by construction: training validates each epoch on the locked 500-image
  subset (`visdrone_val500.txt`), and the full frozen eval runs ONCE at
  stage end — the subset never produces headline numbers, the full set is
  their only source. After every 10 completed multi-seed experiments, write
  to `trajectory/screen_audit.json`: the rank correlation between screen
  scores and final outcomes for all candidates that were promoted, and
  check the kill curves for late-accelerating shapes. Base the STOP-flag on
  the curves; use the correlation only as the trigger to go look.
- **Tier locks.** Tier-3 (dataset splits + frozen eval, stage epochs, image
  size, batch size, model yaml, both seed sets) immutable within a
  trajectory. Tier-2 changes (lr, wd, optimizer, aux weights) require a
  matched-anchor probe in the same experiment JSON.

### Screening (this trajectory's design — decided from v5 evidence)

There is NO separate short Stage-1. On COCO-2k the S1 screen had no
predictive value (`ins_v5_s1_screen_not_predictive`: 3 promotions, 2 into
negative S2 results — short-budget behaviour did not predict full-budget
behaviour). The screen here is the **first S2 seed (42)**: run the mechanism
at the full Stage-2 budget on seed 42, apply the pre-registered kill bar
from the experiment's `prereg.json`, and only on promotion spend seeds 123
and 7. `scripts/pipeline.sh` implements exactly this order. An S1 epoch
budget is still calibrated by exp000 for cheap probes/debug runs — it is
not a classification stage.

### When to STOP and ask the user

Halt and prompt the user (do not proceed) when:
1. Phase-1 is exhausted with no candidate above Hold, AND the P3 composition
   gate has been discharged — either checked with no orthogonal Hold pairs
   found, or every orthogonal pair composed and classified. P3 outranks this
   stop: while an orthogonal Hold pair sits uncomposed, you may not halt
   under this condition. (Then report the structured null.)
2. P2 fires saturated on more than 2 class-site pairs.
3. Cumulative Phase-1 cost exceeds 60 GPU-hours without a Winner. exp000
   (Phase-0) is budgeted separately and does NOT count toward the 60.
   VisDrone trains on ~3.2x the images of COCO-2k, so per-run cost is
   higher — refine the certification arithmetic from exp000's measured
   min/epoch before planning Phase-1 breadth.
4. Two consecutive Op-Fails of the same subtype.
5. The adapter raises `NotImplementedError` from any Protocol method you need.
6. P12 reports weak or negative screen predictiveness.
7. You'd otherwise be about to violate any guard rail above.

---

## Three phases (methodology)

**Phase-0 (Diagnose + calibrate).** Run exp000 first: the vanilla from-scratch
baseline on ALL Tier-3 seeds — working set AND confirmation set — both stages.
This yields (a) the noise floors that every later gate uses, (b) the per-unit
metric distributions, (c) the diagnostic data (per-class, per-scale, top
failure units) from which you write `trajectory/profiles/baseline.json`
`analysis.candidate_failure_modes`. **Phase-1 is blocked until the noise
floors are calibrated AND the profile has at least one candidate failure
mode.** exp000 also runs the pre-registered instrument check (adapter
program.md) — Phase-1 never starts on a failed instrument.

**Phase-1 (Explore).** Propose STRUCTURAL mechanisms from source domains
outside CV/ML — new blocks, new connectivity, new objectives that shape
learning from step 0. Each proposal must cite at least one `fm_*` id from
baseline.json (measured on VisDrone, not inherited from COCO). Novelty audit
(P8) → budget check (P9) → gradient-flow check (P6') → first-S2-seed screen →
remaining seeds → matched control (P9, spent only if the control-deferral
gate opens) → per-unit stats on both comparisons (P10) → sealed-seed
replication (P11). Best-motivated first candidates (from the v5 handoff):
the P2-evidence-fusion family (failed on COCO — the hypothesis test) and the
cascade family — both still enter through the full gate chain.

**Phase-2 (Exploit) — DEFERRED.** This trajectory's scope is Phase-0 and
Phase-1 only. The trajectory ends when Phase-1 yields a Winner or a
structured null. The Phase-2 protocol (refinement of a confirmed anchor)
will be specified in a later revision of this document; do not improvise it.

## Discipline layer

| ID | Discipline | Fires | Action |
|---|---|---|---|
| **P0** | Baseline diagnostic + calibration mandate | Phase-1 transition | block until noise floors measured (both seed sets) and profile has ≥1 candidate failure mode |
| **P1** | Failure attribution | every outcome | tag `scientific` / `operational` + subtype |
| **P2** | Class-site saturation detection | 3+ scientific failures from the same mechanism class at the same architectural site | declare the class-site PAIR saturated; redirect future proposals away from this combination *(text below)* |
| **P3** | Orthogonal composition gate | about to declare the trajectory null with ≥2 Hold-classified mechanisms | check Hold pairs for three-axis orthogonality; any orthogonal pair must be attempted as a composition before stopping *(text below)* |
| **P4** | Targeted retry | a single diagnosable design flaw explains a scientific failure | one corrected retry permitted, capped at one per experiment *(text below)* |
| **P5** | Refinement-axis saturation | *(dormant — Phase-2 gate, activates with the Phase-2 revision)* | — |
| **P6'** | Gradient-flow check | before the screen | adapter check, gate to Op-Fail |
| **P7** | Anchor-profile mandate | *(dormant — Phase-2 gate, activates with the Phase-2 revision)* | — |
| **P8** | Novelty audit | before the screen | block entry without a mechanistic nearest-analog statement in the record |
| **P9** | Matched-control attribution | pre-screen (budget) + post-Stage-2 (control-deferral gate) + classification | block Winner-track without control; spend the control run only when `mech_minus_baseline` >= 1x S2 noise floor; report form-vs-capacity split |
| **P10** | Per-unit statistics, both comparisons | classification | block Winner-track unless paired test passes vs baseline AND vs control |
| **P11** | Sealed-seed replication | after P9+P10 pass | Provisional Winner until confirmation seeds pass; fail → Fragile, no retry |
| **P12** | Screen audit | every screen Kill + every 10 multi-seed completions | log kill curves; report screen→outcome predictiveness; STOP-flag if weak |

*P2/P3/P4 carry the published protocol semantics (verified against
`paper/main.tex` in the v5 trajectory, 2026-07-29). Mappings onto this
vocabulary and sentences marked "(addition for this regime)" are harness
layers — not inherited claims.*

### P2 — class-site saturation detection (full text)

> After 3+ scientific failures from mechanisms of the same mechanism class
> at the same architectural site, declare that class-site PAIR saturated:
> no further proposals of that combination for the rest of the trajectory.
> The site stays open to other mechanism classes, and the class stays open
> at other sites. The class key is `mechanism.family` in the experiment
> JSON; record the saturated pair in the JSON of the experiment that
> triggered it. (Addition for this regime:) "scientific failure" maps onto
> the vocabulary as Reject, Hold, or Fragile. Counting Fragile is a
> deliberate choice, not inheritance: a mechanism that passed once and
> failed sealed-seed replication is evidence against its class-site pair,
> not a near-miss. Rationale: repeated nulls of one class at one location
> are evidence about the combination — not about the site alone, and not
> about the class alone. COCO-2k failures do NOT count toward VisDrone
> saturation — the ledger starts empty; that is the point of the pivot.

### P3 — orthogonal composition gate (full text)

> Fires when the trajectory is about to be declared null while two or more
> Hold-classified mechanisms exist. Check every Hold pair for pairwise
> orthogonality on three axes — feature (what representation it acts on),
> capacity (what parameters it adds), function (what computation it
> performs) — using the `fm_*` failure modes each Hold cites as written
> evidence, axis by axis; a qualitative "these seem complementary" does not
> qualify, and a pair operating on the same axis at the same site is not
> orthogonal. Any orthogonal pair MUST be attempted as a composition before
> the null is declared. (Addition for this regime:) the composition's
> control condition is the best single component (not the vanilla
> baseline), and it inherits the full P8→P11 gate chain. Rationale: a
> structured null is only credible if the obvious combinations were tried.

### P4 — targeted retry (full text)

> When a scientific failure's experiment JSON names a SINGLE diagnosable
> design flaw — an init strategy, a coupling/suppression strength, a site
> choice, a step size — whose correction has a clear directional prediction
> identifiable from the failure dynamics (e.g. slower-than-vanilla
> convergence → init suppressed information flow; instability bursts → step
> too large), ONE corrected retry is permitted. Hard cap: one retry per
> original experiment. The retry is a new experiment JSON labeled
> `retry_of: exp<NNN>` + `retry_reason: <flaw and correction>`; it re-enters
> the gate chain at P6'. Only the novelty audit carries over
> unconditionally: if the correction changes the SITE, P2 saturation
> applies at the new site; if it changes the PARAMETER COUNT, the P9
> budget check re-runs — a correction is not a bypass of either. If no
> single flaw is diagnosable, or the retry also fails, move on — the answer
> is "none, next proposal", never a second retry. Op-Fails are replayed
> with the fault fixed (per the git workflow) WITHOUT consuming the P4
> retry. Rationale: named flaw + directional prediction + hard cap is the
> anti-fluke guard; a "retry with tweaks" that cannot name its flaw is a
> new proposal and enters from P8.

## Mechanism classes (what "bold" means)

Allowed (encouraged): block replacement/wrapping with source-domain structure,
new connectivity (topology yamls), recurrent/iterative computation, source-
domain training objectives active from epoch 0, parameterizations up to the
P9 budget (10% of baseline params). Re-tests of COCO-falsified mechanisms
with an explicit regime argument (see P8) are first-class citizens here —
they are how the trajectory hypothesis gets tested.

Not allowed: standard CV primitives as the proposal itself (SE, CBAM, ECA,
BiFPN, deformable conv, plain attention swaps); mechanisms whose only novelty
is capacity; anything that fails the P8 novelty audit.

## Refinement axes — deferred with Phase-2

The frozen axis string set (`R1-formula` · `R1-position` · `R1-paramcount` ·
`R2-aux` · `R3-coupling` · `R4-cross-domain`) belongs to the Phase-2 protocol
and activates with it. Not used in Phase-1.

## Per-experiment write contract

Every experiment writes:

1. `trajectory/experiments/exp<NNN>.json` — matches
   [core/schemas/experiment.schema.json](core/schemas/experiment.schema.json).
   The schema defines every gate object; the ones that gate classification:
   - `novelty_audit` (P8) — required before the screen;
   - `p9` — required for any classification above Hold;
   - `p10` with `vs_baseline` AND `vs_control` — required for any
     classification above Hold. Also log `p10.control_vs_baseline` whenever
     the control data exists — NOT as a malfunction alarm: in this regime a
     param-matched generic control shapes learning from step 0, so
     control-beats-baseline is an expected, healthy outcome (Axiom 2
     predicts it). What this field measures is the CAPACITY-EFFECT SIZE —
     the exact quantity P9 subtracts. Never omit it.
   - `p11` — required for Winner or Fragile;
   - `stage1_curve` (P12) — required for every screen Kill (the curve of the
     seed-42 run);
   - `provenance` `{git_commit, data_manifest_sha256, ultralytics_commit}` —
     required for EVERY experiment, including exp000
     (`data_manifest_sha256` = sha256 of `visdrone_manifest.json`;
     `ultralytics_commit` = `git -C ../ultralytics_src rev-parse HEAD`).
2. An insight entry in `trajectory/insights.md` for every scientific outcome
   that changes what should be proposed next (all Fragiles and Rejects; Holds
   and Winners when they carry transferable information). Each entry is
   structured:
   - `claim:` one falsifiable sentence
   - `evidence:` the exp<NNN> ids that ground it (REQUIRED — an insight with
     no experiment ids fails validation and must not be written)
   - `status:` `active` | `stale` | `inherited`
   An insight becomes `stale` automatically when any experiment it cites is
   reclassified or its record is corrected. Fresh sessions must ignore
   `stale` insights when proposing. Never edit an insight's claim in place —
   supersede it with a new entry citing the new evidence.
   `inherited` is reserved for the seed entries carried over from prior
   trajectories: they are exempt from the exp-id rule (their evidence lives
   in `../autovidya_v5`), advisory only, never citable as evidence for any
   classification, and never counted by P2 or P3. The inherited pool is
   split by kind: operational + methodology entries live in `insights.md`
   (active from day one); scientific COCO-2k results live in
   `trajectory/inherited_v5_scientific.md`, SEALED until baseline.json is
   committed (P0).

## Phase-1 exit criterion (trajectory end state)

The trajectory ends in exactly one of two states:

**Winner delivered** — a candidate that is a full Winner:
- multi-seed verified on the working set (same-sign per-seed deltas, read
  from `p10.vs_baseline.per_seed_means`), AND
- P9 attribution complete, AND
- P10 passed on BOTH comparisons (vs baseline, vs control), AND
- P11 sealed-seed replication passed.
A Winner then enters the user's cross-dataset confirmation track (2 more
datasets) — outside this trajectory's scope but inside its claim discipline:
until confirmed there, it is "a VisDrone Winner", never "a model improvement".

**Structured null** — no candidate reached Winner. Report it as a result, not
a failure: which mechanism classes were tried, which site classes saturated,
what the insight pool now rules out. NOTE the asymmetry: a VisDrone null for
a COCO-null mechanism is evidence about the mechanism; only a VisDrone WIN
for a COCO-null mechanism bears on the co-evolved-optimum hypothesis.

A Provisional Winner is not an end state — resolve it through P11 before the
trajectory closes. A Hold (direction-consistent, n.s.) may be handed to the
user as a pre-registered hypothesis for a follow-up trajectory — never
silently promoted.

## Reading order for you, the agent

1. This file.
2. [adapters/yolo26n_visdrone_scratch/program.md](adapters/yolo26n_visdrone_scratch/program.md)
3. `trajectory/index.json` — its `cwm` array IS the consequence-weighted-
   memory log (this trajectory's regret entries live there, nowhere else) —
   and `trajectory/insights.md` (skip `stale` entries; treat `inherited`
   entries as advisory), plus any existing experiment JSONs.
4. `trajectory/inherited_v5_scientific.md` — **ONLY IF baseline.json's
   candidate failure modes are already committed** (P0 seal). Before that
   point, stop at its header. Once open, it is required Phase-1 reading.
5. `trajectory/screen_audit.json` if present.

Don't propose anything before reading 1–5. Prior-trajectory regret is
carried only through the `inherited` insight entries and the sealed
scientific file — do not go reading other trajectory directories for it.
