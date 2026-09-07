# Adapter protocol — YOLO26n + VisDrone-DET, from scratch

Adapter-specific context for the agent. The harness section is at
[../../program.md](../../program.md) — read that first.

## The testbed

- **Model**: YOLO26n built from `configs/yolo26n-visdrone.yaml` (ultralytics
  yolo26.yaml with **nc: 10**) — **random init, from scratch, no pretrained
  weights anywhere**. End-to-end NMS-free Detect (`model.end2end == True`),
  reg_max=1 (no DFL), C2PSA attention block at the backbone tail.
  **2.5077M params** (verified 2026-08-21 by building from the rendered
  yaml; the nc=80 build was 2.572M — the nc=10 head is ~64k lighter).
- **Data**: full VisDrone-DET train split (6,471 images) for training;
  **FROZEN eval set = val + test-dev (2,158 images)** for evaluation (user
  decision 2026-08-21). Tier-3 locked. 10 classes: pedestrian, people,
  bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.
- **Primary metric**: mAP50 on the frozen eval set. Secondary: mAP50-95,
  precision, recall.
- **Per-unit metric (P10)**: per-image detection F1 @ IoU 0.5, conf 0.25 —
  2,158 paired units per seed. This is what Winner classification runs on.
  VisDrone averages ~50 GT boxes/image (vs COCO's ~7), so per-image F1 is
  a much denser signal per unit than on COCO-2k — degeneracy (units pinned
  at exactly 0 or 1) should be far below v5's 41-44%; the instrument check
  verifies rather than assumes this.

## Environment (inherited from v5 — re-verify at exp000)

- **ultralytics 8.4.23**, installed EDITABLE from `../ultralytics_src`
  @ commit `b10fa7be2`. Because it is editable, framework code can change
  without this repo's git noticing → every experiment JSON records
  `provenance.ultralytics_commit` (`git -C ../ultralytics_src rev-parse HEAD`),
  and `ultralytics_src` must not be touched during a trajectory. Verify
  `git -C ../ultralytics_src status` is clean before exp000.
- **Optimizer is pinned to MuSGD** in `train_and_eval` (Tier-2 lock).
  Ultralytics' default `optimizer='auto'` silently selects MuSGD when total
  iterations > 10,000 and AdamW otherwise; at batch 32 on 6,471 images
  (~203 iterations/epoch) the flip sits near **50 epochs** — squarely
  inside the plausible budget range, so 'auto' would silently train
  different stages with different optimizers. Never un-pin it.
- **Per-scale AP is not available**: `DetMetrics` in 8.4.x has no
  small/medium/large AP split and pycocotools is not installed in the
  offline env. `per_scale_metric` instead reports per-scale detection
  RECALL (IoU 0.5, conf 0.25, COCO area buckets in native pixels), computed
  with the same greedy matcher as the P10 unit metric. Do not call it
  mAP-by-scale in any record. Note the regime: VisDrone frames are
  ~1400-2000 px wide and most GT is in the COCO-"small" bucket at native
  resolution — the per-scale diagnostic will be dominated by the small
  bucket by construction; per-class and per-unit views carry the
  discriminating structure.

## Why VisDrone, why from scratch (do not revisit)

The pretrained-patch regime was closed empirically (bounded mechanisms on a
converged optimum are capped). The COCO-2k from-scratch trajectory then
closed as a structured null with evidence that YOLO26+COCO is a co-evolved
local optimum (`ins_v5_p3_fusion_locally_optimal`: both directions of
P3-fusion modification hurt). VisDrone is the falsification arena for that
reading: tiny-object aerial imagery is maximally far from COCO's object
statistics (median GT area, boxes/image, class balance), so architecture-
dataset equilibrium arguments do not transfer. Mechanisms that were nulls on
COCO — above all the P2-evidence-fusion family — are RE-TESTED here by
design. Absolute mAP will be far below pretrained VisDrone SOTA — expected
and irrelevant; all comparisons are within-regime.

**No pre-named weaknesses (anti-anchoring rule).** This document
deliberately names NO expected failure modes, and none may be added. What
the model is bad at on VisDrone is an empirical OUTPUT of exp000's
diagnostics (P0), never an input to them — the COCO-2k failure modes do not
carry over as `fm_*` ids. "Tiny objects are hard" is a dataset-selection
rationale (above), not a measured failure mode: even the obvious must show
up in exp000's per-class / per-scale / per-unit data before any proposal
may cite it. (The sites table below describes what each pathway COMPUTES —
architecture facts needed to bind a measured `fm_*` to a site — not claims
about what is weak.)

## Tier-3 locks (this trajectory)

| Parameter | Value |
|---|---|
| Model yaml | `configs/yolo26n-visdrone.yaml` (nc=10, scale n via filename) |
| Dataset (headline) | `visdrone.yaml` (train: 6,471 list; val: FROZEN eval list, 2,158) |
| Dataset (train-time val) | `visdrone_s1.yaml` (same train list, val: `visdrone_val500.txt` — P12 curves only, never headline numbers) |
| Image size | 640 |
| Batch | 32 |
| Working seeds | 42, 123, 7 (all screening, Stage-2, and control runs) |
| Confirmation seeds | 1000, 2000 — **SEALED for P11.** Only exp000's baseline legs and a P11 replication may ever use them. |
| S1 epochs (probes only) | **CALIBRATE via exp000** (placeholder 30) |
| S2 epochs | **CALIBRATE via exp000** (placeholder 100) |
| AMP | True (user decision 2026-09-07 — local RTX 4090 is the compute; flipped before any run existed) |
| Eval checkpoint | **`last.pt`** (final epoch — pre-registered 2026-09-07, user decision). NEVER `best.pt`: best-checkpoint selection scores on `visdrone_val500.txt`, which overlaps the frozen eval (23%), quietly coupling model choice to the eval set. Epochs are Tier-3 fixed and training is deterministic, so `last.pt` is well-defined and selection-free. Applies to headline metrics, per-unit metrics, and diagnostics alike. |

Budgets are placeholders until exp000 measures: (a) min/epoch from scratch
at this train size, (b) the epoch at which vanilla eval-mAP50 slope
flattens (Stage-2 should sit just past it; S1 at roughly 1/3 of S2), (c)
noise floors per stage from the 3 working-seed vanilla runs. Write the
measured numbers to `trajectory/profiles/calibration.json` (schema:
`core/schemas/calibration.schema.json`) — the adapter reads that file at
runtime; you do NOT edit `adapter.py` (guard rail). Record the same numbers
in exp000's JSON, THEN start Phase-1.

**Image size 640 — rationale + standing caveat.** 640 keeps per-run cost
comparable to v5 and matches the common VisDrone YOLO baseline setting.
It also means most GT falls below one stride-8 cell at input resolution —
which is the regime under test, not a flaw. If exp000's instrument check
fails at 640, image size is the first suspect; changing it is a Tier-3
regeneration (user decision + rerun exp000), not a tweak.

## P11 in this adapter

`CONTROL_REPLICATION_NOISE_MULTIPLE = 2.0` (in
`core/discipline/p11_replication.py`): if a Provisional Winner's P9 margin
(mechanism − control, seed-averaged) is below 2× the S2 noise floor, the
control must also be re-run on the confirmation seeds before the combined
P10 tests. Check `control_must_replicate()` BEFORE budgeting the P11 run.

## exp000 — mandatory calibration + diagnosis run

1. Vanilla from-scratch, Stage-2 budget probe: 1 seed, generous epoch cap,
   record the eval-mAP50 curve → pick s2_epochs at slope-flattening.
2. Vanilla at S1 and S2 budgets × working seeds {42, 123, 7} → noise floors
   (max-min spread per stage) + per-unit metric distributions. Write
   `trajectory/profiles/calibration.json`. Note v5's lesson
   (`ins_v5_seed_noise_wider_than_calibrated`): a 3-seed max-min floor can
   come from a lucky narrow cluster — record per-seed values verbatim so
   later evidence can widen the picture without touching the write-once
   calibration file.
3. Vanilla Stage-2 × confirmation seeds {1000, 2000} — the sealed-seed
   baseline legs. Doing them now means a later P11 replication only has to
   train the mechanism (and possibly its control), not the baseline.
4. Diagnostics on the best vanilla seed: per-class AP, per-scale recall,
   top-20 failure images → write `trajectory/profiles/baseline.json`
   candidate failure modes (`fm_*` ids with evidence strings).
5. **INSTRUMENT CHECK (pre-registered — decides whether this regime is
   viable at 640).** Compute from the exp000 data, record the verdict in
   exp000's JSON and `baseline.json`:
   - **Degeneracy:** fraction of the 2,158 per-unit F1 values that are
     exactly 0 or exactly 1, per seed. **Fail if > 70%.** (A mostly-zero
     unit metric starves P10 of paired information.)
   - **Relative noise floor:** S2 noise floor ÷ baseline seed-avg S2
     mAP50. **Fail if > 0.15.** (A floor that large relative to the
     signal makes the P9 margins unreachable for realistic effects.)
   These thresholds are fixed NOW, before any data exists — do not adjust
   them after seeing the numbers. **On fail:** the 640 regime is declared
   instrument-limited; the first revision candidate is image size (user
   decision, Tier-3 regeneration, rerun exp000). Phase-1 NEVER starts on a
   failed instrument check.

Timing: v5 measured 0.32 min/epoch at 2k images (A100, batch 32, AMP off).
Data-proportional estimate at 6,471 images: **~1.0-1.2 min/epoch** — an
ESTIMATE until exp000 measures it (VisDrone's dense labels make val and
loss assembly costlier per image). At the 100-epoch placeholder that is
~2 GPU-h per S2 run; exp000 (probe + 5 baseline legs + diagnostics) lands
around **~12-15 GPU-h**. It is the foundation of every later gate — do not
economize here; DO get its job chain right the first time (fairshare queue
waits are 1-2 days).

## Architectural sites (see sites.py)

Indices verified on the nc=80 build 2026-07-29 (v5); layer graph is
unchanged by nc=10 — only `head.detect`'s cls-branch params shrink.
Re-verify param counts at exp000 via `describe_model`.

| Site | idx | Params (nc=80 build) | Depth | Notes |
|---|---|---|---|---|
| `backbone.c3k2_p2` | 2 | 6.6k | early | high-res, low-level features; small-object pathway |
| `backbone.c3k2_p3` | 4 | 26k | early | small-object pathway |
| `backbone.c3k2_p4` | 6 | 87k | mid | semantic stage |
| `backbone.c3k2_p5` | 8 | 346k | mid | semantic stage |
| `backbone.sppf` | 9 | 165k | mid | multi-scale pooling |
| `backbone.c2psa` | 10 | 250k | mid | attention block — YOLO26's newest component |
| `head.c3k2_fuse_p4` | 13 | 120k | late | FPN fusion |
| `head.c3k2_out_p3` | 16 | 34k | late | small-object output path |
| `head.c3k2_out_p4` | 19 | 95k | late | medium-object output path |
| `head.c3k2_out_p5` | 22 | 463k | late | large-object output path |
| `head.detect` | 23 | ~280k (nc=10) | late | NMS-free end-to-end head |

(Non-site layers: 0/1/3/5/7 backbone Convs, 11/14 Upsample, 12/15/18/21
Concat, 17/20 head downsample Convs.) P9 context: the 10% param budget is
~251k params — comparable to an entire SPPF or C2PSA block, so structural
mechanisms have real room.

Sites are where the default `site_wrap` injection is wired. `topology` and
`aux_objective` mechanism kinds are not limited to this list.

## Mechanism vocabulary hints (detection-flavored, non-binding)

Bind mechanism class → site class:
- Confidence/decision-theoretic mechanisms → `head.detect` (the NMS-free head
  makes train-time detection theory unusually direct here)
- Scale-aware / multi-resolution mechanisms → FPN sites, SPPF, topology yamls
  (the P2-evidence-fusion family lives here)
- Recurrent / iterative-refinement mechanisms → C2PSA or head C3k2 blocks
- Learning-dynamics mechanisms (objectives, schedules from source-domain
  theory) → **driver-side loss only** (see hazards — the `aux_objective`
  kind is metadata, not plumbing)
- Connectivity priors → `topology` kind (modified yaml)

Source domains: as the harness section — outside CV/ML, principle named with
a citable reference. The COCO-2k insight pool (inherited entries in
`trajectory/insights.md`) lists what was falsified THERE; re-proposing one
here requires the regime argument (P8), and doing so for the best-motivated
candidates is the trajectory's founding purpose.

## Controls (P9) in this adapter

- `param_matched_generic`: `GenericResidualConv` at the same site, width
  auto-tuned to the mechanism's added params (±10%).
- `frozen_random`: the mechanism with its own parameters frozen at init.
- A capacity-matched control (widened site block via a per-mechanism topology
  yaml) is designed at Winner-promotion time; it is not a harness enum value.
- Topology mechanisms (e.g. a P2-fusion head) need a topology-kind control —
  a param-matched generic connectivity change — designed per-experiment and
  pre-registered before the screen.

Control runs reuse the baseline's seeds; each control needs the same stages
as the mechanism it matches. Cost model: Winner certification ≈ 3x a bare
mechanism run (≈2x when the P9 deferral gate stays shut).

## Known operational hazards (inherited from v5 — all of these BIT once)

- **SurgeryTrainer pattern is MANDATORY for site_wrap training**
  (`Model.train` re-builds the model from yaml and silently drops in-place
  surgery — v5 exp001 trained VANILLA for a full stage before this was
  caught). Every site_wrap driver overrides `DetectionTrainer.get_model` to
  re-apply the wrap and asserts the wrap class is present at the site index
  before AND after trainer setup. No bare `model.train()` on a surgered
  object, ever.
- **`aux_objective` kind is dead plumbing** (`ins_v5_aux_objective_dead`):
  nothing reads `model.autovidya_aux`. Objective mechanisms implement the
  loss driver-side (E2ELoss(loss_fn=...) pattern, one2many branch) with an
  `applied_batches` tripwire proving the term ran. Verify before proposing.
- Ultralytics DDP re-builds the model from yaml in subprocesses, discarding
  in-place modifications → **single-GPU only** (adapter enforces device=0).
- The framework is an **editable install** (`../ultralytics_src`): a stray
  edit there changes every experiment, past comparisons included. Record
  `provenance.ultralytics_commit` in every JSON; verify clean before exp000.
- **nc mismatch rebuild**: the trainer builds its model from the yaml +
  data nc. The Tier-3 model yaml already carries nc: 10, so no silent
  override — but this is exactly the rebuild path that drops surgery (see
  SurgeryTrainer above).
- BN defaults: any BN inside new modules must use Ultralytics' eps=1e-3,
  momentum=0.03, or from-scratch dynamics drift.
- `deepcopy` of a YOLO object detaches its trainer state — always inject
  into a freshly built model, never a trained one.
- Full-eval cost: 2,158 large images ≈ minutes per pass, so training NEVER
  validates per-epoch on the frozen eval. The adapter trains with
  `visdrone_s1.yaml` (per-epoch val on the locked 500-image subset — feeds
  the P12 curve) and runs the frozen eval exactly once at stage end.
  Headline metrics come from that final full pass only.
- **Per-image predict for per_unit_metric** — a list source becomes ONE
  batch in ultralytics 8.4.x (OOM at 2,158 images). The adapter already
  iterates image-by-image; keep it that way in drivers too.
- **Quota:** home ~95/100G soft. `runs_visdrone/` weights are the main
  growth; prune `best.pt` checkpoints freely (they are never evaluated —
  see the `last.pt` Tier-3 lock) and keep `last.pt` until the experiment
  JSON is committed. Watch for silent output truncation when near quota.
