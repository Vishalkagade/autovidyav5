# Insights — YOLO26n / VisDrone from-scratch trajectory

## Inherited from prior trajectories — operational + methodology ONLY

All entries in this section: `status: inherited` — advisory only, exempt
from the exp-id evidence rule (their evidence lives in `../autovidya_v5` and
earlier repos), never citable as evidence for any classification, never
counted by P2 or P3. These are facts about the harness, the framework, and
the statistics — they inherit with full force and carry no claims about
what VisDrone's baseline is bad at.

**The inherited SCIENTIFIC results (COCO-2k mechanism outcomes, failure-mode
structure, bounded-regime falsifications) live in
[`inherited_v5_scientific.md`](inherited_v5_scientific.md) and are SEALED
until `profiles/baseline.json` has committed candidate failure modes (P0
anti-anchoring — see that file's header and program.md). Do not open that
file during Phase-0.**

### Methodology (statistics + screening)

- `ins_aggregate_metrics_lie` (pre-v5) — Batch-weighted aggregation once
  produced a false +0.005 headline. Hence P10.
- `ins_v5_s1_screen_not_predictive` (v5 P12, after exp004) — A short
  screening stage did not predict full-budget outcomes (3 promotions, 2 into
  negative S2). Consequence baked into THIS trajectory's design: the screen
  is the first full-budget S2 seed, not a shorter stage (program.md
  "Screening", scripts/pipeline.sh).
- `ins_v5_training_deterministic_within_seed` (v5 exp004) — Identical pinned
  args reproduced mAP50 bit-for-bit and all per-unit values exactly. Paired
  comparisons have zero measurement error; the "noise floor" is entirely
  BETWEEN-seed variation. Verify once on VisDrone (cheap: rerun one short
  probe) before leaning on it.
- `ins_v5_seed_noise_wider_than_calibrated` (v5, 7 baseline seeds) — A
  3-seed max-min noise floor came from a lucky narrow cluster; 7 seeds
  showed ~1.6x that spread (SD 0.0032 vs floor 0.0063 max-min). Read exp000's
  floor as a lower bound; judge Winner margins against per-seed SD, and
  record per-seed baseline values verbatim in the experiment JSONs.
- `ins_v5_regression_to_mean_at_floor` (v5 exp002/exp003) — Sub-floor
  effects move seeds back toward the seed-average; never read a sub-floor
  delta as real without per-unit paired evidence.

### Operational (harness + framework hazards — see also adapter program.md)

- `ins_v5_surgery_dropped_by_train` (v5 exp001) — `Model.train` rebuilds
  from yaml and silently drops in-place site_wrap surgery; a full stage
  trained vanilla before this was caught. SurgeryTrainer pattern MANDATORY.
- `ins_v5_aux_objective_dead` (v5 exp002) — The `aux_objective` mechanism
  kind is metadata with no plumbing; objective mechanisms are implemented
  driver-side with an applied-batches tripwire.
- `ins_v5_zero_params_not_zero_cost` (v5 exp004) — A parameter-free
  mechanism ran 2.3-3.1x baseline wall-clock (large-activation ops). Budget
  mechanisms by estimated compute, not param count.
- `ins_v5_list_predict_ooms` (v5 exp000) — ultralytics 8.4.x loads a list
  source as ONE batch; per-unit metrics must predict image-by-image.

## Findings (populate as trajectory runs)
