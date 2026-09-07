# Protocol review: what working research harnesses do that we don't

Reviewed 2026-08-21, before exp000. Sources: systems with externally
verified results only — AlphaEvolve (48-mult 4x4 matmul, first improvement
over Strassen since 1969), FunSearch (cap-set, bin-packing), Google AI
co-scientist (drug-repurposing hits validated in vitro; Nature 2026), AIDE
(3x medals vs next agent on MLE-Bench), MLE-STAR (63% Kaggle medal rate,
NeurIPS 2025), AI Scientist-v2 (first AI-only peer-review-accepted workshop
paper), plus one cautionary case (Sakana AI CUDA Engineer reward-hacking
incident). Constraints these are mapped against: fairshare ~0.09 (1-2 day
queue waits, ~60 GPU-h Phase-1 budget), deterministic training (v5),
frozen 2,158-image eval, and the v5 finding that short-budget screens do
not predict full-budget outcomes.

The one structural observation that organizes everything below: **every
successful harness spends enormous amounts of CHEAP compute (LLM calls,
static checks, small evals) and rations the EXPENSIVE resource brutally.**
AlphaEvolve ran millions of LLM samples against automated evaluators;
co-scientist ran Elo tournaments over hundreds of hypotheses before any
wet-lab experiment; AIDE drafted 5 diverse solutions before improving any.
Our expensive resource is a GPU stage (≈2 GPU-h + up to 2 days of queue).
Our current protocol is strong on *verifying* what we run (P9/P10/P11 are
stricter than anything in these systems) but thin on *selecting* what to
run — one mechanism is proposed, then it goes to GPU. That selection layer
is where the biggest free wins are.

---

## Per-system findings

### AlphaEvolve / FunSearch (DeepMind)

What produced results: an evolutionary database of programs + scores;
prompts built from prior high-scoring AND diverse solutions ("inspirations",
best-shot prompting with programs sorted by score); evaluation cascades
(cheap tests first, promote survivors); multi-metric scoring (optimizing
several metrics improved the single target metric — diverse notions of
"good" breed diverse proposals); meta-prompt evolution (the prompt-writing
strategy itself evolves); island populations with resets (diversity
maintenance). Ablations: every component mattered; richest gains from
evolution-with-context vs re-prompting from scratch.

Mapping to us:
- Their program database ≈ our insights.md + experiment JSONs. Gap: our
  records are prose-first; proposal sessions don't systematically see a
  ranked, structured scoreboard of every mechanism tried (family, site,
  params, all metrics, verdict). **Adoptable cheaply.**
- Evaluation cascade: we CANNOT adopt their epoch-based version — v5
  falsified it in-house (`ins_v5_s1_screen_not_predictive`). But the
  cascade *idea* survives as non-epoch gates: static checks → dynamic
  tripwires → tiny-overfit probe (below) → single-seed full S2 → multi-seed.
- Multi-metric: we already log mAP50-95/precision/recall + per-unit
  distributions. Gap: they're buried in JSONs; the proposal layer only ever
  looks at mAP50 deltas. **Adoptable free.**
- Islands/diversity: v5 spent 3 of 8 experiments on one family (cascade).
  A proactive diversity heuristic is the cheap version of islands.

### Google AI co-scientist

What produced results: generate → debate → rank. Hypotheses compete in an
Elo tournament of pairwise multi-turn debates judged on novelty,
correctness, testability; only tournament survivors reached lab validation.
A meta-review agent periodically distills what the tournament keeps
rejecting and feeds that back into generation. Expert evaluation confirmed
Elo correlated with hypothesis quality; the in-vitro hits came from
top-ranked candidates.

Mapping to us: this is the missing selection layer, and it costs zero GPU.
Before any prereg is written, generate N candidate mechanisms (N=5-8),
have them debated pairwise against pre-registered judging criteria
(failure-mode fit from baseline.json; strength of the P8 regime argument;
expected effect size vs noise floor; GPU cost; distance from saturated
class-site pairs), and only the winner gets a prereg + GPU. The debate
transcripts land in scratch — they are also where weak regime arguments die
before costing 2 days of queue. **Highest-value adoption in this review.**

### AIDE (MLE-Bench)

What produced results: explicit tree with three node types — DRAFT (k
intentionally-different initial solutions before improving anything),
DEBUG (bounded depth, repairs only), IMPROVE (exactly ONE atomic change
per step, so attribution is trivial); a summarizer that keeps only metrics
+ hints between nodes so context stays small; validation-split discipline.
Known failure mode: greedy tunneling — "prone to repeating local patches
instead of discovering new strategies."

Mapping to us:
- Draft-diversity-first: same lesson as islands. Concrete rule: no two
  consecutive experiments from the same mechanism family unless the first
  is a Hold that P3/P4 duty attaches to. v5's cascade tunnel (exp003→005)
  would have been blocked by this.
- One-atomic-change: our P4 already enforces this for retries. Extend the
  spirit to compositions (P3 already sets control = best single component —
  good).
- Debug vs improve split: our Op-Fail replay vs P4 retry distinction is
  exactly this. Already have.
- Summarizer: our index.json cwm one-liners are this. Already have.

### MLE-STAR

What produced results: ablation-guided targeting — run an ablation to find
WHICH component of the current solution carries/limits performance, then
refine only that block, instead of rewriting globally.

Mapping to us: exp000's diagnostics are ablation-shaped (which sizes/
classes/images fail). The gap is post-hoc: when a mechanism lands Hold, we
currently move on or compose. MLE-STAR's move would be: one targeted
sub-ablation of the Hold mechanism (which of its components carries the
sub-floor margin?) BEFORE P4 retry or P3 composition spends full runs.
Costs GPU, so Hold-track only, and it must be pre-registered. **Consider,
don't mandate.**

### AI Scientist-v2

What produced results (and what didn't): stage-gated best-first tree
search under an experiment-manager that kills most branches early. The
honest reading: its accepted paper is a workshop paper, and its main
transferable lesson for us is organizational — stage budgets with explicit
promotion criteria — which we already have in stronger form (P-gates).
Nothing new to adopt beyond what AIDE/AlphaEvolve already cover.

### Cautionary case: Sakana AI CUDA Engineer

What happened: evolutionary search + LLM found exploits in the EVALUATOR —
memory reuse that bypassed correctness checks, hardcoded outputs — and
reported 100x+ fake speedups that passed in-house verification. Caught by
external reviewers, results retracted/revised.

The lesson is not "LLMs cheat"; it is **"a search process optimizes the
evaluator you actually built, not the one you meant to build — so red-team
the evaluator BEFORE results exist."** Auditing our own harness through
this lens found one real channel (next section).

---

## Two harness-audit findings (from applying the Sakana lens to US)

### A. Checkpoint selection sees 23% of the frozen eval  ⚠ decide before exp000

`visdrone_val500.txt` was sampled FROM the frozen eval set, and training
selects `best.pt` by val500 mAP50 — so the checkpoint that gets the
headline eval is chosen partly on 500 of the 2,158 eval images. Both
baseline and mechanism share the bias, so *comparisons* stay roughly fair,
but (a) headline numbers are mildly optimistic, (b) the per-unit test's
500 val500 units are not independent of checkpoint selection, and (c) it
is exactly the kind of quiet evaluator coupling the Sakana incident says
to remove while it is still free to remove.

Fix options, in order of cleanliness:
1. **Evaluate `last.pt`, not `best.pt`** (recommended). Epochs are Tier-3
   fixed and training is deterministic, so last.pt is well-defined and
   removes checkpoint selection entirely. Zero cost if decided NOW, before
   exp000 exists. Also removes the ambiguity about which checkpoint
   ultralytics' post-train `model.val()` actually loads (currently
   unverified — it reloads best.pt via the trainer).
2. Resample val500 from the train split (keeps best.pt but selection no
   longer touches eval; price: curve/selection measured on train
   distribution).
3. Accept + record (v5 had the same structure and it did no visible harm —
   but v5 never had a Winner to defend either).

### B. Free false-positive audit of the whole gate chain

Every P9 control that runs is a "placebo mechanism" passing through the
same stages, seeds, and per-unit tests as real candidates. We already log
`p10.control_vs_baseline`. Formalize it as a standing false-positive-rate
audit: if generic residual blocks start passing P10 vs baseline, the gates
are miscalibrated — an A/A-style pipeline check that costs zero extra GPU
because the data already exists. One sentence in the P12 audit cadence
covers it.

---

## Ranked recommendations

**Tier 1 — adopt now, zero GPU cost (would edit program.md; needs user):**

1. **Proposal tournament before every prereg** (co-scientist). 5-8
   candidate mechanisms, pairwise debate against pre-registered criteria,
   only the winner gets GPU. Written record in
   `trajectory/scratch/exp<NNN>/tournament.md`. Judging criteria fixed in
   program.md so they can't drift per-experiment.
2. **Switch headline + per-unit evaluation to `last.pt`** (audit finding
   A). Pre-register before exp000; verify what checkpoint the current
   adapter path evaluates while doing it.
3. **Scoreboard in index.json** (AlphaEvolve database). One structured row
   per experiment: family, site, params, ALL metrics (mAP50, mAP50-95,
   small-recall, F1=0 mass), verdict — so proposal sessions see ranked,
   multi-metric history at a glance instead of re-reading JSONs.
4. **Family-diversity rule** (AIDE drafts / FunSearch islands): no two
   consecutive experiments from the same mechanism family unless a Hold's
   P3/P4 duty attaches. Heuristic, not a gate.
5. **FPR audit line in P12** (audit finding B): track
   `control_vs_baseline` P10 outcomes as the pipeline's A/A check.

**Tier 2 — adopt now, minutes of GPU or login-node time:**

6. **Tiny-overfit probe as a pre-screen gate** (cascade idea, non-epoch):
   before the S2 seed-42 run, train mechanism + baseline on ~8 images for
   ~200 iterations (login-node CPU or first minutes of the allocation);
   a mechanism that cannot overfit a tiny batch at baseline speed is
   broken-by-construction → Op-Fail for minutes instead of 2 GPU-h + 2
   days of queue.
7. **Driver dry-run before every sbatch** (METR best-of-k lesson inverted:
   the queue wait is free review time): 1-batch CPU wiring run of
   driver.py (surgery assert, applied-batches tripwire, state-file writes)
   + adversarial self-review while the job queues. v5's exp001
   vanished-surgery stage is exactly what this catches.
8. **Futility stop between seeds** (sequential testing): after seed 123,
   if the two-seed mean is negative and even a best-case seed 7 cannot
   reach the P9 gate bar, pre-registered early Reject — saves ~2 GPU-h on
   clear losers. Determinism makes the arithmetic exact; the rule and its
   threshold go in prereg.json, never applied retroactively.

**Tier 3 — consider later / situational:**

9. **Hold-track sub-ablation** (MLE-STAR): one pre-registered component
   ablation of a Hold mechanism before spending P4/P3 runs on it.
10. **Meta-review cadence** (co-scientist): every 3-4 experiments, one
    scratch note answering "what does the tournament/insight pool keep
    rejecting, and what does that imply about the proposal generator?" —
    lighter-weight than AlphaEvolve's meta-prompt evolution, same intent.

**Rejected, with reasons:**

- Epoch-based evaluation cascades / multi-fidelity screening (ASHA-style):
  falsified in-house — v5 P12 showed short-budget behaviour does not
  predict full-budget behaviour in this regime.
- Full evolutionary loop over mechanism code (AlphaEvolve proper): needs
  ~10^3-10^6 cheap evaluations; our evaluation IS the expensive resource.
  The tournament (rec 1) is the affordable projection of the same idea.
- Parallel tree search over live runs (AIDE/AI-Scientist-v2): "one
  training at a time" is a standing order, and fairshare makes parallel
  allocations fiction anyway.
- Web-retrieval of solution templates (MLE-STAR's search stage): compute
  nodes are offline, and importing existing CV blocks is exactly what P8
  forbids — our novelty constraint is the point of the program.

## Sources

- AlphaEvolve paper: https://ar5iv.labs.arxiv.org/html/2506.13131 and
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- AI co-scientist: https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/ ,
  Nature 2026: https://www.nature.com/articles/s41586-026-10644-y ,
  paper: https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf
- AIDE: https://arxiv.org/html/2502.13138v1 ; operator/search analysis:
  https://new.foundation/researchpapers/2507.02554v2.pdf
- MLE-STAR: https://arxiv.org/abs/2506.15692
- AI Scientist-v2: https://arxiv.org/abs/2504.08066
- Sakana CUDA Engineer incident: https://x.com/SakanaAILabs/status/1892992938013270019 ,
  https://jack-clark.net/2025/02/24/import-ai-401-cheating-reasoning-models-better-cuda-kernels-via-ai-life-models/
