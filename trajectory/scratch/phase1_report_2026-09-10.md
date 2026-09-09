# Phase-1 report — YOLO26n + VisDrone from scratch (2026-09-10)

**End state: Winner delivered.** exp001 (P2 evidence fusion) passed P9, P10 on both comparisons, and P11 sealed-seed replication. Six further mechanisms, five of them cross-domain, closed as nulls and together explain *why* exp001 works. Phase-1 spend: 34.8 of 60 GPU-h (exp000 calibration 7.5 GPU-h separate).

## Calibration (exp000)
Budgets s2 = 80, s1 = 30 epochs (pre-registered 97%-of-plateau rule). Baseline mAP50 0.2704 (seeds 42/123/7: 0.2697/0.2733/0.2683; sealed 1000/2000: 0.2693/0.2689). S2 noise floor 0.0050 (relative 0.018). Instrument check PASS (per-unit degeneracy 1.5-1.7%). Measured failure modes: fm_small_gt_missed (68% of GT is COCO-small, recall 0.23), fm_dense_scene_f1_collapse (F1 0.55 -> 0.28 with box count), fm_sibling_class_confusion (van->car 58% of located vans).

## Scoreboard
| Exp | Mechanism | Family | Verdict | mAP50 Δ (seed-avg) | per-unit F1 95% CI | GPU-h |
|---|---|---|---|---|---|---|
| exp001 | P2 evidence fusion (stride-4 feature routed into the head, | evidence-fusion | **Winner** | +0.0163 | [+0.0077,+0.0122] | 23.7 |
| exp002 | Divisive normalization on one-to-one class logits | lateral-inhibition | **Op-Fail** | -0.2244 | — | 0.9 |
| exp003 | Divisive normalization on one-to-one class logits | lateral-inhibition | **Reject** | -0.0000 | [-0.0031,+0.0010] | 2.2 |
| exp004 | Horvitz-Thompson detection-probability loss reweighting | estimator-weighting | **Kill** | -0.0114 | — | 0.7 |
| exp005 | Multi-hypothesis decision at P3 (K=4 sub-cell hypotheses,  | multi-hypothesis-decision | **Reject** | +0.0030 | [-0.0054,-0.0013] | 4.4 |
| exp006 | Polyphase evidence route: stride-4 P2 feature carried as 4 | evidence-fusion | **Reject** | +0.0002 | [-0.0018,+0.0022] | 2.2 |
| exp007 | Sibling-pair margin objective (ECOC one-bit margin on meas | code-margin | **Kill** | -0.0072 | — | 0.7 |

## What the ledger says (insights ins_003..ins_012)
1. **The P2 head's gain is real and replicates** (ins_006): +0.015 mAP50, five seeds all positive, control at baseline. The same computation was COCO-2k's largest harm (-0.022). This is the first direct support for the co-evolved-optimum reading: YOLO26's stride set is tuned to COCO's object sizes.
2. **It is the conjunction of evidence and grid** (ins_011): stride-4 evidence at stride-8 density (exp006) gives 0.08x of the gain; stride-4 slots with stride-8 evidence (exp005) give 0.26x with negative per-unit F1. No cheap half exists; 3x wall-clock is the price.
3. **Emphasis is zero-sum** (ins_009): weighting the loss toward small objects moves detections between scales (small recall +0.026, medium/large -0.06) and nets -0.011.
4. **Duplicates are not the dense-scene bottleneck** (ins_008): a lateral-inhibition field removed duplicates on every seed with a null metric effect.
5. **Sibling confusion is a resolution problem** (ins_012): a verified logit margin never reached its margin and changed nothing.
6. **Instrument lessons** (ins_004, 005, 007): recall@threshold discriminators miss AP-visible gains; params mis-price topology mechanisms by 10x; source-domain constraints must be enforced by parameterisation.

## Harness audit
- A/A check (P12 FPR): one control passed through the full chain (exp001's stride-8 control): per-unit p = 0.68, no false positive.
- Screen predictiveness: 4 multi-seed completions — audit not yet due (every 10).
- Operational regret: exp002 (0.9 GPU-h, NaN from an unconstrained parameter), two wall-clock losses from script bugs (logged in cwm). P4 retries consumed: 0.
- P0 seal breach by the diagnosing session is on record (index.json cwm).

## Claim discipline
exp001 is **a VisDrone Winner**. It becomes a model-improvement claim only after the cross-dataset track: SKU-110K (win required) and Pascal VOC (no-regression bar). That track starts next.
