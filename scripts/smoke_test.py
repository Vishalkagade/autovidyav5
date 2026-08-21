"""Smoke test for the autovidya_visdrone scaffold.

Checks, in order:
1. Harness + gate imports (core.adapter, P9, P10, P11).
2. The YOLO26n VisDrone adapter satisfies the Adapter Protocol.
3. Sites are well-formed and consistent with SITE_INDEX.
4. The experiment schema's classification enum matches the program.md
   vocabulary table (kept in sync BY THIS TEST — change both together).
5. Metric fixtures: image_f1 on hand-computed cases. These PIN the metric's
   conventions (empty-image scoring, greedy matching). If you change the
   metric, these fixtures must change in the same commit.
6. P10 statistical gate on synthetic data: a planted effect must pass,
   pure noise must not (needs numpy; skipped with a warning if missing).
7. P11 replication logic on toy inputs.
8. Calibration + dataset-lock status report (informational — missing before
   exp000 / setup is expected, not a failure).

Run from the repo root (the outer autovidya/.venv has torch+numpy+scipy):
    ../.venv/bin/python -m scripts.smoke_test
or with uv:
    uv run --with numpy,scipy,torch python -m scripts.smoke_test

The default run needs no GPU, no dataset, and no ultralytics install
(torch is needed only because core.adapter imports it).
On a GPU node with ultralytics available, add --with-yolo to also build the
real model and check param count + a probe forward/backward.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# program.md vocabulary table, verbatim. Test 4 asserts the schema matches.
VOCABULARY = ["Op-Fail", "Kill", "Reject", "Hold", "Fragile",
              "Provisional Winner", "Winner"]

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(f"{label}: {detail}")


def approx(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-yolo", action="store_true",
                    help="Also build the real YOLO26n model (needs ultralytics).")
    args = ap.parse_args()

    print("=== autovidya_visdrone smoke test ===\n")

    # ── 1. Imports ────────────────────────────────────────────────────────
    print("[1/8] Imports")
    try:
        from core.adapter import Adapter
        from core.discipline.p9_matched_control import attribution_check, budget_check
        from core.discipline.p10_stats_gate import P10Result, paired_unit_test
        from core.discipline.p11_replication import (
            CONTROL_REPLICATION_NOISE_MULTIPLE,
            control_must_replicate,
            replication_check,
        )
        check("core + discipline gates import", True)
    except Exception as e:
        check("core + discipline gates import", False, f"{type(e).__name__}: {e}")
        return _report()

    # ── 2. Adapter Protocol ───────────────────────────────────────────────
    print("[2/8] Adapter Protocol")
    try:
        from adapters.yolo26n_visdrone_scratch.adapter import (
            CONFIRMATION_SEEDS,
            WORKING_SEEDS,
            Yolo26nVisdroneScratchAdapter,
            image_f1,
        )
        adapter = Yolo26nVisdroneScratchAdapter()
        check("adapter satisfies Protocol", isinstance(adapter, Adapter))
        check("working/confirmation seeds disjoint",
              not set(WORKING_SEEDS) & set(CONFIRMATION_SEEDS),
              f"overlap: {set(WORKING_SEEDS) & set(CONFIRMATION_SEEDS)}")
        check("confirmation set has >=2 seeds", len(CONFIRMATION_SEEDS) >= 2)
    except Exception as e:
        check("adapter constructs", False, f"{type(e).__name__}: {e}")
        return _report()

    # ── 3. Sites ──────────────────────────────────────────────────────────
    print("[3/8] Sites")
    from adapters.yolo26n_visdrone_scratch.sites import SITE_INDEX
    sites = adapter.sites()
    check("sites() non-empty", bool(sites))
    check("site depths valid",
          all(s["depth"] in ("early", "mid", "late") for s in sites))
    check("every site has an index",
          all(s["name"] in SITE_INDEX for s in sites),
          f"missing: {[s['name'] for s in sites if s['name'] not in SITE_INDEX]}")

    # ── 4. Schema vocabulary sync ─────────────────────────────────────────
    print("[4/8] Schema <-> program.md vocabulary")
    schema = json.loads(
        (PROJECT / "core/schemas/experiment.schema.json").read_text())
    enum = schema["properties"]["classification"]["enum"]
    check("classification enum == vocabulary table", enum == VOCABULARY,
          f"schema has {enum}")

    # ── 5. image_f1 fixtures (pin the metric's conventions) ──────────────
    print("[5/8] image_f1 fixtures")
    # Convention: empty image, no predictions -> perfect score.
    check("no GT + no preds -> 1.0", approx(image_f1([], []), 1.0))
    # Convention: hallucinating on an empty image -> zero.
    check("no GT + preds -> 0.0",
          approx(image_f1([(0, 0.9, 0, 0, 10, 10)], []), 0.0))
    # Convention: missing everything -> zero.
    check("GT + no preds -> 0.0", approx(image_f1([], [[0, 0, 0, 10, 10]]), 0.0))
    # Exact match -> perfect.
    check("perfect match -> 1.0",
          approx(image_f1([(0, 0.9, 0, 0, 10, 10)], [[0, 0, 0, 10, 10]]), 1.0))
    # Class must match even at IoU 1.0.
    check("class mismatch -> 0.0",
          approx(image_f1([(1, 0.9, 0, 0, 10, 10)], [[0, 0, 0, 10, 10]]), 0.0))
    # 2 GT, 1 matched pred: precision 1, recall 0.5 -> F1 = 2/3.
    check("partial recall -> 2/3",
          approx(image_f1([(0, 0.9, 0, 0, 10, 10)],
                          [[0, 0, 0, 10, 10], [0, 50, 50, 60, 60]]), 2 / 3))
    # GREEDY PIN — this is the case where greedy scores WORSE than the
    # optimal assignment, on purpose:
    #   GT A = (0,0,10,10), GT B = (6,0,16,10)
    #   pred1 (conf .9) = (3,0,13,10): IoU with A = IoU with B = 7/13 ≈ .538.
    #     Tie -> strict '>' keeps the FIRST GT in file order -> takes A.
    #   pred2 (conf .8) = (0,0,10,10): A already taken; IoU with B = .25 -> unmatched.
    #   tp=1, prec=rec=0.5 -> F1 = 0.5. (Optimal pairing would give 1.0.)
    greedy_case = image_f1(
        [(0, 0.9, 3, 0, 13, 10), (0, 0.8, 0, 0, 10, 10)],
        [[0, 0, 0, 10, 10], [0, 6, 0, 16, 10]],
    )
    check("greedy matching + first-GT tie-break -> 0.5",
          approx(greedy_case, 0.5), f"got {greedy_case}")
    # Below the IoU threshold -> no match.
    check("IoU below 0.5 -> 0.0",
          approx(image_f1([(0, 0.9, 0, 0, 10, 10)], [[0, 7, 0, 17, 10]]), 0.0))
    # COCO area buckets used by per_scale_metric (boundaries are inclusive
    # on the upper side: 32^2 is medium, 96^2 is large).
    from adapters.yolo26n_visdrone_scratch.adapter import _area_bucket
    check("area buckets follow COCO convention",
          _area_bucket(32 * 32 - 1) == "small"
          and _area_bucket(32 * 32) == "medium"
          and _area_bucket(96 * 96 - 1) == "medium"
          and _area_bucket(96 * 96) == "large")

    # ── 6. P10 gate on synthetic data ────────────────────────────────────
    print("[6/8] P10 gate sanity (synthetic)")
    try:
        import numpy as np
    except ImportError:
        print("  [WARN] numpy not available — run via "
              "`uv run --with numpy,scipy python -m scripts.smoke_test` "
              "to exercise this check.")
        np = None
    if np is not None:
        rng = np.random.default_rng(0)
        units = [f"img{i:04d}" for i in range(300)]

        def synth(mean_shift: float) -> tuple[dict, dict]:
            """3 seeds of paired per-unit values: candidate = ref + shift + noise."""
            cand, ref = {}, {}
            for seed in (42, 123, 7):
                base = rng.uniform(0.2, 0.8, size=len(units))
                noise = rng.normal(0, 0.01, size=len(units))
                ref[seed] = list(zip(units, base))
                cand[seed] = list(zip(units, base + mean_shift + noise))
            return cand, ref

        cand, ref = synth(mean_shift=0.02)   # planted, clearly real effect
        r = paired_unit_test(cand, ref, n_boot=2000)
        check("planted +0.02 effect passes", r.passed, r.reason)

        cand, ref = synth(mean_shift=0.0)    # pure noise
        r = paired_unit_test(cand, ref, n_boot=2000)
        check("pure noise does NOT pass", not r.passed,
              f"false positive: {r.reason}")

    # ── 7. P11 replication logic ─────────────────────────────────────────
    print("[7/8] P11 replication logic")
    passing = P10Result(passed=True, reason="synthetic pass")
    failing = P10Result(passed=False, reason="synthetic fail")
    r11 = replication_check(
        per_seed_confirmation_deltas={1000: 0.004, 2000: 0.003},
        combined_vs_baseline=passing, combined_vs_control=passing,
        control_replicated=False,
    )
    check("all-favorable + both tests pass -> Winner", r11.verdict == "Winner")
    r11 = replication_check(
        per_seed_confirmation_deltas={1000: 0.004, 2000: -0.001},
        combined_vs_baseline=passing, combined_vs_control=passing,
        control_replicated=False,
    )
    check("one seed backwards -> Fragile", r11.verdict == "Fragile")
    r11 = replication_check(
        per_seed_confirmation_deltas={1000: 0.004, 2000: 0.003},
        combined_vs_baseline=passing, combined_vs_control=failing,
        control_replicated=True,
    )
    check("combined vs-control fail -> Fragile", r11.verdict == "Fragile")
    check("control_must_replicate fires under threshold",
          control_must_replicate(0.001, noise_floor_s2=0.003)
          and not control_must_replicate(0.010, noise_floor_s2=0.003))
    check("threshold constant is sane", CONTROL_REPLICATION_NOISE_MULTIPLE > 0)

    # ── 8. Calibration + dataset status (informational) ──────────────────
    print("[8/8] Trajectory prerequisites (informational)")
    cal = PROJECT / "trajectory/profiles/calibration.json"
    print(f"  calibration.json: {'present' if cal.exists() else 'MISSING (expected before exp000)'}")
    for name in ("visdrone.yaml", "visdrone_s1.yaml", "visdrone_train.txt",
                 "visdrone_eval.txt", "visdrone_val500.txt",
                 "visdrone_manifest.json",
                 "adapters/yolo26n_visdrone_scratch/configs/yolo26n-visdrone.yaml"):
        present = (PROJECT / name).exists()
        print(f"  {name}: {'present' if present else 'MISSING (run setup/create_visdrone_lists.py)'}")

    if args.with_yolo:
        print("\n[bonus] Real model build (needs ultralytics)")
        try:
            desc = adapter.describe_model()
            # nc=10 detect head is ~30k params lighter than v5's nc=80 build
            # (~2.57M); exact count recorded by exp000's describe_model.
            check("param count ~2.3-2.8M",
                  2.3 < desc["total_params_M"] < 2.8,
                  f"got {desc['total_params_M']:.2f}M")
            baseline = adapter.build_baseline()
            probe = adapter.probe_batch()
            ok, gnorm = adapter.gradient_flow_check(baseline, probe)
            check("probe forward/backward runs", True)
            print(f"  (no mechanism injected: gradient check trivially "
                  f"passed={ok}, gnorm={gnorm:.3f})")
        except ImportError as e:
            print(f"  SKIP — ultralytics not importable: {e}")
        except Exception as e:
            check("model build", False, f"{type(e).__name__}: {e}")

    return _report()


def _report() -> int:
    print()
    if _failures:
        print(f"FAILED — {len(_failures)} check(s):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
