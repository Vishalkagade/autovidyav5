"""exp000 driver — vanilla baseline calibration + diagnosis (Phase-0).

No mechanism, no control: every run here is the from-scratch YOLO26n
baseline. Produces, in order (see run_exp000.sh for the chain):

  probe     generous-cap curve run (seed 42, 200 epochs) -> stage budgets
            via the PRE-REGISTERED rule below
  stage     vanilla run at a chosen stage budget x seed (S2: working seeds
            42/123/7 AND sealed baseline legs 1000/2000; S1: working only)
  units     per-image F1 on the frozen eval (2,158 units), from LAST.PT
            (Tier-3 eval-checkpoint lock)
  diagnose  per-class AP, per-scale recall, top-20 failure units
  assemble  noise floors -> trajectory/profiles/calibration.json (schema-
            compliant, write-once), pre-registered instrument check,
            cost-log rows, provenance -> state/assembly.json

PRE-REGISTERED budget rule (fixed before any curve exists, auditable from
calibration.json's epoch_probe_curve): smooth the probe's per-epoch val500
mAP50 with an 11-epoch centered median; s2_epochs = first epoch at >= 97%
of the smoothed peak, rounded UP to a multiple of 10, clamped to [60, 150];
s1_epochs = s2/3 rounded to a multiple of 10, clamped to [10, 60].

PRE-REGISTERED instrument check (thresholds fixed in adapter program.md):
degeneracy (per-unit F1 exactly 0 or 1) > 70% on any working seed -> FAIL;
S2 noise floor / seed-avg S2 mAP50 > 0.15 -> FAIL. Phase-1 never starts on
a failed instrument.

Writing baseline.json's candidate failure modes stays a judgment task done
from diagnostics.json AFTER this pipeline finishes — never automated here
(P0 anti-anchoring: the sealed v5 file stays closed until that is done).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

PROJECT = "/home/hpc/v134ce/v134ce15/vishal/autovidya/autovidya_visdrone"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

STATE = os.path.join(PROJECT, "trajectory", "scratch", "exp000", "state")
os.makedirs(STATE, exist_ok=True)

PROBE_EPOCHS = 200
WORKING = (42, 123, 7)
SEALED = (1000, 2000)


# ── budget rule (pure; selftest pins it) ──────────────────────────────────
def _median_smooth(curve: list[float], w: int = 11) -> list[float]:
    h = w // 2
    return [statistics.median(curve[max(0, i - h):i + h + 1])
            for i in range(len(curve))]


def pick_budgets(curve: list[float]) -> dict:
    sm = _median_smooth(curve)
    peak = max(sm)
    thresh = 0.97 * peak
    first = next(i + 1 for i, v in enumerate(sm) if v >= thresh)  # 1-indexed epoch
    s2 = min(150, max(60, ((first + 9) // 10) * 10))
    s1 = min(60, max(10, round(s2 / 3 / 10) * 10))
    return {"s1_epochs": s1, "s2_epochs": s2,
            "rule": "97%-of-smoothed-peak, ceil10, s2 in [60,150], s1=s2/3 in [10,60]",
            "first_epoch_at_97pct": first, "smoothed_peak": peak}


# ── training (vanilla only — no surgery, so plain Model.train is safe) ────
def train_vanilla(epochs: int, seed: int, name: str) -> dict:
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A

    t0 = time.time()
    model = YOLO(A.MODEL_YAML)
    model.train(
        data=A.TRAIN_DATA_YAML, epochs=epochs, imgsz=A.IMG_SIZE, batch=A.BATCH,
        seed=seed, device=0, workers=8, amp=A.AMP, val=True, plots=False,
        pretrained=False, project=os.path.join(PROJECT, "runs_visdrone"),
        name=name, exist_ok=True, optimizer="MuSGD",
    )
    save_dir = str(model.trainer.save_dir)
    curve = _read_curve(save_dir)
    # Tier-3 eval-checkpoint lock: headline metrics from LAST.PT only.
    last = YOLO(os.path.join(save_dir, "weights", "last.pt"))
    final = last.val(data=A.DATA_YAML, imgsz=A.IMG_SIZE, batch=A.BATCH,
                     plots=False, verbose=False, device=0)
    return {
        "seed": seed, "epochs": epochs, "save_dir": save_dir,
        "primary": float(final.box.map50),
        "secondary": {"mAP50_95": float(final.box.map),
                      "precision": float(final.box.mp),
                      "recall": float(final.box.mr)},
        "curve_val500_map50": curve,
        "wall_min": (time.time() - t0) / 60.0,
    }


def _read_curve(save_dir: str) -> list[float]:
    try:
        with open(os.path.join(save_dir, "results.csv")) as f:
            rows = list(csv.DictReader(f))
        col = next(c for c in rows[0] if c.strip().endswith("mAP50(B)"))
        return [float(r[col]) for r in rows]
    except Exception:
        return []


def _write(name: str, obj: dict) -> None:
    path = os.path.join(STATE, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[driver] wrote {path}")


def _load(name: str) -> dict:
    with open(os.path.join(STATE, name)) as f:
        return json.load(f)


# ── commands ──────────────────────────────────────────────────────────────
def cmd_probe(_args) -> None:
    r = train_vanilla(PROBE_EPOCHS, 42, "exp000_probe_seed42")
    budgets = pick_budgets(r["curve_val500_map50"])
    r["budgets"] = budgets
    r["min_per_epoch"] = r["wall_min"] / PROBE_EPOCHS
    _write("probe.json", r)
    print(f"[driver] budgets: {budgets}")


def cmd_stage(args) -> None:
    budgets = _load("probe.json")["budgets"]
    epochs = budgets["s1_epochs"] if args.stage == "S1" else budgets["s2_epochs"]
    r = train_vanilla(epochs, args.seed, f"exp000_{args.stage}_seed{args.seed}")
    r["stage"] = args.stage
    _write(f"{args.stage}_seed{args.seed}.json", r)


def cmd_units(args) -> None:
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter

    model = YOLO(args.weights)
    units = Yolo26nVisdroneScratchAdapter().per_unit_metric(model)
    vals = [v for _, v in units]
    deg = sum(1 for v in vals if v in (0.0, 1.0)) / len(vals)
    _write(f"per_unit_S2_seed{args.seed}.json", {
        "seed": args.seed, "weights": args.weights, "n_units": len(units),
        "mean_f1": sum(vals) / len(vals),
        "degenerate_frac": deg,
        "f1_zero_frac": sum(1 for v in vals if v == 0.0) / len(vals),
        "units": units,
    })


def cmd_diagnose(args) -> None:
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter

    adapter = Yolo26nVisdroneScratchAdapter()
    model = YOLO(args.weights)
    per_class = adapter.per_class_metric(model)
    per_scale = adapter.per_scale_metric(model)
    units = _load(f"per_unit_S2_seed{args.seed}.json")["units"]
    worst = sorted(units, key=lambda t: t[1])[:20]
    _write("diagnostics.json", {
        "seed": args.seed, "weights": args.weights,
        "per_class_ap50": per_class,
        "per_scale_recall": per_scale,
        "top20_failure_units": [{"unit_id": u, "f1": v} for u, v in worst],
    })


def cmd_assemble(_args) -> None:
    probe = _load("probe.json")
    s1 = {s: _load(f"S1_seed{s}.json") for s in WORKING}
    s2w = {s: _load(f"S2_seed{s}.json") for s in WORKING}
    s2c = {s: _load(f"S2_seed{s}.json") for s in SEALED}
    pu = {s: _load(f"per_unit_S2_seed{s}.json") for s in WORKING}

    floor_s1 = max(r["primary"] for r in s1.values()) - min(r["primary"] for r in s1.values())
    floor_s2 = max(r["primary"] for r in s2w.values()) - min(r["primary"] for r in s2w.values())
    avg_s2 = sum(r["primary"] for r in s2w.values()) / len(s2w)

    cal_path = os.path.join(PROJECT, "trajectory", "profiles", "calibration.json")
    if os.path.exists(cal_path):
        print("[driver] calibration.json already exists — NOT overwriting (write-once)")
    else:
        with open(cal_path, "w") as f:
            json.dump({
                "s1_epochs": probe["budgets"]["s1_epochs"],
                "s2_epochs": probe["budgets"]["s2_epochs"],
                "noise_floor_s1_map50": floor_s1,
                "noise_floor_s2_map50": floor_s2,
                "measured_by": "exp000",
                "seeds_used": {"working": list(WORKING), "confirmation": list(SEALED)},
                "epoch_probe_curve": probe["curve_val500_map50"],
                "notes": (f"budget rule: {probe['budgets']['rule']}; "
                          f"min/epoch={probe['min_per_epoch']:.3f}; "
                          "eval checkpoint locked to last.pt (2026-09-07)"),
            }, f, indent=2)
        print(f"[driver] wrote {cal_path}")

    # Pre-registered instrument check (thresholds in adapter program.md).
    deg = {s: pu[s]["degenerate_frac"] for s in WORKING}
    rel_floor = floor_s2 / avg_s2 if avg_s2 > 0 else float("inf")
    verdict = {
        "degeneracy_by_seed": deg,
        "degeneracy_pass": all(d <= 0.70 for d in deg.values()),
        "relative_noise_floor": rel_floor,
        "relative_floor_pass": rel_floor <= 0.15,
    }
    verdict["instrument_pass"] = verdict["degeneracy_pass"] and verdict["relative_floor_pass"]

    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()

    # cost ledger
    runs = [("probe", 42, probe)] + \
           [("S1", s, r) for s, r in s1.items()] + \
           [("S2", s, r) for s, r in {**s2w, **s2c}.items()]
    gpu_h = 0.0
    with open(os.path.join(PROJECT, "trajectory", "cost_log.csv"), "a") as f:
        for stage, seed, r in runs:
            h = r["wall_min"] / 60.0
            gpu_h += h
            f.write(f"{seed},{stage},{h:.3f},{gpu_h:.3f}\n")

    _write("assembly.json", {
        "budgets": probe["budgets"],
        "min_per_epoch": probe["min_per_epoch"],
        "noise_floor_s1_map50": floor_s1,
        "noise_floor_s2_map50": floor_s2,
        "baseline_s2_map50_by_seed": {str(s): r["primary"] for s, r in {**s2w, **s2c}.items()},
        "baseline_s2_seedavg_map50_working": avg_s2,
        "per_unit_mean_f1_by_seed": {str(s): pu[s]["mean_f1"] for s in WORKING},
        "f1_zero_frac_by_seed": {str(s): pu[s]["f1_zero_frac"] for s in WORKING},
        "instrument_check": verdict,
        "gpu_hours_total": gpu_h,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "data_manifest_sha256": _sha(os.path.join(PROJECT, "visdrone_manifest.json")),
            "ultralytics_commit": _git("-C", "../ultralytics_src", "rev-parse", "HEAD"),
        },
    })
    print(f"[driver] INSTRUMENT_{'PASS' if verdict['instrument_pass'] else 'FAIL'} "
          f"deg={max(deg.values()):.3f} rel_floor={rel_floor:.3f} gpu_h={gpu_h:.1f}")


def cmd_selftest(_args) -> None:
    """No-GPU wiring check: imports, paths, and the budget rule on synthetic
    curves. Run on the login node before sbatch (driver dry-run practice)."""
    ok = True
    # budget rule pins
    flat_late = [0.01 * min(i, 80) / 80 for i in range(1, 201)]   # flattens ~epoch 78
    b = pick_budgets(flat_late)
    ok &= b["s2_epochs"] == 80 and b["s1_epochs"] == 30
    print(f"  budget rule (flatten@~78): s2={b['s2_epochs']} s1={b['s1_epochs']} "
          f"{'OK' if ok else 'FAIL'}")
    fast = [0.01] * 200                                            # instant plateau
    b2 = pick_budgets(fast)
    ok2 = b2["s2_epochs"] == 60 and b2["s1_epochs"] == 20          # clamp floor
    print(f"  budget rule (instant plateau): s2={b2['s2_epochs']} s1={b2['s1_epochs']} "
          f"{'OK' if ok2 else 'FAIL'}")
    slow = [0.0001 * i for i in range(1, 201)]                     # never flattens
    b3 = pick_budgets(slow)
    ok3 = b3["s2_epochs"] == 150                                   # clamp ceiling
    print(f"  budget rule (never flattens): s2={b3['s2_epochs']} "
          f"{'OK' if ok3 else 'FAIL'}")
    # wiring
    from adapters.yolo26n_visdrone_scratch import adapter as A
    paths_ok = all(os.path.exists(p) for p in
                   (A.MODEL_YAML, A.DATA_YAML, A.TRAIN_DATA_YAML))
    print(f"  lock files exist: {'OK' if paths_ok else 'FAIL'}")
    print("SELFTEST", "PASS" if (ok and ok2 and ok3 and paths_ok) else "FAIL")
    sys.exit(0 if (ok and ok2 and ok3 and paths_ok) else 1)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sp = sub.add_parser("stage")
    sp.add_argument("--stage", choices=["S1", "S2"], required=True)
    sp.add_argument("--seed", type=int, required=True)
    su = sub.add_parser("units")
    su.add_argument("--seed", type=int, required=True)
    su.add_argument("--weights", required=True)
    sd = sub.add_parser("diagnose")
    sd.add_argument("--seed", type=int, required=True)
    sd.add_argument("--weights", required=True)
    sub.add_parser("assemble")
    sub.add_parser("selftest")
    args = p.parse_args()
    {"probe": cmd_probe, "stage": cmd_stage, "units": cmd_units,
     "diagnose": cmd_diagnose, "assemble": cmd_assemble,
     "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
