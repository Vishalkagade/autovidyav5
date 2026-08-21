"""YOLO26n + VisDrone FROM-SCRATCH adapter.

Implements core.adapter.Adapter for the bold regime on VisDrone-DET:
- YOLO26n built from a local yaml with nc=10 (random init), trained from
  scratch on the full VisDrone train split (6,471 images), evaluated on the
  FROZEN eval set: val + test-dev = 2,158 images (user decision 2026-08-21).
- Mechanisms are structural (wrap/replace whole blocks), budgeted by P9
  (param_budget_fraction), attributed via matched controls.
- per_unit_metric returns TRUE per-image values (per-image detection F1 at
  IoU 0.5) — never batch-aggregated (see P10 postmortem).

Calibration: stage budgets and noise floors are NOT constants in this file.
exp000 measures them and writes trajectory/profiles/calibration.json (see
core/schemas/calibration.schema.json); this adapter reads that file at
runtime. Reason: the agent is forbidden from editing adapter.py (guard rail),
so measured state must live in the agent-writable trajectory, not in code.
Until calibration.json exists, epoch budgets fall back to rough placeholders
and noise_floor() raises — which is exactly the P0 block.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time
from typing import Any

import torch
from torch import Tensor, nn

from core.adapter import ControlFamily, Site, StageMetrics, StageName
from .sites import SITE_INDEX, SITES

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))

# ── Tier-3 locks ──────────────────────────────────────────────────────────
# Local yaml = ultralytics yolo26.yaml with nc: 10 (rendered by
# setup/create_visdrone_lists.py). The filename keeps the "yolo26n" prefix so
# ultralytics resolves scale 'n' from it. NO pretrained weights anywhere.
MODEL_YAML = os.path.join(_HERE, "configs", "yolo26n-visdrone.yaml")
IMG_SIZE = 640
BATCH = 32
AMP = False                          # cluster constraint — never override

# Headline metrics ALWAYS come from a full frozen-eval pass with this yaml
# (val: visdrone_eval.txt = val + test-dev, 2,158 images):
DATA_YAML = os.path.join(_PROJECT, "visdrone.yaml")
# Training-time validation uses this yaml instead: same train list, but val
# is the locked 500-image subset (visdrone_val500.txt). Per-epoch val on
# 2,158 large images would cost minutes/epoch; the 500-image subset is cheap
# and its only job is the P12 learning curve. It NEVER produces headline
# numbers.
TRAIN_DATA_YAML = os.path.join(_PROJECT, "visdrone_s1.yaml")

# Seed sets (Tier-3, both locked before the trajectory starts):
WORKING_SEEDS = (42, 123, 7)         # all screening, Stage-2, and control runs
# Sealed P11 confirmation seeds — chosen arbitrarily before any experiment
# ran. NO run may use them except exp000's baseline legs and a P11
# replication of a Provisional Winner. See program.md "Seed sets".
CONFIRMATION_SEEDS = (1000, 2000)
SEEDS = WORKING_SEEDS                # what classification runs iterate over

# ── Calibration (measured by exp000, stored as a trajectory artifact) ─────
CALIBRATION_PATH = os.path.join(_PROJECT, "trajectory", "profiles", "calibration.json")
_PLACEHOLDER_EPOCHS = {"S1": 30, "S2": 100}   # used ONLY before calibration


def _calibration() -> dict:
    """exp000's measured numbers, or {} before calibration.json exists."""
    try:
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def stage_epochs(stage: StageName) -> int:
    cal = _calibration()
    key = "s1_epochs" if stage == "S1" else "s2_epochs"
    if key in cal and cal[key] is not None:
        return int(cal[key])
    return _PLACEHOLDER_EPOCHS[stage]


class Yolo26nVisdroneScratchAdapter:
    """One P10 unit = one frozen-eval image (per-image detection F1)."""

    name = "yolo26n_visdrone_scratch"
    primary_metric_name = "mAP50"
    primary_metric_higher_is_better = True
    param_budget_fraction = 0.10     # bold regime: up to 10% of baseline params

    # ─── construction ────────────────────────────────────────────────────
    def build_baseline(self) -> Any:
        from ultralytics import YOLO
        return YOLO(MODEL_YAML)      # random init; no .pt anywhere

    def inject_mechanism(self, baseline: Any, mechanism_spec: dict) -> Any:
        """mechanism_spec kinds:

        {"kind": "site_wrap", "site": <sites.py name>, "module_path": str,
         "class_name": str, "kwargs": {...}}
            -> wraps model.model[idx] as Sequential(original, custom) or
               replaces it, per the custom class's contract.
        {"kind": "aux_objective", ...}   -> DRIVER-SIDE ONLY. This kind is
            metadata: nothing in train_and_eval reads model.autovidya_aux
            (ins_harness_aux_objective_dead, inherited from v5). Objective
            mechanisms implement their loss in the experiment driver with an
            applied-batches tripwire.
        {"kind": "topology", "yaml": str} -> build from a modified yaml.

        HAZARD (inherited, mandatory): Model.train rebuilds the model from
        yaml and silently drops in-place site_wrap surgery. All site_wrap
        training goes through the SurgeryTrainer pattern in the experiment
        driver (override DetectionTrainer.get_model to re-apply the wrap),
        never through a bare model.train() on the surgered object.
        """
        model = copy.deepcopy(baseline)
        kind = mechanism_spec.get("kind")
        if kind == "site_wrap":
            import importlib
            idx = SITE_INDEX[mechanism_spec["site"]]
            mod = importlib.import_module(mechanism_spec["module_path"])
            cls = getattr(mod, mechanism_spec["class_name"])
            original = model.model.model[idx]
            wrapped = cls(original, **mechanism_spec.get("kwargs", {}))
            before = sum(p.numel() for p in original.parameters())
            after = sum(p.numel() for p in wrapped.parameters())
            model.model.model[idx] = wrapped
            meta = {"param_count_added": after - before,
                    "site": mechanism_spec["site"], "kind": kind}
        elif kind == "topology":
            from ultralytics import YOLO
            model = YOLO(mechanism_spec["yaml"])
            base_params = sum(p.numel() for p in baseline.model.parameters())
            new_params = sum(p.numel() for p in model.model.parameters())
            meta = {"param_count_added": new_params - base_params, "kind": kind}
        elif kind == "aux_objective":
            meta = {"param_count_added": int(mechanism_spec.get("extra_params", 0)),
                    "kind": kind, "spec": mechanism_spec}
            model.autovidya_aux = mechanism_spec
        else:
            raise ValueError(f"unknown mechanism kind: {kind}")
        model.autovidya_meta = meta
        return model

    def build_control(self, baseline: Any, mechanism_spec: dict,
                      family: ControlFamily) -> Any:
        """Matched controls for P9."""
        if family == "param_matched_generic":
            # Generic residual conv block at the same site, width chosen to
            # land within ±10% of the mechanism's added params.
            target = mechanism_spec["_mechanism_param_count"]
            spec = {
                "kind": "site_wrap",
                "site": mechanism_spec["site"],
                "module_path": "adapters.yolo26n_visdrone_scratch.generic_control",
                "class_name": "GenericResidualConv",
                "kwargs": {"target_params": target},
            }
            return self.inject_mechanism(baseline, spec)
        if family == "frozen_random":
            model = self.inject_mechanism(baseline, mechanism_spec)
            site_idx = SITE_INDEX[mechanism_spec["site"]]
            wrapped = model.model.model[site_idx]
            for name, p in wrapped.named_parameters():
                # freeze only the mechanism's own (new) params; the original
                # block keeps training
                if "original" not in name:
                    p.requires_grad_(False)
            model.autovidya_meta["control_family"] = "frozen_random"
            return model
        raise ValueError(f"unknown control family: {family}")

    # ─── training ────────────────────────────────────────────────────────
    def train_and_eval(self, model: Any, *, stage: StageName, seed: int) -> StageMetrics:
        """Train from scratch, then evaluate.

        Two-yaml design (see the constants at the top of this file):
        - training runs with TRAIN_DATA_YAML: per-epoch val on the 500-image
          subset, which is cheap and yields the P12 learning curve;
        - the returned `primary`/`secondary` come from ONE full frozen-eval
          pass with DATA_YAML after training. Headline numbers never come
          from the subset.

        NOTE: valid for the vanilla baseline and topology mechanisms only.
        site_wrap mechanisms MUST train via the driver-side SurgeryTrainer
        (Model.train rebuilds from yaml and drops in-place surgery).
        """
        epochs = stage_epochs(stage)
        t0 = time.time()
        model.train(
            data=TRAIN_DATA_YAML, epochs=epochs, imgsz=IMG_SIZE, batch=BATCH,
            seed=seed, device=0, workers=8, amp=AMP, val=True, plots=False,
            pretrained=False, project=os.path.join(_PROJECT, "runs_visdrone"),
            name=f"{self.name}_{stage}_seed{seed}_{int(t0)}", exist_ok=True,
            # Tier-2 lock. Ultralytics' optimizer='auto' picks MuSGD when
            # total iterations > 10000, else AdamW — at batch 32 on 6,471
            # images (~203 iters/epoch) that flip sits near 50 EPOCHS: short
            # and long runs would silently get different optimizers. Pinning
            # MuSGD keeps every run on the same optimizer regardless of
            # epoch count.
            optimizer="MuSGD",
        )
        curve = self._read_epoch_curve(model)
        final = model.val(data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH,
                          plots=False, verbose=False, device=0)
        return StageMetrics(
            primary=float(final.box.map50),
            secondary={
                "mAP50_95": float(final.box.map),
                "precision": float(final.box.mp),
                "recall": float(final.box.mr),
            },
            seed=seed, stage=stage, epochs_or_seconds=time.time() - t0,
            curve=curve,
        )

    def _read_epoch_curve(self, model: Any) -> list[float]:
        """Per-epoch mAP50 (on the 500-image val subset) from the run's
        results.csv. Diagnostic only — an unreadable csv returns [] rather
        than failing the run."""
        try:
            csv_path = os.path.join(str(model.trainer.save_dir), "results.csv")
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            col = next(c for c in rows[0] if c.strip().endswith("mAP50(B)"))
            return [float(r[col]) for r in rows]
        except Exception:
            return []

    # ─── diagnostics ─────────────────────────────────────────────────────
    def probe_batch(self) -> Tensor:
        g = torch.Generator().manual_seed(0)
        return torch.rand(2, 3, IMG_SIZE, IMG_SIZE, generator=g)

    def gradient_flow_check(self, model: Any, probe: Tensor) -> tuple[bool, float]:
        """Mandatory pre-screen gate (P6'): mechanism params receive grads."""
        net = model.model  # DetectionModel
        net.train()
        meta = getattr(model, "autovidya_meta", {})
        site = meta.get("site")
        out = net(probe)
        # end2end detect returns dict/tuple in train mode; reduce to a scalar
        flat = out
        while isinstance(flat, (tuple, list, dict)):
            flat = list(flat.values())[0] if isinstance(flat, dict) else flat[0]
        loss = flat.float().abs().mean()
        net.zero_grad()
        loss.backward()
        if site is None:
            return True, 0.0  # no site-bound mechanism (e.g. aux objective)
        idx = SITE_INDEX[site]
        gnorm = 0.0
        for p in net.model[idx].parameters():
            if p.grad is not None:
                gnorm += float(p.grad.norm())
        return gnorm > 0.0, gnorm

    def sites(self) -> list[Site]:
        return list(SITES)

    def describe_model(self) -> dict:
        base = self.build_baseline()
        mods = []
        for i, m in enumerate(base.model.model):
            mods.append({"idx": i, "type": type(m).__name__})
        return {
            "name": "yolo26n nc=10 (from yaml, random init)",
            "total_params_M": sum(p.numel() for p in base.model.parameters()) / 1e6,
            "modules": mods,
            "patch_compatible_sites": list(SITE_INDEX),
            "other_intervention_modes": [
                "aux_objective (training-time loss terms — driver-side only)",
                "topology (modified yaml — new connectivity)",
            ],
            "notes": "end2end NMS-free detect; reg_max=1 (no DFL); C2PSA attention at backbone tail",
        }

    def per_unit_metric(self, model: Any, split: str = "val") -> list[tuple[str, float]]:
        """Per-image detection F1 at IoU 0.5, conf 0.25 — TRUE per-image values.

        Units = the frozen eval set (visdrone_eval.txt, 2,158 images).
        The actual scoring lives in the pure function `image_f1` below, so the
        smoke test can pin its conventions with fixtures (no model needed).
        """
        import yaml as _yaml

        with open(DATA_YAML) as f:
            data_cfg = _yaml.safe_load(f)
        val_images = self._val_image_list(data_cfg)

        # One predict call per image. ultralytics 8.4.x loads a LIST source
        # as a SINGLE batch (LoadPilAndNumpy.bs = len(list)) — at 2,158 eval
        # images that is one giant forward pass, guaranteed OOM. Per-image
        # calls are the canonical deterministic path (v5 postmortem,
        # user-authorized 2026-08-02).
        results = (model.predict(p, imgsz=IMG_SIZE, conf=0.25,
                                 verbose=False, device=0)[0]
                   for p in val_images)
        out = []
        for r in results:
            img_id = os.path.basename(r.path)
            gt = self._load_gt(r.path, data_cfg)  # [[cls, x1,y1,x2,y2], ...] px
            preds = [
                (int(r.boxes.cls[i]), float(r.boxes.conf[i]),
                 *[float(v) for v in r.boxes.xyxy[i]])
                for i in range(len(r.boxes))
            ]
            out.append((img_id, image_f1(preds, gt)))
        return out

    def _val_image_list(self, data_cfg: dict) -> list[str]:
        """The frozen eval list. `val` here is a TXT file of absolute image
        paths (val + test-dev) — unlike v5, where it was a directory."""
        root = data_cfg.get("path", "")
        val = data_cfg["val"]
        val_path = val if os.path.isabs(val) else os.path.join(root, val)
        if val_path.endswith(".txt"):
            with open(val_path) as f:
                return sorted(line.strip() for line in f if line.strip())
        return sorted(
            os.path.join(val_path, f) for f in os.listdir(val_path)
            if f.lower().endswith((".jpg", ".png"))
        )

    def _load_gt(self, image_path: str, data_cfg: dict) -> list[list[float]]:
        import numpy as np
        from PIL import Image
        label_path = (
            image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        )
        if not os.path.exists(label_path):
            return []
        w, h = Image.open(image_path).size
        rows = []
        with open(label_path) as f:
            for line in f:
                c, xc, yc, bw, bh = map(float, line.split()[:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                rows.append([c, x1, y1, x2, y2])
        return rows

    def per_site_stats(self, model: Any) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        hooks = []
        net = model.model

        def mk_hook(name):
            def hook(_m, _i, o):
                t = o[0] if isinstance(o, (tuple, list)) else o
                if torch.is_tensor(t):
                    stats[name] = {
                        "mean": float(t.mean()), "std": float(t.std()),
                        "frac_zero": float((t == 0).float().mean()),
                    }
            return hook

        for sname, idx in SITE_INDEX.items():
            hooks.append(net.model[idx].register_forward_hook(mk_hook(sname)))
        with torch.no_grad():
            net.eval()
            net(self.probe_batch())
        for hk in hooks:
            hk.remove()
        return stats

    def mechanism_activation_stats(self, model: Any) -> dict[str, float]:
        meta = getattr(model, "autovidya_meta", None)
        if not meta or "site" not in meta:
            return {}
        site_stats = self.per_site_stats(model)
        return site_stats.get(meta["site"], {})

    # ─── Phase-0 diagnostics ─────────────────────────────────────────────
    def per_class_metric(self, model: Any, split: str = "val") -> list[dict[str, Any]]:
        m = model.val(data=DATA_YAML, imgsz=IMG_SIZE, plots=False, verbose=False)
        out = []
        for i, ap in zip(m.box.ap_class_index, m.box.ap50):
            out.append({"class_name": model.names[int(i)], "value": float(ap)})
        return out

    def per_scale_metric(self, model: Any, split: str = "val") -> dict[str, float]:
        """Per-scale detection RECALL at IoU 0.5 / conf 0.25, bucketed by GT
        box area with the COCO convention (small < 32^2 <= medium < 96^2 <= large),
        measured in NATIVE image pixels (VisDrone frames are ~1400-2000 px wide,
        so the "small" bucket is far below the model's 640-px input stride —
        exactly the regime this trajectory targets).

        NOT COCO mAP-by-scale: ultralytics 8.4.x does not expose per-scale AP
        and pycocotools is not in the offline env. Recall by scale answers the
        same diagnostic question for exp000 ("which object sizes does the
        baseline miss"), using the same greedy matcher as the P10 unit metric.
        """
        import yaml as _yaml

        with open(DATA_YAML) as f:
            data_cfg = _yaml.safe_load(f)
        val_images = self._val_image_list(data_cfg)

        # Per-image predict — same OOM rationale as per_unit_metric above.
        results = (model.predict(p, imgsz=IMG_SIZE, conf=0.25,
                                 verbose=False, device=0)[0]
                   for p in val_images)
        matched = {"small": 0, "medium": 0, "large": 0}
        total = {"small": 0, "medium": 0, "large": 0}
        for r in results:
            gt = self._load_gt(r.path, data_cfg)
            if not gt:
                continue
            preds = [
                (int(r.boxes.cls[i]), float(r.boxes.conf[i]),
                 *[float(v) for v in r.boxes.xyxy[i]])
                for i in range(len(r.boxes))
            ]
            matched_idx = _greedy_match(preds, gt)
            for j, (_, x1, y1, x2, y2) in enumerate(gt):
                bucket = _area_bucket((x2 - x1) * (y2 - y1))
                total[bucket] += 1
                if j in matched_idx:
                    matched[bucket] += 1
        return {
            f"{b}_recall": (matched[b] / total[b] if total[b] else 0.0)
            for b in ("small", "medium", "large")
        }

    def loss_decomposition(self, model: Any, split: str = "val") -> dict[str, float]:
        raise NotImplementedError("read from final training epoch's loss items instead")

    def top_failure_units(self, model: Any, k: int = 20, split: str = "val") -> list[dict[str, Any]]:
        units = sorted(self.per_unit_metric(model, split), key=lambda t: t[1])[:k]
        return [{"unit_id": u, "value": v} for u, v in units]

    def noise_floor(self, stage: StageName) -> float:
        cal = _calibration()
        key = "noise_floor_s1_map50" if stage == "S1" else "noise_floor_s2_map50"
        v = cal.get(key)
        if v is None:
            raise RuntimeError(
                f"noise floor for {stage} not calibrated — run exp000 "
                f"(>=3 vanilla from-scratch seeds per stage) and write "
                f"{CALIBRATION_PATH} (schema: core/schemas/calibration.schema.json)"
            )
        return float(v)


# ── Per-image F1: pure functions, pinned by smoke-test fixtures ──────────
# If you change ANY convention here, the fixtures in scripts/smoke_test.py
# must change in the same commit — that is the point of them.

def image_f1(
    preds: list[tuple],   # (cls, conf, x1, y1, x2, y2) per prediction, pixels
    gts: list[list],      # (cls, x1, y1, x2, y2) per GT box, pixels
    iou_thresh: float = 0.5,
) -> float:
    """Detection F1 for ONE image.

    Conventions:
    - no GT and no predictions -> 1.0  (image correctly left empty)
    - no GT but predictions    -> 0.0
    - GT but no predictions    -> 0.0
    - matching is GREEDY in descending confidence order: each prediction
      takes the highest-IoU unmatched GT of its own class (strict '>', so on
      an exact IoU tie the earlier GT in file order wins). Greedy can score
      below the optimal assignment — accepted and pinned by fixture.
    """
    n_gt, n_pred = len(gts), len(preds)
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    tp = len(_greedy_match(preds, gts, iou_thresh))
    prec = tp / n_pred
    rec = tp / n_gt
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _greedy_match(preds: list[tuple], gts: list[list],
                  iou_thresh: float = 0.5) -> set[int]:
    """Greedy class-aware matching; returns the indices of matched GT boxes.

    Shared by image_f1 (P10 unit metric) and per_scale_metric (exp000
    diagnostics) so both report the same notion of 'detected'.
    """
    matched_gt: set[int] = set()
    for pcls, _conf, px1, py1, px2, py2 in sorted(preds, key=lambda p: -p[1]):
        best_iou, best_j = 0.0, -1
        for j, (gcls, gx1, gy1, gx2, gy2) in enumerate(gts):
            if j in matched_gt or int(gcls) != int(pcls):
                continue
            iou = _iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
    return matched_gt


def _area_bucket(area: float) -> str:
    """COCO area convention: small < 32^2 <= medium < 96^2 <= large."""
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
