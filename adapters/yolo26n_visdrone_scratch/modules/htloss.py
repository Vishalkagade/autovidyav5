"""Horvitz-Thompson detection-probability loss reweighting (exp004 mechanism).

Source domain: statistical ecology / survey sampling. Under size-dependent
detection probability p(size), an unbiased estimate weights each detected
unit by 1/p (Horvitz & Thompson 1952; distance sampling, Buckland et al.).
Applied to the training objective: every positive target's cls and box loss
is multiplied by 1/p(scale bucket), p measured on the Phase-0 baseline, and
the weights are normalised to mean 1 over the batch's positives so the loss
scale is unchanged. Zero parameters.

Control (P9, designed for this experiment): the SAME weight multiset per
batch, randomly permuted across the positive anchors — identical loss scale
and variance, no size structure.

Tripwire (ins_v5_aux_objective_dead): `applied_batches`, running mean weight,
and the running weight-vs-area correlation (mechanism > 0, control ~ 0).
"""
from __future__ import annotations
import json
import torch
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.utils.tal import make_anchors


def make_ht_loss(variant: str, scale_json: str, weights: dict[str, float], seed: int):
    """Returns (loss_fn class for E2ELoss, shared stats dict)."""
    s = json.load(open(scale_json))["native_to_input_scale"]
    thr_small, thr_med = (32 * s) ** 2, (96 * s) ** 2          # COCO buckets mapped to input pixels
    wv = torch.tensor([weights["small"], weights["medium"], weights["large"]])
    stats = {"applied_batches": 0, "sum_w": 0.0, "n_pos": 0, "sum_wa": 0.0, "sum_a": 0.0, "sum_ww": 0.0, "sum_aa": 0.0,
             "variant": variant, "thr_small_px2": thr_small, "thr_med_px2": thr_med}
    gen = torch.Generator().manual_seed(seed)

    class HTLoss(v8DetectionLoss):
        def get_assigned_targets_and_loss(self, preds, batch):
            loss = torch.zeros(3, device=self.device)
            pred_distri, pred_scores = preds["boxes"].permute(0, 2, 1).contiguous(), preds["scores"].permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
            dtype = pred_scores.dtype; batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2); mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
            # ── HT weights per positive anchor ─────────────────────────────
            w = torch.ones(target_scores.shape[:2], device=self.device, dtype=target_scores.dtype)   # (B, A)
            if fg_mask.any():
                tb = target_bboxes[fg_mask]                                     # input-pixel xyxy
                area = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])).float()
                bucket = (area >= thr_small).long() + (area >= thr_med).long()  # 0 small, 1 medium, 2 large
                wpos = wv.to(self.device)[bucket]
                if variant == "control":
                    perm = torch.randperm(wpos.numel(), generator=gen).to(self.device); wpos = wpos[perm]
                wpos = wpos / wpos.mean()                                       # loss scale unchanged
                w[fg_mask] = wpos.to(w.dtype)
                a = area.detach().cpu(); ww = wpos.detach().cpu().float()
                stats["n_pos"] += int(a.numel()); stats["sum_w"] += float(ww.sum()); stats["sum_a"] += float(a.sum())
                stats["sum_wa"] += float((ww * a).sum()); stats["sum_ww"] += float((ww * ww).sum()); stats["sum_aa"] += float((a * a).sum())
            stats["applied_batches"] += 1
            target_scores_w = target_scores * w.unsqueeze(-1)
            target_scores_sum = max(target_scores.sum(), 1)
            loss[1] = (self.bce(pred_scores, target_scores.to(dtype)) * w.unsqueeze(-1)).sum() / target_scores_sum
            if fg_mask.sum():
                loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                                  target_scores_w, target_scores_sum, fg_mask, imgsz, stride_tensor)
            loss[0] *= self.hyp.box; loss[1] *= self.hyp.cls; loss[2] *= self.hyp.dfl
            return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()

    return HTLoss, stats


def corr_from_stats(st: dict) -> float | None:
    n = st["n_pos"]
    if n < 2: return None
    mw, ma = st["sum_w"] / n, st["sum_a"] / n
    cov = st["sum_wa"] / n - mw * ma
    vw, va = st["sum_ww"] / n - mw * mw, st["sum_aa"] / n - ma * ma
    return cov / ((vw * va) ** 0.5) if vw > 0 and va > 0 else None


def attach(model, variant: str, scale_json: str, weights: dict[str, float], seed: int) -> dict:
    """Attach E2ELoss(loss_fn=HTLoss) as model.criterion; returns the shared stats dict."""
    HT, stats = make_ht_loss(variant, scale_json, weights, seed)
    model.criterion = E2ELoss(model, loss_fn=HT)
    return stats
