"""Negative-trained masked anchors for the band-gated stride-4 level (exp015, Phase-2 axis R3-coupling).

Per-level breakdown (exp014 record): every gating loses ~0.005 on SKU-110K with recall nearly intact but 400-600 extra
duplicates per 300 images, because a stride-4 anchor that is masked in training receives NO signal (its outputs are
replaced by constants) and so is never trained as a negative for objects the stride-8 anchor owns; when its cell is in
the band at inference it fires. Here, in training, the stride-4 level keeps its real outputs; the live mask is applied
INSIDE the assigner (a masked anchor can never be a positive), so masked anchors are trained as negatives; inference
masks exactly as the anchor does.
"""
from __future__ import annotations
import torch
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class LiveMaskAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner whose first `n_small` anchors are candidates only where `live` (B, n_small) is True."""
    n_small = 0; stats = None   # the head is kept in a plain list (an nn.Module attribute would be registered as a submodule and shadowed by a class default)

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        if self.n_small:
            live = self._det_ref[0].last_live.to(mask_in_gts.device)               # (B, n_small)
            keep = torch.ones_like(mask_in_gts, dtype=torch.bool); keep[..., : self.n_small] = live.unsqueeze(1)
            mask_in_gts = mask_in_gts * keep
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        if self.stats is not None and self.n_small:
            self.stats["batches"] += 1; self.stats["pos_small_level"] += int(mask_pos[..., : self.n_small].sum())
            self.stats["pos_on_masked"] += int((mask_pos[..., : self.n_small] * (~live).unsqueeze(1)).sum()); self.stats["pos_total"] += int(mask_pos.sum())
        return mask_pos, align_metric, overlaps


def make_loss(det, stats: dict):
    class LiveMaskDetectionLoss(v8DetectionLoss):
        def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
            super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
            self.assigner = LiveMaskAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, stride=self.stride.tolist(), topk2=tal_topk2)
            self.assigner._det_ref, self.assigner.stats = [det], stats

        def loss(self, preds, batch):
            f0 = preds["feats"][0]; self.assigner.n_small = f0.shape[2] * f0.shape[3] if float(self.stride[0]) < 8 else 0
            return super().loss(preds, batch)
    return LiveMaskDetectionLoss


def attach(model) -> dict:
    """model.criterion = E2ELoss(loss_fn=LiveMaskDetectionLoss) reading the live mask from the BandGatedDetect head; returns the tripwire stats."""
    det = model.model[-1]; assert type(det).__name__ == "BandGatedDetect" and det.train_mask == "negative", (type(det).__name__, getattr(det, "train_mask", None))
    stats = {"batches": 0, "pos_small_level": 0, "pos_on_masked": 0, "pos_total": 0}
    model.criterion = E2ELoss(model, loss_fn=make_loss(det, stats))
    return stats


def demo() -> None:
    import os, sys; sys.path.insert(0, os.getcwd())
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from adapters.yolo26n_visdrone_scratch.modules import bandgate as BG
    yaml = os.path.join("adapters", "yolo26n_visdrone_scratch", "configs", "yolo26n-p2-visdrone.yaml")
    m = YOLO(yaml).model; BG.apply_surgery(m, "band", train_mask="negative"); m.args = get_cfg(); m.train()
    torch.manual_seed(0); x = torch.rand(2, 3, 640, 640); preds = m(x)
    assert (preds["one2one"]["scores"][..., :25600] == BG.MASK_LOGIT).float().mean() < 0.01 and (preds["one2one"]["boxes"][..., :25600] != 0).any(1).float().mean() > 0.9, "training outputs must be the real stride-4 outputs (no replacement)"
    live = m.model[-1].last_live; assert live.shape == (2, 25600) and abs(live.float().mean().item() - 0.25) < 0.01
    boxes = torch.tensor([[0.2, 0.2, 0.0125, 0.0125], [0.5, 0.5, 0.03, 0.03], [0.3, 0.7, 0.1, 0.1], [0.8, 0.8, 0.06, 0.06]] * 2)
    batch = {"batch_idx": torch.tensor([0] * 4 + [1] * 4).float(), "cls": torch.zeros(8, 1), "bboxes": boxes}
    stats = attach(m); loss, items = m.criterion(preds, batch); assert torch.isfinite(loss).all() and stats["batches"] == 2, stats
    assert stats["pos_on_masked"] == 0, stats                                       # masked anchors are never positives
    # masked anchors DO receive gradient (trained as negatives): the stride-4 cls head gets gradient from masked positions
    m.zero_grad(); loss.sum().backward(); det = m.model[-1]
    g = sum(float(p_.grad.norm()) for p_ in det.cv3[0].parameters() if p_.grad is not None); assert g > 0, g
    # inference masks exactly as the anchor
    m.eval()
    with torch.no_grad(): y, raw = m(x)
    assert (raw["one2one"]["boxes"][..., :25600] != 0).any(1).float().mean().item() <= 0.25 + 1e-6 and y.shape == (2, 300, 6)
    print(f"negmask demo OK: live {live.float().mean():.3f}, stride-4 positives {stats['pos_small_level']} (on masked: {stats['pos_on_masked']}) of {stats['pos_total']}, stride-4 cls-head grad {g:.3f}")


if __name__ == "__main__":
    demo()
