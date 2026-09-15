"""Scale-owned assignment for the band-gated stride-4 level (exp014, Phase-2 axis R3-coupling).

ins_022: with a gated stride-4 level, a live stride-4 anchor that wins the one-to-one assignment for an object the
stride-8 anchor could have owned turns that stride-8 anchor into a negative, keeps its posterior undecided, and so
keeps the cell in the band — a self-reinforcing hand-over that costs recall wherever band membership does not
reproduce at inference (dense medium-object scenes). This coupling restricts OWNERSHIP in the assigner: stride-4
anchors are candidates only for ground truth whose longer side is below `own_thr` px (input scale); larger objects
are assigned among the stride-8/16/32 anchors only. The selection rule (which cells take a second look) is untouched.

Attach with `attach(model, own_thr)` on the trainer-built model (E2ELoss(loss_fn=...) as model.criterion, the
exp004 pattern); `stats` is a tripwire dict filled per batch.
"""
from __future__ import annotations
import torch
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class ScaleOwnedAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner whose first `n_small` anchors (the stride-4 level) may only be candidates for small GT."""
    own_thr = 32.0; n_small = 0; stats = None; report_thr = None   # report_thr: size boundary used for the stats only (defaults to own_thr)

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        if self.n_small:
            side = torch.maximum(gt_bboxes[..., 2] - gt_bboxes[..., 0], gt_bboxes[..., 3] - gt_bboxes[..., 1])   # (B, n_max) px
            own = torch.ones_like(mask_in_gts, dtype=torch.bool); own[..., : self.n_small] = (side < self.own_thr).unsqueeze(-1)
            mask_in_gts = mask_in_gts * own
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        if self.stats is not None and self.n_small:
            large = (side >= (self.report_thr if self.report_thr is not None else self.own_thr)).unsqueeze(-1)
            self.stats["batches"] += 1; self.stats["pos_small_level"] += int(mask_pos[..., : self.n_small].sum())
            self.stats["pos_small_level_on_large_gt"] += int((mask_pos[..., : self.n_small] * large).sum()); self.stats["pos_total"] += int(mask_pos.sum())
        return mask_pos, align_metric, overlaps


def make_loss(own_thr: float, stats: dict):
    class ScaleOwnedDetectionLoss(v8DetectionLoss):
        def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
            super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
            self.assigner = ScaleOwnedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, stride=self.stride.tolist(), topk2=tal_topk2)
            self.assigner.own_thr, self.assigner.stats = own_thr, stats

        def loss(self, preds, batch):
            f0 = preds["feats"][0]; self.assigner.n_small = f0.shape[2] * f0.shape[3] if float(self.stride[0]) < 8 else 0   # anchors are laid out level by level
            return super().loss(preds, batch)
    return ScaleOwnedDetectionLoss


def attach(model, own_thr: float = 32.0) -> dict:
    """model.criterion = E2ELoss(loss_fn=ScaleOwnedDetectionLoss); returns the tripwire stats dict."""
    stats = {"batches": 0, "pos_small_level": 0, "pos_small_level_on_large_gt": 0, "pos_total": 0, "own_thr": own_thr}
    model.criterion = E2ELoss(model, loss_fn=make_loss(own_thr, stats))
    return stats


def demo() -> None:
    import os, sys; sys.path.insert(0, os.getcwd())
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from adapters.yolo26n_visdrone_scratch.modules import bandgate as BG
    yaml = os.path.join("adapters", "yolo26n_visdrone_scratch", "configs", "yolo26n-p2-visdrone.yaml")
    m = YOLO(yaml).model; BG.apply_surgery(m, "band"); m.args = get_cfg(); m.train()
    torch.manual_seed(0); x = torch.rand(2, 3, 640, 640)
    # 2 images x 6 GT: three sub-cell objects (8-20 px) and three large ones (60-200 px), xywh normalised
    small = [[0.2, 0.2, 0.0125, 0.0125], [0.5, 0.5, 0.03, 0.03], [0.7, 0.3, 0.02, 0.02]]; large = [[0.3, 0.7, 0.1, 0.1], [0.8, 0.8, 0.3, 0.3], [0.5, 0.2, 0.15, 0.1]]
    boxes = torch.tensor(small + large + small + large); batch = {"batch_idx": torch.tensor([0] * 6 + [1] * 6).float(), "cls": torch.zeros(12, 1), "bboxes": boxes}
    stats = attach(m, own_thr=32.0); preds = m(x); loss, items = m.criterion(preds, batch)
    assert torch.isfinite(loss).all() and stats["batches"] == 2, stats                      # one2many + one2one assigners
    assert stats["pos_small_level_on_large_gt"] == 0, stats                                 # large GT never owned by stride 4
    assert stats["pos_small_level"] > 0, stats                                              # small GT can be owned by stride 4 (some fall in live cells)
    # synthetic ownership test (random-init predictions never favour stride 4 for large GT, so build the case directly):
    # stride-4 anchors predict the GT boxes perfectly with high scores; unrestricted they must win, restricted they must not
    n0, hw = 25600, 6400; na = n0 + hw + 1600 + 400; from ultralytics.utils.tal import make_anchors
    feats = [torch.zeros(1, 1, 160, 160), torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 40, 40), torch.zeros(1, 1, 20, 20)]
    anc, strd = make_anchors(feats, torch.tensor([4., 8., 16., 32.]), 0.5); anc = anc * strd
    gt = torch.tensor([[[100., 100., 110., 110.], [300., 300., 400., 400.]]])                    # one small (10 px), one large (100 px)
    pdb = torch.zeros(1, na, 4); pds = torch.full((1, na, 1), 1e-3)
    for g in range(2):
        inside = (anc[:, 0] > gt[0, g, 0]) & (anc[:, 0] < gt[0, g, 2]) & (anc[:, 1] > gt[0, g, 1]) & (anc[:, 1] < gt[0, g, 3])
        lvl0 = inside.clone(); lvl0[n0:] = False; pdb[0, lvl0] = gt[0, g]; pds[0, lvl0] = 0.9          # stride-4 anchors inside: perfect boxes
        rest = inside.clone(); rest[:n0] = False; pdb[0, rest] = gt[0, g] + torch.tensor([5., 5., -5., -5.]); pds[0, rest] = 0.5   # others: worse
    for thr, want_large in ((32.0, 0), (1e9, 1)):
        a2 = ScaleOwnedAssigner(topk=10, num_classes=1, alpha=0.5, beta=6.0, stride=[4, 8, 16, 32]); a2.own_thr, a2.report_thr, a2.n_small = thr, 32.0, n0; a2.stats = {"batches": 0, "pos_small_level": 0, "pos_small_level_on_large_gt": 0, "pos_total": 0}
        a2.bs, a2.n_max_boxes = 1, 2; mask_pos, _, _ = a2.get_pos_mask(pds, pdb, torch.zeros(1, 2, 1), gt, anc, torch.ones(1, 2, 1))
        on_large_l0 = int(mask_pos[0, 1, :n0].sum()); on_large_rest = int(mask_pos[0, 1, n0:].sum()); on_small_l0 = int(mask_pos[0, 0, :n0].sum())
        assert (on_large_l0 > 0) == bool(want_large), (thr, on_large_l0); assert on_small_l0 > 0 and (on_large_rest > 0 or want_large), (thr, on_large_rest, on_small_l0)   # restricted: the large GT must still get owners at stride 8+
    st = {"pos_small_level_on_large_gt": "restricted 0 / unrestricted >0 (synthetic)"}
    print(f"scaleowned demo OK: restricted -> stride-4 positives {stats['pos_small_level']} (on large GT: {stats['pos_small_level_on_large_gt']}) of {stats['pos_total']}; unrestricted control -> {st['pos_small_level_on_large_gt']} stride-4 positives on large GT")


if __name__ == "__main__":
    demo()
