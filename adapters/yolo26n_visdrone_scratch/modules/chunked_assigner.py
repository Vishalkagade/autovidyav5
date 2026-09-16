"""Batch-chunked TaskAlignedAssigner: on CUDA OOM, split the batch on the GPU instead of falling back to CPU.

The assignment is independent per image (every tensor is indexed by batch), so chunking is numerically identical
to the full call. Needed for dense sets with a 4-level head (SKU-110K: ~141 boxes per image x 34,000 anchors x
batch 32 OOMs on 24 GB on EVERY batch; the stock CPU fallback then costs ~95 min per epoch instead of ~1.5).
"""
from __future__ import annotations
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.tal import TaskAlignedAssigner

_orig_forward = TaskAlignedAssigner.forward


def _chunked(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, lo, hi):
    """Assign images lo..hi-1 on the GPU, halving on OOM; single images that still OOM take the stock CPU path."""
    try:
        self.bs, self.n_max_boxes = hi - lo, gt_bboxes.shape[1]; self._chunk = (lo, hi)   # assigners that read per-image side inputs (negmask) slice by this
        out = self._forward(pd_scores[lo:hi], pd_bboxes[lo:hi], anc_points, gt_labels[lo:hi], gt_bboxes[lo:hi], mask_gt[lo:hi])
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        if hi - lo == 1:
            return _orig_forward(self, pd_scores[lo:hi], pd_bboxes[lo:hi], anc_points, gt_labels[lo:hi], gt_bboxes[lo:hi], mask_gt[lo:hi])
        mid = (lo + hi) // 2
        a = _chunked(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, lo, mid)
        b = _chunked(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, mid, hi)
        return tuple(torch.cat([x, y], 0) for x, y in zip(a, b))   # target_gt_idx is per-image (0..n_max_boxes-1): no offset
    return out


def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    if gt_bboxes.shape[1] == 0 or not pd_scores.is_cuda:
        return _orig_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    return _chunked(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, 0, pd_scores.shape[0])


def apply() -> None:
    if TaskAlignedAssigner.forward is not forward:
        TaskAlignedAssigner.forward = forward; LOGGER.info("TaskAlignedAssigner: batch-chunked OOM path enabled")


def demo() -> None:
    torch.manual_seed(0); bs, na, nc, ng = 4, 300, 3, 7
    ta = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
    ps = torch.rand(bs, na, nc); pb = torch.rand(bs, na, 4) * 100; pb[..., 2:] += pb[..., :2]
    ap = torch.rand(na, 2) * 100; gl = torch.randint(0, nc, (bs, ng, 1)); gb = torch.rand(bs, ng, 4) * 100; gb[..., 2:] += gb[..., :2]
    mg = (torch.rand(bs, ng, 1) > 0.3).float()
    full = _orig_forward(ta, ps, pb, ap, gl, gb, mg)
    # force the split path by chunking manually (no CUDA needed): chunks of 1 image concatenated must equal the full call
    parts = [_chunked(ta, ps, pb, ap, gl, gb, mg, i, i + 1) for i in range(bs)]
    cat = tuple(torch.cat([p[k] for p in parts], 0) for k in range(5))
    for k, (x, y) in enumerate(zip(full, cat)): assert torch.equal(x, y), f"output {k} differs under chunking"
    print("chunked_assigner demo OK: per-image chunks reproduce the full assignment exactly")


if __name__ == "__main__":
    demo()
