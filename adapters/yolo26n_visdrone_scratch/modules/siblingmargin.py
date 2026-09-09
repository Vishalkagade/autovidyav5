"""Sibling-pair margin objective (exp007 mechanism) — coding-theory framing.

Source domain: error-correcting output codes (Dietterich & Bakiri 1995).
The baseline's class errors are concentrated on visually-similar SIBLING
pairs (exp000: van->car 58% of located vans, bicycle->motor 57%,
people->pedestrian 25%, awning-tricycle 75% wrong, truck<->bus). In ECOC
terms, those pairs sit at Hamming distance 1 in the one-hot code. This
objective enforces a margin on the ONE bit that separates a sibling pair:

    L_sib = mean over positives whose class c has a sibling s of
            softplus(margin - (z_c - z_s))            (logit margin)

added to the cls loss with gain lambda. Zero parameters, no architecture
change; the head's existing logits carry the code.

Control (P9, designed for this experiment): the SAME margin loss applied to
a fixed random non-sibling pairing of the classes (same loss form and mass,
no sibling structure).

Tripwire: applied_batches, running mean of the margin term.
"""
from __future__ import annotations
import torch, torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.utils.tal import make_anchors

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
SIBLINGS = [("pedestrian", "people"), ("bicycle", "motor"), ("car", "van"), ("truck", "bus"), ("tricycle", "awning-tricycle")]
# fixed random non-sibling pairing for the control (seeded once, recorded here so it is auditable)
CONTROL_PAIRS = [("pedestrian", "truck"), ("people", "van"), ("bicycle", "bus"), ("car", "awning-tricycle"), ("tricycle", "motor")]


def pair_table(pairs) -> torch.Tensor:
    t = torch.full((len(NAMES),), -1, dtype=torch.long)
    for a, b in pairs:
        ia, ib = NAMES.index(a), NAMES.index(b); t[ia] = ib; t[ib] = ia
    return t


def make_loss(variant: str, margin: float, lam: float):
    table = pair_table(SIBLINGS if variant == "mech" else CONTROL_PAIRS)
    stats = {"applied_batches": 0, "sum_term": 0.0, "n_pos_paired": 0, "variant": variant, "margin": margin, "lambda": lam}

    class SiblingMarginLoss(v8DetectionLoss):
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
            target_scores_sum = max(target_scores.sum(), 1)
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
            # ── sibling margin on positives ───────────────────────────────
            if fg_mask.any():
                z = pred_scores[fg_mask].float()                                  # (P, nc) logits
                c = target_scores[fg_mask].argmax(-1)                             # assigned class per positive
                sib = table.to(self.device)[c]; has = sib >= 0
                if has.any():
                    zc = z[has].gather(1, c[has].unsqueeze(1)).squeeze(1); zs = z[has].gather(1, sib[has].unsqueeze(1)).squeeze(1)
                    term = F.softplus(margin - (zc - zs)).mean()
                    loss[1] = loss[1] + lam * term
                    stats["sum_term"] += float(term); stats["n_pos_paired"] += int(has.sum())
            stats["applied_batches"] += 1
            if fg_mask.sum():
                loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                                  target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
            loss[0] *= self.hyp.box; loss[1] *= self.hyp.cls; loss[2] *= self.hyp.dfl
            return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()

    return SiblingMarginLoss, stats


def attach(model, variant: str, margin: float = 2.0, lam: float = 0.5) -> dict:
    Cls, stats = make_loss(variant, margin, lam)
    model.criterion = E2ELoss(model, loss_fn=Cls)
    return stats
