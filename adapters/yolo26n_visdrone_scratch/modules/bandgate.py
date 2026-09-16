"""Band-gated stride-4 decision level (exp011 mechanism).

Source domain: sequential analysis (Wald's SPRT): a cell whose first-sample
posterior leaves the undecided band is decided; only the undecided band takes
a second sample. Here the second sample is the full stride-4 decision level
of the yolo26-p2 topology (exp001's Winner), and the band gates it:

  stage 1  the stride-8/16/32 levels — first sample, unchanged;
  band     stride-8 cells with one-to-one objectness >= B_hi are decided; of
           the rest the top-q fraction by objectness are selected;
  stage 2  the four stride-4 anchors of every selected cell stay live; every
           other stride-4 anchor is masked (logit -10, box 0) in both branches.

The decision at stride 4 is the P2 head's own (exp001 capacity); what is new
is where it is allowed to decide. Measured basis: exp010's band covered 85% of
small GT centres at q = 0.25 (random: 25%) while its thin second stage could
not decide well (ins_016).

Control (P9, designed): the same topology gated on a RANDOM q fraction of
cells (seeded) — identical params and compute, no evidence-driven selection.

Surgery is a class swap on the built 4-level Detect (no parameters added
relative to the p2 topology).
"""
from __future__ import annotations
import torch
from ultralytics.nn.modules.head import Detect

MASK_LOGIT = -10.0


class BandGatedDetect(Detect):
    # set by surgery: q, b_hi, selection ('band'|'random'), gen (torch.Generator, CPU); size_thr (exp012, px or None)
    size_thr = None   # class default: exp011 checkpoints predate the knob
    b_lo = None       # exp013: absolute lower objectness bound (a true two-sided band); None = rank-only (top-q)
    train_mask = "replace"   # exp015: "negative" = in training keep the real stride-4 outputs (masked anchors are excluded from assignment by the loss and trained as negatives); inference masks as always

    def _select(self, scores_l1: torch.Tensor, hw: int, boxes_l1: torch.Tensor = None):
        """scores_l1: (B, nc, h*w) stride-8 one2one logits; boxes_l1: (B, 4, h*w) ltrb in stride units.
        -> (B, K) selected stride-8 cell indices and a (B, K) validity mask (fewer than K eligible cells -> invalid slots)."""
        B = scores_l1.shape[0]; K = max(1, int(self.q * hw))
        if self.selection == "random":
            sel = torch.stack([torch.randperm(hw, generator=self.gen)[:K] for _ in range(B)]).to(scores_l1.device)
            return sel, torch.ones_like(sel, dtype=torch.bool)
        o = scores_l1.max(1).values.sigmoid()
        o = o.masked_fill(o >= self.b_hi, -1.0)               # decided cells take no second sample
        if self.b_lo is not None:
            o = o.masked_fill(o < self.b_lo, -1.0)            # exp013: background-level posteriors are decided-negative, not "undecided"
        if self.size_thr is not None and boxes_l1 is not None:   # exp012: a second sample only where stage 1 says the object is sub-cell
            side = torch.maximum((boxes_l1[:, 0] + boxes_l1[:, 2]).clamp(min=0), (boxes_l1[:, 1] + boxes_l1[:, 3]).clamp(min=0)) * float(self.stride[1])
            o = o.masked_fill(side >= self.size_thr, -1.0)
        val, sel = o.topk(K, dim=1)
        return sel, val > -1.0

    @staticmethod
    def _live(sel: torch.Tensor, h: int, w: int, valid: torch.Tensor = None) -> torch.Tensor:
        """(B, K) stride-8 cells -> (B, 4hw) bool mask of their four stride-4 anchors (row 2i+di, col 2j+dj); invalid slots add nothing."""
        i, j = sel // w, sel % w
        vidx = torch.stack([((2 * i + di) * (2 * w) + (2 * j + dj)) for di in (0, 1) for dj in (0, 1)], -1).reshape(sel.shape[0], -1)
        src = torch.ones_like(vidx, dtype=torch.bool) if valid is None else valid.repeat_interleave(4, dim=1)
        return torch.zeros(sel.shape[0], 4 * h * w, dtype=torch.bool, device=sel.device).scatter(1, vidx, src)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)             # {} once Detect.fuse() removed the one2many heads
        x_det = [xi.detach() for xi in x]
        one2one = self.forward_head(x_det, **self.one2one)
        H2, W2 = x[0].shape[2:]; h, w = x[1].shape[2:]; assert (H2, W2) == (2 * h, 2 * w), (H2, W2, h, w)
        n0, hw = H2 * W2, h * w
        sel, valid = self._select(one2one["scores"][..., n0:n0 + hw].detach(), hw, one2one["boxes"][..., n0:n0 + hw].detach())
        live = self._live(sel, h, w, valid).unsqueeze(1)           # (B, 1, n0)
        out = {}; self.last_live = live[:, 0].detach()             # (B, n0) for a live-mask-aware assigner (exp015)
        keep_real = self.training and self.train_mask == "negative"
        for name, pr in (("one2many", preds), ("one2one", one2one)):
            if not pr: continue
            s, b = pr["scores"], pr["boxes"]
            if keep_real: out[name] = {"boxes": b, "scores": s, "feats": pr["feats"]}; continue
            s0 = torch.where(live, s[..., :n0], torch.full_like(s[..., :n0], MASK_LOGIT))
            b0 = torch.where(live, b[..., :n0], torch.zeros_like(b[..., :n0]))
            out[name] = {"boxes": torch.cat([b0, b[..., n0:]], -1), "scores": torch.cat([s0, s[..., n0:]], -1), "feats": pr["feats"]}
        self.last_sel = sel.detach().masked_fill(~valid, -1)   # -1 = empty slot (never matches a cell index)
        if self.training:
            return out
        y = self._inference(out["one2one"])
        y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, out)


def apply_surgery(det_model, selection: str, q: float = 0.25, b_hi: float = 0.5, seed: int = 0, size_thr: float = None, b_lo: float = None, train_mask: str = "replace") -> int:
    det = det_model.model[-1]; assert type(det).__name__ == "Detect", type(det).__name__
    assert det.nl == 4 and [int(s) for s in det.stride.tolist()] == [4, 8, 16, 32], det.stride
    assert selection in ("band", "random"), selection
    assert train_mask in ("replace", "negative"), train_mask
    det.q, det.b_hi, det.selection, det.size_thr, det.b_lo, det.train_mask = q, b_hi, selection, size_thr, b_lo, train_mask
    det.gen = torch.Generator().manual_seed(seed)
    det.__class__ = BandGatedDetect
    return 0


def surgery_present(det_model) -> dict:
    det = det_model.model[-1]
    return {"class": type(det).__name__, "selection": getattr(det, "selection", None), "q": getattr(det, "q", None),
            "strides": [float(s) for s in det.stride.tolist()], "nl": det.nl, "size_thr": getattr(det, "size_thr", None), "b_lo": getattr(det, "b_lo", None), "train_mask": getattr(det, "train_mask", "replace")}


def demo() -> None:
    import os, sys; sys.path.insert(0, os.getcwd())
    from ultralytics import YOLO
    yaml = os.path.join("adapters", "yolo26n_visdrone_scratch", "configs", "yolo26n-p2-visdrone.yaml")
    m = YOLO(yaml).model; n = sum(p.numel() for p in m.parameters()); apply_surgery(m, "band"); info = surgery_present(m)
    assert info["class"] == "BandGatedDetect" and info["strides"] == [4.0, 8.0, 16.0, 32.0] and sum(p.numel() for p in m.parameters()) == n, info
    x = torch.rand(2, 3, 640, 640); m.train(); p = m(x)
    for br in ("one2many", "one2one"):
        assert p[br]["scores"].shape == (2, 10, 25600 + 6400 + 1600 + 400), p[br]["scores"].shape
        live = (p[br]["boxes"][..., :25600] != 0).any(1).float().mean().item()
        assert abs(live - 0.25) < 0.01, live                        # q of the stride-8 cells -> q of the stride-4 anchors live
    sel = m.model[-1].last_sel; lv = BandGatedDetect._live(sel, 80, 80)
    assert (p["one2one"]["boxes"][..., :25600] != 0).any(1).eq(lv).all(), "live anchors must be exactly the selected cells' four sub-anchors"
    # size-aware band (exp012): cells whose stage-1 predicted side >= size_thr never take a second sample; slots beyond the eligible count stay empty
    m3 = YOLO(yaml).model; apply_surgery(m3, "band", size_thr=32.0); m3.train(); p3 = m3(x); det3 = m3.model[-1]; s3 = det3.last_sel
    live3 = (p3["one2one"]["boxes"][..., :25600] != 0).any(1).float().mean().item(); assert live3 <= 0.25 + 1e-6, live3
    b1 = p3["one2one"]["boxes"][..., 25600:25600 + 6400]; side = torch.maximum((b1[:, 0] + b1[:, 2]).clamp(min=0), (b1[:, 1] + b1[:, 3]).clamp(min=0)) * 8
    for bi in range(2):
        chosen = s3[bi][s3[bi] >= 0]; assert (side[bi][chosen] < 32.0).all(), "a cell with a large predicted object was selected"
    m3.zero_grad(); (p3["one2many"]["scores"][..., :25600].abs().mean() + p3["one2many"]["boxes"][..., :25600].abs().mean()).backward()
    g3 = sum(float(q_.grad.norm()) for q_ in list(det3.cv2[0].parameters()) + list(det3.cv3[0].parameters()) if q_.grad is not None)
    assert live3 == 0 or g3 > 0, "gradient must reach the stride-4 heads through the live anchors"
    print(f"  size-aware band: live stride-4 anchors {live3:.3f} (q=0.25 cap), stride-4 head grad {g3:.3f}")
    # absolute band (exp013): with b_lo above every random-init posterior no cell is eligible -> all slots empty, level fully masked, forward still valid
    m4 = YOLO(yaml).model; apply_surgery(m4, "band", b_lo=0.999); m4.train(); p4 = m4(x)
    assert (m4.model[-1].last_sel < 0).all() and (p4["one2one"]["boxes"][..., :25600] == 0).all(), "b_lo must be able to empty the band"
    m4.eval()
    with torch.no_grad(): y4 = m4(x)
    y4 = y4[0] if isinstance(y4, tuple) else y4; assert y4.shape == (2, 300, 6) and torch.isfinite(y4).all()
    print("  absolute band: empty band handled (all stride-4 anchors masked, inference valid)")
    # gradient reaches the stride-4 heads only through live anchors
    m.zero_grad(); (p["one2many"]["scores"][..., :25600].abs().mean() + p["one2many"]["boxes"][..., :25600].abs().mean()).backward()
    g = sum(float(q_.grad.norm()) for q_ in list(m.model[-1].cv2[0].parameters()) + list(m.model[-1].cv3[0].parameters()) if q_.grad is not None); assert g > 0, g
    m.eval()
    with torch.no_grad(): y = m(x)
    y = y[0] if isinstance(y, tuple) else y; assert y.shape == (2, 300, 6), y.shape
    m.fuse()
    with torch.no_grad(): yf = m(x)
    yf = yf[0] if isinstance(yf, tuple) else yf; assert yf.shape == (2, 300, 6) and torch.isfinite(yf).all()
    m2 = YOLO(yaml).model; apply_surgery(m2, "random"); m2.train(); m2(x)
    assert not torch.equal(m.model[-1].last_sel, m2.model[-1].last_sel)
    print(f"bandgate demo OK: p2 topology {n} params (+0 by surgery), live stride-4 anchors {live:.3f}, stride-4 head grad {g:.3f}")


if __name__ == "__main__":
    demo()
