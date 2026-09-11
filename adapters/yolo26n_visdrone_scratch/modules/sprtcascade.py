"""SPRT cascade: sequential evidence accumulation to stride 4 (exp008 mechanism).

Source domain: sequential analysis — Wald's sequential probability ratio
test (1945): accumulate log-likelihood evidence and decide only when it
leaves the undecided band; otherwise take another sample. Applied per
stride-8 cell of the detection head:

  stage 1  the unchanged YOLO26 head (strides 8/16/32) — first sample;
  band     cells whose stage-1 one-to-one objectness is >= B_hi are decided;
           of the rest, the top q fraction by objectness (the undecided
           band, budgeted) request a second sample;
  stage 2  the stride-4 P2 feature in a 4x4 window around the cell plus the
           cell's P3 feature -> small conv trunk -> four sub-cell decisions;
           the sub-cell class logit is stage-1 logit + stage-2 residual
           (accumulation), the box is predicted fresh at the sub-anchor.

Sub-cell predictions occupy a virtual stride-4 anchor level (2h x 2w), so
the stock assigner/decoder see an ordinary 4-level head [4, 8, 16, 32];
unselected sub-anchors are masked (logit -10, box 0).

Control (P9, designed): the SAME stage-2 network on a RANDOM q fraction of
cells — same params, same compute, no evidence-driven selection.

Surgery is in place on the built model: Detect gets the P2 feature as a
4th input (det.f += [2], model.save += [2]) and is class-swapped.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Detect

MASK_LOGIT = -10.0


class Stage2(nn.Module):
    """Shared trunk (4x4 window -> 2x2) + per-branch heads (one2many, one2one)."""
    def __init__(self, c_p2: int, c_p3: int, nc: int, hidden: int = 96, c_extra: int = 0):
        super().__init__()
        self.nc = nc
        self.trunk = nn.Sequential(nn.Conv2d(c_p2 + c_p3 + c_extra, hidden, 3), nn.SiLU(), nn.Conv2d(hidden, hidden, 1), nn.SiLU())   # valid 3x3: 4x4 -> 2x2
        self.box = nn.ModuleList(nn.Conv2d(hidden, 4, 1) for _ in range(2))
        self.cls = nn.ModuleList(nn.Conv2d(hidden, nc, 1) for _ in range(2))
        for b in self.box: nn.init.constant_(b.bias, 1.0)
        for c in self.cls: nn.init.zeros_(c.bias)

    def forward(self, win: torch.Tensor, branch: int):
        h = self.trunk(win)                                   # (N, hidden, 2, 2)
        return self.box[branch](h), self.cls[branch](h)       # (N, 4, 2, 2), (N, nc, 2, 2)


class SPRTDetect(Detect):
    # set by surgery: sprt (Stage2), q, b_hi, selection ('sprt'|'random'), gen (torch.Generator, CPU)
    def _select(self, scores_l0: torch.Tensor, hw: int) -> torch.Tensor:
        """scores_l0: (B, nc, h*w) stage-1 one2one logits -> (B, K) selected cell indices."""
        B = scores_l0.shape[0]; K = max(1, int(self.q * hw))
        if self.selection == "random":
            return torch.stack([torch.randperm(hw, generator=self.gen)[:K] for _ in range(B)]).to(scores_l0.device)
        o = scores_l0.max(1).values.sigmoid()                 # (B, hw) objectness
        o = o.masked_fill(o >= self.b_hi, -1.0)               # decided cells never re-sampled
        return o.topk(K, dim=1).indices

    def _stage2_level(self, p2, p3, scores_l0, sel, branch):
        """Build the virtual stride-4 level: boxes (B,4,4hw), scores (B,nc,4hw)."""
        B, C2, H2, W2 = p2.shape; h, w = p3.shape[2:]; K = sel.shape[1]; nc = self.nc
        win = F.unfold(p2, kernel_size=4, stride=2, padding=1)                 # (B, C2*16, h*w)
        win = win.gather(2, sel.unsqueeze(1).expand(-1, win.shape[1], -1))    # (B, C2*16, K)
        win = win.transpose(1, 2).reshape(B * K, C2, 4, 4)
        cell = p3.flatten(2).gather(2, sel.unsqueeze(1).expand(-1, p3.shape[1], -1))   # (B, C3, K)
        cell = cell.transpose(1, 2).reshape(B * K, -1, 1, 1).expand(-1, -1, 4, 4)
        z1 = scores_l0.gather(2, sel.unsqueeze(1).expand(-1, nc, -1))         # (B, nc, K) stage-1 logit
        if self.accumulate == "feature":   # exp010: stage 1 enters as a (detached) feature; the sub-cell logit is stage 2's own decision
            feat = [win, cell, z1.detach().transpose(1, 2).reshape(B * K, nc, 1, 1).expand(-1, -1, 4, 4)]
        else:
            feat = [win, cell]
        box, dcls = self.sprt(torch.cat(feat, 1), branch)                     # (BK,4,2,2), (BK,nc,2,2)
        cls = dcls.view(B, K, nc, 2, 2)
        if self.accumulate == "add":       # exp008/exp009: additive accumulation (parent logit + residual)
            cls = cls + z1.transpose(1, 2).reshape(B, K, nc, 1, 1)
        box = box.view(B, K, 4, 2, 2)
        # scatter into the virtual grid (2h x 2w): sub-cell (di,dj) of cell (i,j) -> row 2i+di, col 2j+dj
        i, j = sel // w, sel % w                                              # (B, K)
        vidx = torch.stack([((2 * i + di) * (2 * w) + (2 * j + dj)) for di in (0, 1) for dj in (0, 1)], -1)   # (B, K, 4) order: (0,0),(0,1),(1,0),(1,1)
        vidx = vidx.reshape(B, K * 4)
        S = torch.full((B, nc, 4 * h * w), MASK_LOGIT, device=p2.device, dtype=cls.dtype)
        Bx = torch.zeros((B, 4, 4 * h * w), device=p2.device, dtype=box.dtype)
        S = S.scatter(2, vidx.unsqueeze(1).expand(-1, nc, -1), cls.permute(0, 2, 1, 3, 4).reshape(B, nc, K * 4))
        Bx = Bx.scatter(2, vidx.unsqueeze(1).expand(-1, 4, -1), box.permute(0, 2, 1, 3, 4).reshape(B, 4, K * 4))
        return Bx, S

    def forward(self, x):
        p2 = x[3]; x = x[:3]
        preds = self.forward_head(x, **self.one2many)
        x_det = [xi.detach() for xi in x]
        one2one = self.forward_head(x_det, **self.one2one)
        h, w = x[0].shape[2:]; hw = h * w
        sel = self._select(one2one["scores"][..., :hw].detach(), hw)
        out = {}
        branches = [("one2one", one2one, p2.detach(), x_det[0], 1)]
        if preds:                                                    # one2many is dropped by Detect.fuse() at inference
            branches.insert(0, ("one2many", preds, p2, x[0], 0))
        for name, pr, pp2, pp3, br in branches:
            bx, sc = self._stage2_level(pp2, pp3, pr["scores"][..., :hw], sel, br)
            s1, b1 = pr["scores"], pr["boxes"]
            if self.parent == "mask":   # exp009 (P4 retry of exp008): a cell that takes a second sample emits no stage-1 decision
                s1 = s1.scatter(2, sel.unsqueeze(1).expand(-1, self.nc, -1), MASK_LOGIT)
                b1 = b1.scatter(2, sel.unsqueeze(1).expand(-1, 4, -1), 0.0)
            out[name] = {"boxes": torch.cat([bx, b1], -1), "scores": torch.cat([sc, s1], -1),
                         "feats": [p2.new_zeros(p2.shape[0], 1, 2 * h, 2 * w)] + list(pr["feats"])}
        self.last_sel = sel.detach()
        if self.training:
            return out
        y = self._inference(out["one2one"])
        y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, out)


def apply_surgery(det_model, selection: str, q: float = 0.25, b_hi: float = 0.5, seed: int = 0, parent: str = "keep", accumulate: str = "add") -> int:
    det = det_model.model[-1]; assert type(det).__name__ == "Detect", type(det).__name__
    before = sum(p.numel() for p in det.parameters())
    c_p2 = det_model.model[2].cv2.conv.out_channels if hasattr(det_model.model[2], "cv2") else 64
    c_p3 = det.cv3[0][0][0].conv.in_channels
    det.f = list(det.f) + [2]; det_model.save = sorted(set(det_model.save) | {2})
    assert accumulate in ("add", "feature") and parent in ("keep", "mask"), (accumulate, parent)
    det.sprt = Stage2(c_p2, c_p3, det.nc, c_extra=det.nc if accumulate == "feature" else 0)
    det.q, det.b_hi, det.selection, det.parent, det.accumulate = q, b_hi, selection, parent, accumulate
    det.gen = torch.Generator().manual_seed(seed)
    det.__class__ = SPRTDetect
    det.stride = torch.tensor([det.stride[0].item() / 2, *det.stride.tolist()], dtype=det.stride.dtype)   # [4, 8, 16, 32]
    det.shape = None
    return sum(p.numel() for p in det.parameters()) - before


def surgery_present(det_model) -> dict:
    det = det_model.model[-1]
    return {"class": type(det).__name__, "selection": getattr(det, "selection", None), "q": getattr(det, "q", None),
            "strides": [float(s) for s in det.stride.tolist()], "f": list(det.f), "save_has_2": 2 in det_model.save, "parent": getattr(det, "parent", "keep"), "accumulate": getattr(det, "accumulate", "add")}


def demo() -> None:
    import sys, os; sys.path.insert(0, os.getcwd())
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    m = YOLO(A.MODEL_YAML).model; base = sum(p.numel() for p in m.parameters())
    added = apply_surgery(m, "sprt"); info = surgery_present(m)
    assert info["class"] == "SPRTDetect" and info["strides"] == [4.0, 8.0, 16.0, 32.0] and info["save_has_2"], info
    x = torch.rand(2, 3, 640, 640); m.train(); p = m(x)
    for br in ("one2many", "one2one"):
        assert p[br]["scores"].shape == (2, 10, 4 * 6400 + 6400 + 1600 + 400), p[br]["scores"].shape
        live = (p[br]["boxes"][..., :25600] != 0).any(1).float().mean().item()   # masked sub-anchors have box == 0 exactly
        assert abs(live - 0.25) < 0.01, live               # exactly q of the virtual anchors are live
    m.eval()
    with torch.no_grad(): y = m(x)
    y = y[0] if isinstance(y, tuple) else y; assert y.shape == (2, 300, 6), y.shape
    m.fuse()                                                        # inference path: one2many heads removed
    with torch.no_grad(): yf = m(x)
    yf = yf[0] if isinstance(yf, tuple) else yf; assert yf.shape == (2, 300, 6) and torch.isfinite(yf).all(), "fused path broken"   # BN folding + top-k reorder: shape/finiteness only
    m = YOLO(A.MODEL_YAML).model; apply_surgery(m, "sprt")
    # grads reach stage 2
    m.train(); out = m(x); loss = out["one2one"]["scores"].abs().mean() + out["one2many"]["scores"].abs().mean() + out["one2many"]["boxes"].abs().mean()
    m.zero_grad(); loss.backward(); g = sum(float(q_.grad.norm()) for q_ in m.model[-1].sprt.parameters() if q_.grad is not None); assert g > 0
    # random control has the same param count and different selection
    m2 = YOLO(A.MODEL_YAML).model; added2 = apply_surgery(m2, "random"); m2.train(); m2(x)
    assert added2 == added and not torch.equal(m.model[-1].last_sel, m2.model[-1].last_sel)
    # parent="mask": the stride-8 parent of every selected cell emits nothing (logit -10, box 0); unselected parents untouched
    m3 = YOLO(A.MODEL_YAML).model; added3 = apply_surgery(m3, "sprt", parent="mask"); m3.train(); p3 = m3(x); sel = m3.model[-1].last_sel
    for br in ("one2many", "one2one"):
        par_s = p3[br]["scores"][..., 25600:25600 + 6400]; par_b = p3[br]["boxes"][..., 25600:25600 + 6400]
        gi = sel.unsqueeze(1); assert (par_s.gather(2, gi.expand(-1, 10, -1)) == MASK_LOGIT).all() and (par_b.gather(2, gi.expand(-1, 4, -1)) == 0).all()
        assert ((par_s != MASK_LOGIT).all(1).float().mean() - 0.75).abs() < 0.01, "unselected parents must stay live"
    assert added3 == added
    # accumulate="feature": stage-2 logits are its own (no additive parent term); z1 is an input, gradient must NOT reach stage 1 through it
    m4 = YOLO(A.MODEL_YAML).model; added4 = apply_surgery(m4, "sprt", accumulate="feature"); m4.train(); p4 = m4(x)
    assert added4 == added + 10 * 9 * 96, added4                              # +nc input channels on the shared 3x3 trunk conv
    sub = p4["one2one"]["scores"][..., :25600]; lv = (p4["one2one"]["boxes"][..., :25600] != 0).any(1)
    assert sub[lv.unsqueeze(1).expand(-1, 10, -1)].abs().max() < 5, "stage-2 own logits start near zero (no parent term)"
    m4.zero_grad(); sub[lv.unsqueeze(1).expand(-1, 10, -1)].sum().backward()
    o2o_cls_grad = sum(float(q_.grad.abs().sum()) for q_ in m4.model[-1].one2one_cv3.parameters() if q_.grad is not None) if hasattr(m4.model[-1], "one2one_cv3") else 0.0
    assert o2o_cls_grad == 0.0, "sub-cell loss must not reach the stage-1 class head via the detached z1 feature"
    print(f"sprtcascade demo OK: +{added} params ({added / base:.2%}), live virtual anchors {live:.3f}, stage-2 grad {g:.3f}")


if __name__ == "__main__":
    demo()
