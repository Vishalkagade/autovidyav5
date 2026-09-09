"""Multi-hypothesis decision at P3 (exp005 mechanism).

Source domain: target tracking / statistics — multiple-hypothesis tracking
(Reid 1979) and mixture-density outputs (Bishop 1994): a resolution cell that
may hold several targets emits K hypotheses, each with its own position and
class; assignment resolves them one-to-one.

Implementation: the P3 (stride-8) box and cls output convs of BOTH heads
(one2many, one2one) emit K=4 hypotheses per cell; a pixel-shuffle lays them
out on a virtual stride-4 grid (2h x 2w), so every hypothesis owns a sub-cell
anchor position and the unmodified assigner / decoder treat them as ordinary
anchors. Evidence (features) stays at stride 8. Detect.stride becomes
[4, 16, 32]. Params added: box conv 16->16 (was 16->4) and cls conv 64->40
(was 64->10), x2 heads = +4,308 on the nc=10 build.

Surgery is IN PLACE on the built Detect module (class swap to the subclass),
so every attribute the trainer/loss reads (nc, reg_max, no, stride, end2end,
one2one_cv2/cv3, max_det) is preserved. Mandatory SurgeryTrainer pattern.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Detect

K_SIDE = 2                     # hypotheses per cell = K_SIDE**2 = 4


class MultiHypDetect(Detect):
    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes, scores = [], []
        for i in range(self.nl):
            b, s = box_head[i](x[i]), cls_head[i](x[i])
            if i == 0:                                   # K hypotheses -> virtual stride-4 grid
                b, s = F.pixel_shuffle(b, K_SIDE), F.pixel_shuffle(s, K_SIDE)
            boxes.append(b.view(bs, 4 * self.reg_max, -1)); scores.append(s.view(bs, self.nc, -1))
        h, w = x[0].shape[2:]
        feats = [x[0].new_zeros(bs, 1, h * K_SIDE, w * K_SIDE)] + list(x[1:])   # shapes drive anchors/imgsz
        return dict(boxes=torch.cat(boxes, -1), scores=torch.cat(scores, -1), feats=feats)


def apply_surgery(det_model) -> int:
    det = det_model.model[-1]
    assert type(det).__name__ == "Detect", f"expected a plain Detect at model[-1], got {type(det).__name__}"
    before = sum(p.numel() for p in det.parameters()); K = K_SIDE ** 2
    for heads in ((det.cv2, det.cv3), (det.one2one_cv2, det.one2one_cv3)):
        box, cls = heads[0][0], heads[1][0]
        ob, oc = box[-1], cls[-1]
        nb = nn.Conv2d(ob.in_channels, ob.out_channels * K, 1); nc_ = nn.Conv2d(oc.in_channels, oc.out_channels * K, 1)
        nb.bias.data[:] = 2.0                                                   # as Detect.bias_init (box)
        nc_.bias.data[:] = math.log(5 / det.nc / (640 / (det.stride[0].item() / K_SIDE)) ** 2)   # cls prior at the virtual stride
        box[-1], cls[-1] = nb, nc_
    det.__class__ = MultiHypDetect
    det.stride = torch.tensor([det.stride[0].item() / K_SIDE, *det.stride[1:].tolist()], dtype=det.stride.dtype)
    det.shape = None                                                            # invalidate cached anchors
    return sum(p.numel() for p in det.parameters()) - before


def surgery_present(det_model) -> dict:
    det = det_model.model[-1]
    return {"class": type(det).__name__, "strides": [float(s) for s in det.stride.tolist()],
            "p3_cls_out": det.cv3[0][-1].out_channels, "p3_o2o_cls_out": det.one2one_cv3[0][-1].out_channels}


def demo() -> None:
    import sys, os
    sys.path.insert(0, os.getcwd())
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    m = YOLO(A.MODEL_YAML).model; base = sum(p.numel() for p in m.parameters())
    added = apply_surgery(m); info = surgery_present(m)
    assert info["class"] == "MultiHypDetect" and info["strides"] == [4.0, 16.0, 32.0] and info["p3_cls_out"] == 40, info
    x = torch.rand(2, 3, 640, 640)
    m.train(); p = m(x)["one2one"]
    assert p["scores"].shape == (2, 10, 4 * 80 * 80 + 40 * 40 + 20 * 20), p["scores"].shape
    assert p["boxes"].shape[-1] == p["scores"].shape[-1] and p["feats"][0].shape[2:] == (160, 160)
    m.eval()
    with torch.no_grad(): y = m(x)
    y = y[0] if isinstance(y, tuple) else y
    assert y.shape == (2, 300, 6), y.shape
    print(f"mhdetect demo OK: +{added} params ({added / base:.2%}), anchors {p['scores'].shape[-1]}")


if __name__ == "__main__":
    demo()
