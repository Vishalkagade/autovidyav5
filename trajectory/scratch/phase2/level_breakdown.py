"""Per-level detection breakdown on SKU-110K (300 eval images): for each checkpoint, TP / duplicate / FP counts of
conf>=.25 detections split by anchor level (stride 4 vs 8+), plus the level-off mAP50 proxy (per-image F1 with stride-4 dropped)."""
import json, os, sys, random, cv2, torch, numpy as np; sys.path.insert(0, os.getcwd())
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from adapters.yolo26n_visdrone_scratch.adapter import _iou, image_f1
import adapters.yolo26n_visdrone_scratch.modules.bandgate as BG
imgs = sorted(l.strip() for l in open('trajectory/scratch/confirm_sku/sku_eval.txt') if l.strip()); random.Random(0).shuffle(imgs); imgs = imgs[:300]; lb = LetterBox((640, 640), auto=False)
def gts(p, r, px, py):
    lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"; w0, h0 = None, None; out = []
    im = cv2.imread(p); h0, w0 = im.shape[:2]
    for l in open(lp):
        c, xc, yc, bw, bh = map(float, l.split()[:5]); x1, y1, x2, y2 = (xc - bw / 2) * w0 * r + px, (yc - bh / 2) * h0 * r + py, (xc + bw / 2) * w0 * r + px, (yc + bh / 2) * h0 * r + py; out.append([0, x1, y1, x2, y2])
    return out
def match(preds, gt):
    """greedy by confidence: returns (tp, dup, fp)."""
    preds = sorted(preds, key=lambda t: -t[1]); matched = set(); tp = dup = fp = 0
    for pc, cf, *pb in preds:
        best, bj = 0.0, -1
        for j, (gc, *gb) in enumerate(gt):
            if j in matched: continue
            i = _iou(pb, gb)
            if i > best: best, bj = i, j
        if best >= 0.5: matched.add(bj); tp += 1
        elif any(_iou(pb, gb) >= 0.5 for j, (gc, *gb) in enumerate(gt) if j in matched): dup += 1
        else: fp += 1
    return tp, dup, fp
import sys as _s; CK = {"negband": "runs_sku/sku_negband_S2_seed42/weights/last.pt"} if "--only-negband" in _s.argv else {"baseline": "runs_sku/sku_baseline_S2_seed42/weights/last.pt", "ungated": "runs_sku/sku_mech_S2_seed42/weights/last.pt", "anchor": "runs_sku/sku_bandgate_S2_seed42/weights/last.pt", "absband": "runs_sku/sku_absband_S2_seed42/weights/last.pt", "ownband": "runs_sku/sku_ownband_S2_seed42/weights/last.pt"}
out = {}
for name, w in CK.items():
    m = YOLO(w).model.cuda().eval().float(); det = m.model[-1]; four = det.nl == 4; n0 = 25600 if four else 0
    agg = {"l4": [0, 0, 0], "l8": [0, 0, 0], "gt": 0, "f1_all": [], "f1_no_l4": [], "conf_l4": [], "conf_l8": []}
    for p in imgs:
        im0 = cv2.imread(p)
        if im0 is None: continue
        h0, w0 = im0.shape[:2]; r = min(640 / h0, 640 / w0); px, py = (640 - round(w0 * r)) / 2, (640 - round(h0 * r)) / 2
        x = torch.from_numpy(lb(image=im0)[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
        with torch.no_grad(): y, raw = m(x)
        o = raw["one2one"]; sc = o["scores"][0].sigmoid().max(0).values; boxes = det._get_decode_boxes(o)[0].T   # (na, 4) xyxy px
        keep = sc > 0.25; idx = torch.nonzero(keep).flatten(); gt = gts(p, r, px, py); agg["gt"] += len(gt)
        preds_l4 = [(0, float(sc[i]), *boxes[i].tolist()) for i in idx if i < n0]; preds_l8 = [(0, float(sc[i]), *boxes[i].tolist()) for i in idx if i >= n0]
        # matched jointly (as the metric sees it): stride-8 first then stride-4 appended -> credit by greedy confidence order
        allp = [(0, c, *bb, "l4") for (_, c, *bb) in preds_l4] + [(0, c, *bb, "l8") for (_, c, *bb) in preds_l8]
        allp.sort(key=lambda t: -t[1]); matched = set()
        for _, cf, x1, y1, x2, y2, lv in allp:
            best, bj = 0.0, -1
            for j, (gc, *gb) in enumerate(gt):
                if j in matched: continue
                i = _iou([x1, y1, x2, y2], gb)
                if i > best: best, bj = i, j
            if best >= 0.5: matched.add(bj); agg[lv][0] += 1
            elif any(_iou([x1, y1, x2, y2], gb) >= 0.5 for j, (gc, *gb) in enumerate(gt) if j in matched): agg[lv][1] += 1
            else: agg[lv][2] += 1
        agg["f1_all"].append(image_f1([(0, c, *bb) for (_, c, *bb) in preds_l4 + preds_l8], gt)); agg["f1_no_l4"].append(image_f1(preds_l8, gt))
        agg["conf_l4"] += [c for (_, c, *bb) in preds_l4]; agg["conf_l8"] += [c for (_, c, *bb) in preds_l8]
    G = agg["gt"]; out[name] = {"gt": G, "stride4": {"tp": agg["l4"][0], "dup": agg["l4"][1], "fp": agg["l4"][2], "mean_conf": float(np.mean(agg["conf_l4"])) if agg["conf_l4"] else None}, "stride8plus": {"tp": agg["l8"][0], "dup": agg["l8"][1], "fp": agg["l8"][2], "mean_conf": float(np.mean(agg["conf_l8"]))},
                 "recall_all": (agg["l4"][0] + agg["l8"][0]) / G, "f1_mean": float(np.mean(agg["f1_all"])), "f1_mean_no_stride4": float(np.mean(agg["f1_no_l4"]))}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open("trajectory/scratch/phase2/level_breakdown_sku_seed42" + ("_negband" if "--only-negband" in _s.argv else "") + ".json", "w"), indent=1); print("BREAKDOWN_DONE")
