"""Anchor (exp011) behaviour across datasets: stage-1 objectness, decided/band composition, where GT falls, and what the live level emits.
Runs on trained checkpoints, 300 eval images per dataset (seed 0), light enough to share the GPU with a training job."""
import json, os, random, sys, yaml, cv2, torch, numpy as np
sys.path.insert(0, os.getcwd())
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from adapters.yolo26n_visdrone_scratch.modules import bandgate as BG
SETS = {"visdrone": ("runs_visdrone/exp011_mech_S2_seed42/weights/last.pt", "visdrone_eval.txt", "runs_visdrone/exp000_S2_seed42/weights/last.pt"),
        "voc": ("runs_voc/voc_bandgate_S2_seed42/weights/last.pt", "trajectory/scratch/confirm_voc/voc_eval.txt", "runs_voc/voc_baseline_S2_seed42/weights/last.pt"),
        "sku": ("runs_sku/sku_bandgate_S2_seed42/weights/last.pt", "trajectory/scratch/confirm_sku/sku_eval.txt", "runs_sku/sku_baseline_S2_seed42/weights/last.pt")}
lb = LetterBox((640, 640), auto=False); out = {}
def gts(p):
    lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    return [list(map(float, l.split()[:5])) for l in open(lp)] if os.path.exists(lp) else []
for name, (w, lst, wb) in SETS.items():
    imgs = sorted(l.strip() for l in open(lst) if l.strip()); random.Random(0).shuffle(imgs); imgs = imgs[:300]
    m = YOLO(w).model.cuda().eval().float(); det = m.model[-1]; assert type(det).__name__ == "BandGatedDetect"
    orig = det._select; det._select = lambda s, hw, _o=orig: (setattr(det, "last_s1", s.detach()), _o(s, hw))[1]
    dec, band_obj, obj_all = [], [], []; gt_loc = {k: {"decided": 0, "band": 0, "other": 0} for k in ("small", "medium", "large")}; live_hits = []; l0_frac = []
    for p in imgs:
        im0 = cv2.imread(p)
        if im0 is None: continue
        h0, w0 = im0.shape[:2]; r = min(640 / h0, 640 / w0); px, py = (640 - round(w0 * r)) / 2, (640 - round(h0 * r)) / 2
        x = torch.from_numpy(lb(image=im0)[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
        with torch.no_grad(): y, raw = m(x)
        o = det.last_s1[0].max(0).values.sigmoid().cpu().numpy(); sel = set(det.last_sel[0].tolist()); decided = o >= det.b_hi
        dec.append(float(decided.mean())); obj_all.append(np.quantile(o, [0.5, 0.9, 0.99]).tolist()); band_obj.append(float(np.mean([o[i] for i in sel])))
        for c, xc, yc, bw, bh in gts(p):
            side = max(bw * w0, bh * h0) * r; k = "small" if side < 32 else ("medium" if side < 96 else "large"); X, Y = xc * w0 * r + px, yc * h0 * r + py; cell = int(Y // 8) * 80 + int(X // 8)
            gt_loc[k]["decided" if decided[cell] else ("band" if cell in sel else "other")] += 1
        sc = raw["one2one"]["scores"][0].sigmoid(); n0 = 25600; conf = sc.max(0).values; l0 = (conf[:n0] > 0.25).sum().item(); rest = (conf[n0:] > 0.25).sum().item()
        l0_frac.append(l0 / max(1, l0 + rest))
    out[name] = {"n_images": len(dec), "decided_cell_frac_mean": float(np.mean(dec)), "objectness_quantiles_p50_p90_p99": np.mean(obj_all, 0).tolist(), "band_mean_objectness": float(np.mean(band_obj)),
                 "gt_by_cell_state": {k: {s: v / max(1, sum(d.values())) for s, v in d.items()} | {"n": sum(d.values())} for k, d in gt_loc.items()},
                 "frac_detections_conf25_from_stride4": float(np.mean(l0_frac))}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open("trajectory/scratch/phase2/anchor_analysis.json", "w"), indent=1); print("ANALYSIS_DONE")
