"""exp011 driver — band-gated stride-4 decision level (yolo26-p2 topology; site_wrap class swap at head.detect).
Mech = SPRT band gating (q=0.25, B_hi=0.5) of the P2 level; control = random gating, same topology and compute.
SURGERY TRAINER pattern (mandatory): DetectionTrainer.get_model re-applies the class swap on the trainer-built model;
a callback asserts it on trainer.model and trainer.ema.ema before epoch 1. CLI as exp008-010 (stage/units/screen/gradcheck/dryrun/tinyprobe).
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import torch
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, PROJECT); os.chdir(PROJECT)
STATE = os.path.join(PROJECT, "trajectory", "scratch", "exp011", "state"); os.makedirs(STATE, exist_ok=True)
B0 = os.path.join(PROJECT, "trajectory", "scratch", "exp000", "state")
PREREG = json.load(open(os.path.join(PROJECT, "trajectory", "scratch", "exp011", "prereg.json")))
BASE_S2 = {s: json.load(open(f"{B0}/S2_seed{s}.json")) for s in (42, 123, 7, 1000, 2000) if os.path.exists(f"{B0}/S2_seed{s}.json")}

from torch import nn
from adapters.yolo26n_visdrone_scratch.modules import bandgate as BG
P2_YAML = os.path.join(PROJECT, "adapters", "yolo26n_visdrone_scratch", "configs", "yolo26n-p2-visdrone.yaml")
SEED = [0]
Q, BHI = PREREG["mechanism"]["q"], PREREG["mechanism"]["b_hi"]

def _w(name, obj):
    p = os.path.join(STATE, name); json.dump(obj, open(p, "w"), indent=2); print(f"[driver] wrote {p}")

# ── surgery ───────────────────────────────────────────────────────────────
def apply_surgery(det_model, variant: str, seed: int = 0) -> int:
    return BG.apply_surgery(det_model, "band" if variant == "mech" else "random", q=Q, b_hi=BHI, seed=seed)

def surgery_classes(det_model) -> list[str]:
    i = BG.surgery_present(det_model); return [i["class"], str(i["selection"]), str(i["strides"]), str(i["nl"])]

def assert_surgery(det_model, variant: str, where: str):
    i = BG.surgery_present(det_model); want = "band" if variant == "mech" else "random"
    ok = i["class"] == "BandGatedDetect" and i["selection"] == want and i["strides"] == [4.0, 8.0, 16.0, 32.0] and i["nl"] == 4
    assert ok, f"SURGERY MISSING on {where} ({variant}): {i}"

def make_trainer(variant: str):
    from ultralytics.models.yolo.detect import DetectionTrainer
    class SurgeryTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            m = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            added = apply_surgery(m, variant, SEED[0]); assert_surgery(m, variant, "get_model")
            print(f"[surgery] {variant}: +{added} params on trainer-built model"); return m
    return SurgeryTrainer

def _check_cb(variant):
    def cb(trainer):
        assert_surgery(trainer.model, variant, "trainer.model @pretrain_end")
        if getattr(trainer, "ema", None) is not None: assert_surgery(trainer.ema.ema, variant, "trainer.ema.ema @pretrain_end")
        print(f"[surgery] verified on trainer.model + ema before epoch 1 ({variant})")
    return cb

# ── training ──────────────────────────────────────────────────────────────
def _curve(save_dir):
    try:
        rows = list(csv.DictReader(open(os.path.join(save_dir, "results.csv"))))
        col = next(c for c in rows[0] if c.strip().endswith("mAP50(B)")); return [float(r[col]) for r in rows]
    except Exception: return []

def train(variant, epochs, seed, name, data=None, val_full=True, batch=None, extra=None):
    SEED[0] = seed
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    t0 = time.time(); model = YOLO(P2_YAML)
    model.add_callback("on_pretrain_routine_end", _check_cb(variant))
    kw = dict(trainer=make_trainer(variant), data=data or A.TRAIN_DATA_YAML, epochs=epochs, imgsz=A.IMG_SIZE, batch=batch or A.BATCH, seed=seed,
              device=0, workers=8, amp=A.AMP, val=True, plots=False, pretrained=False, project=os.path.join(PROJECT, "runs_visdrone"),
              name=name, exist_ok=True, optimizer="MuSGD", cache="ram")
    if extra: kw.update(extra)
    model.train(**kw); tr = model.trainer
    assert_surgery(tr.model, variant, "trainer.model @end"); assert_surgery(tr.ema.ema, variant, "trainer.ema.ema @end")
    out = {"variant": variant, "seed": seed, "epochs": epochs, "save_dir": str(tr.save_dir),
           "tripwire": {"trainer_model": surgery_classes(tr.model), "ema": surgery_classes(tr.ema.ema), "trainer_model_params": sum(p.numel() for p in tr.model.parameters())},
           "curve_val500_map50": _curve(str(tr.save_dir)), "wall_min": (time.time() - t0) / 60}
    if val_full:
        last = YOLO(str(tr.last)); assert_surgery(last.model, variant, "last.pt")
        f = last.val(data=A.DATA_YAML, imgsz=A.IMG_SIZE, batch=A.BATCH, plots=False, verbose=False, device=0)
        out.update(primary=float(f.box.map50), secondary={"mAP50_95": float(f.box.map), "precision": float(f.box.mp), "recall": float(f.box.mr)})
        ad = A.Yolo26nVisdroneScratchAdapter(); out["per_scale_recall"] = ad.per_scale_metric(last); out["per_class_ap50"] = ad.per_class_metric(last)
    return out

def dup_stats(model) -> dict:
    """Per-image predictions at conf .25: fraction matching an already-matched GT (duplicates) vs matching nothing (FP)."""
    import yaml
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter, _iou, DATA_YAML, IMG_SIZE
    A = Yolo26nVisdroneScratchAdapter(); cfg = yaml.safe_load(open(DATA_YAML)); imgs = A._val_image_list(cfg)
    npred = dup = fp = 0
    for p in imgs:
        r = model.predict(p, imgsz=IMG_SIZE, conf=0.25, verbose=False, device=0)[0]; gt = A._load_gt(p, cfg)
        preds = sorted([(int(r.boxes.cls[i]), float(r.boxes.conf[i]), *[float(v) for v in r.boxes.xyxy[i]]) for i in range(len(r.boxes))], key=lambda t: -t[1])
        npred += len(preds); matched = set()
        for pc, _, *pb in preds:
            best, bj = 0.0, -1
            for j, (gc, *gb) in enumerate(gt):
                if j in matched: continue
                i = _iou(pb, gb)
                if i > best: best, bj = i, j
            if best >= 0.5: matched.add(bj)
            elif any(_iou(pb, gb) >= 0.5 for j, (gc, *gb) in enumerate(gt) if j in matched): dup += 1
            else: fp += 1
    return {"n_pred": npred, "dup_frac": dup / max(1, npred), "fp_frac": fp / max(1, npred)}

def coverage_stats(model) -> dict:
    """Fraction of GT centres (by input-scale size) inside cells the gate selects; random gating expects q."""
    import cv2, yaml
    from ultralytics.data.augment import LetterBox
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter, DATA_YAML
    A = Yolo26nVisdroneScratchAdapter(); cfg = yaml.safe_load(open(DATA_YAML)); imgs = A._val_image_list(cfg)
    net = model.model.cuda().eval().float(); det = net.model[-1]; lb = LetterBox((640, 640), auto=False)
    hit = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for p in imgs:
        im0 = cv2.imread(p); h0, w0 = im0.shape[:2]; r = min(640 / h0, 640 / w0); nh, nw = round(h0 * r), round(w0 * r); py, px = (640 - nh) / 2, (640 - nw) / 2
        x = torch.from_numpy(lb(image=im0)[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
        with torch.no_grad(): net(x)
        sel = set(det.last_sel[0].tolist()); lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp): continue
        for line in open(lp):
            c, cx, cy, bw, bh = map(float, line.split()[:5]); X, Y = cx * w0 * r + px, cy * h0 * r + py; side = max(bw * w0, bh * h0) * r
            k = "small" if side < 32 else ("medium" if side < 96 else "large"); hit[k][1] += 1; hit[k][0] += (int(Y // 8) * 80 + int(X // 8)) in sel
    return {k: {"covered": v[0] / max(1, v[1]), "n": v[1]} for k, v in hit.items()}

# ── commands ──────────────────────────────────────────────────────────────
def cmd_stage(a):
    cal = json.load(open(f"{PROJECT}/trajectory/profiles/calibration.json"))
    r = train(a.variant, cal["s2_epochs"], a.seed, f"exp011_{a.variant}_S2_seed{a.seed}"); r["stage"] = "S2"
    r["delta_vs_baseline_seed"] = (r["primary"] - BASE_S2[a.seed]["primary"]) if a.seed in BASE_S2 else None
    _w(f"{a.variant}_S2_seed{a.seed}.json", r)

def cmd_units(a):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter
    m = YOLO(a.weights); assert_surgery(m.model, a.variant, "units weights")
    units = Yolo26nVisdroneScratchAdapter().per_unit_metric(m); v = [x for _, x in units]
    _w(f"per_unit_S2_{a.variant}_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "weights": a.weights, "n_units": len(v), "mean_f1": sum(v) / len(v),
        "degenerate_frac": sum(1 for x in v if x in (0.0, 1.0)) / len(v), "f1_zero_frac": sum(1 for x in v if x == 0.0) / len(v), "dup": dup_stats(m), "coverage": coverage_stats(m), "units": units})

def cmd_screen(a):
    m = json.load(open(f"{STATE}/mech_S2_seed{a.seed}.json")); d = m["primary"] - BASE_S2[a.seed]["primary"]; bar = PREREG["kill_bar_seed42_delta"]
    _w(f"screen_seed{a.seed}.json", {"seed": a.seed, "mech": m["primary"], "baseline": BASE_S2[a.seed]["primary"], "delta": d, "kill_bar": bar, "decision": "KILL" if d < bar else "PROMOTE"})

def cmd_basedup(_a):
    from ultralytics import YOLO
    for s in (42, 123, 7):
        p = f"{B0}/dup_S2_seed{s}.json"
        if os.path.exists(p): continue
        w = BASE_S2[s]["save_dir"] + "/weights/last.pt"; r = dup_stats(YOLO(w)); json.dump({"seed": s, "weights": w, **r}, open(p, "w"), indent=2); print("[basedup]", s, r)

def cmd_gradcheck(_a):
    from ultralytics import YOLO
    net = YOLO(P2_YAML).model; apply_surgery(net, "mech"); net.train()
    out = net(torch.rand(2, 3, 640, 640, generator=torch.Generator().manual_seed(0)))
    loss = sum(out[h][k].float().abs().mean() for h in ("one2one", "one2many") for k in ("scores", "boxes")); net.zero_grad(); loss.backward()
    det = net.model[-1]; gn = lambda mods: sum(float(p.grad.norm()) for m_ in mods for p in m_.parameters() if p.grad is not None)
    g = {"p2_o2m": gn([det.cv2[0], det.cv3[0]]), "p2_o2o": gn([det.one2one_cv2[0], det.one2one_cv3[0]]), "p2_neck": gn([net.model[19]])}
    ok = all(v > 0 for v in g.values()); _w("gradcheck.json", {"grad_norm": g, "pass": ok}); print("GRADCHECK", "PASS" if ok else "FAIL", g); sys.exit(0 if ok else 1)

def cmd_dryrun(_a):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    base = sum(p.numel() for p in YOLO(A.MODEL_YAML).model.parameters()); res = {}
    for v in ("mech", "control"):
        net = YOLO(P2_YAML).model; n = sum(p.numel() for p in net.parameters()); apply_surgery(net, v); assert_surgery(net, v, "dryrun"); net.train()
        with torch.no_grad(): p = net(torch.rand(1, 3, 640, 640))
        live = float((p["one2one"]["boxes"][..., :25600] != 0).any(1).float().mean())
        net.eval()
        with torch.no_grad(): y = net(torch.rand(1, 3, 640, 640))
        y0 = y[0] if isinstance(y, tuple) else y
        res[v] = {"params_added_vs_baseline": n - base, "classes": surgery_classes(net), "n_anchors": int(p["one2one"]["scores"].shape[-1]), "live_stride4_frac": live, "out_shape": list(y0.shape)}
    ok = res["mech"]["params_added_vs_baseline"] == res["control"]["params_added_vs_baseline"] and res["mech"]["params_added_vs_baseline"] <= 0.10 * base and res["mech"]["n_anchors"] == 34000 and abs(res["mech"]["live_stride4_frac"] - Q) < 0.01 and abs(res["control"]["live_stride4_frac"] - Q) < 0.01
    res.update(ratio_control_over_mech=1.0, **{"pass": ok}); _w("dryrun.json", res); print("DRYRUN", "PASS" if ok else "FAIL", res); sys.exit(0 if ok else 1)

def cmd_tinyprobe(_a):
    imgs = sorted(l.strip() for l in open(f"{PROJECT}/visdrone_train.txt") if l.strip())[:8]
    lst = os.path.join(STATE, "tiny8.txt"); open(lst, "w").write("\n".join(imgs) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]))
    ty = os.path.join(STATE, "tiny.yaml"); open(ty, "w").write(f"path: {os.path.expanduser('~/datasets/VisDrone')}\ntrain: {lst}\nval: {lst}\nnames:\n{names}\n")
    out = {}
    for v in ("control", "mech"):   # control (generic) stands in for 'baseline speed'; both go through the surgery trainer = surgery dry-run
        r = train(v, 200, 42, f"exp011_tinyprobe_{v}", data=ty, val_full=False, batch=8, extra={"mosaic": 0.0, "val": False, "warmup_epochs": 0})
        rows = list(csv.DictReader(open(os.path.join(r["save_dir"], "results.csv")))); cols = [c for c in rows[0] if "train/" in c]
        loss = [sum(float(x[c]) for c in cols) for x in rows]
        out[v] = {"loss_first": loss[0], "loss_last": loss[-1], "loss_every_20": loss[::20], "wall_min": r["wall_min"], "tripwire": r["tripwire"]}
    ratio = out["mech"]["loss_last"] / out["control"]["loss_last"]; ok = ratio <= 1.5
    out.update(ratio_mech_over_control=ratio, bar=1.5, **{"pass": ok}); _w("tinyprobe.json", out); print("TINYPROBE", "PASS" if ok else "FAIL", f"ratio={ratio:.3f}"); sys.exit(0 if ok else 1)

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--variant", choices=["mech", "control"], required=True); s.add_argument("--seed", type=int, required=True)
    u = sub.add_parser("units"); u.add_argument("--variant", choices=["mech", "control"], required=True); u.add_argument("--seed", type=int, required=True); u.add_argument("--weights", required=True)
    sc = sub.add_parser("screen"); sc.add_argument("--seed", type=int, required=True)
    for n in ("gradcheck", "dryrun", "tinyprobe"): sub.add_parser(n)
    a = p.parse_args()
    {"stage": cmd_stage, "units": cmd_units, "screen": cmd_screen, "gradcheck": cmd_gradcheck, "dryrun": cmd_dryrun, "tinyprobe": cmd_tinyprobe}[a.cmd](a)

if __name__ == "__main__": main()
