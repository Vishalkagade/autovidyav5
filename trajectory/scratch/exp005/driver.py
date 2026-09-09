"""exp005 driver — divisive normalization on the one2one class logits (site_wrap
at head.detect) vs the adapter's GenericResidualConv param-matched control.

SURGERY TRAINER (mandatory for site_wrap): Model.train rebuilds the model from
yaml and drops in-place surgery (ins_v5_surgery_dropped_by_train). We override
DetectionTrainer.get_model to re-apply the surgery on the trainer-built model,
and a callback asserts the surgery on trainer.model AND trainer.ema.ema at
pretrain-routine end (fails fast, before any GPU epoch). The stage JSON records
the classes found on both after training (tripwire).

CLI (scripts/pipeline_local.sh contract):
  stage  --variant mech|control --seed S
  units  --variant mech|control --seed S --weights W
  screen --seed 42
  gradcheck | dryrun | tinyprobe | basedup   (pre-screen gates; basedup = baseline duplicate stats for D1)
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, PROJECT); os.chdir(PROJECT)
STATE = os.path.join(PROJECT, "trajectory", "scratch", "exp005", "state"); os.makedirs(STATE, exist_ok=True)
B0 = os.path.join(PROJECT, "trajectory", "scratch", "exp000", "state")
PREREG = json.load(open(os.path.join(PROJECT, "trajectory", "scratch", "exp005", "prereg.json")))
BASE_S2 = {s: json.load(open(f"{B0}/S2_seed{s}.json")) for s in (42, 123, 7, 1000, 2000) if os.path.exists(f"{B0}/S2_seed{s}.json")}

from torch import nn
from adapters.yolo26n_visdrone_scratch.modules import mhdetect as MH
from adapters.yolo26n_visdrone_scratch.generic_control import GenericResidualConv
CTL_TARGET = PREREG["control"].get("target_params_each") or 2154   # half the mechanism's +4,308

def _w(name, obj):
    p = os.path.join(STATE, name); json.dump(obj, open(p, "w"), indent=2); print(f"[driver] wrote {p}")

# ── surgery ───────────────────────────────────────────────────────────────
def apply_surgery(det_model, variant: str) -> int:
    if variant == "mech": return MH.apply_surgery(det_model)
    det = det_model.model[-1]; before = sum(p.numel() for p in det.parameters())
    det.cv3[0] = GenericResidualConv(det.cv3[0], target_params=CTL_TARGET)
    det.one2one_cv3[0] = GenericResidualConv(det.one2one_cv3[0], target_params=CTL_TARGET)
    return sum(p.numel() for p in det.parameters()) - before

def surgery_classes(det_model) -> list[str]:
    det = det_model.model[-1]
    return [type(det).__name__, type(det.cv3[0]).__name__, type(det.one2one_cv3[0]).__name__, str([float(v) for v in det.stride.tolist()])]

def assert_surgery(det_model, variant: str, where: str):
    got = surgery_classes(det_model)
    if variant == "mech": ok = got[0] == "MultiHypDetect" and got[3] == "[4.0, 16.0, 32.0]"
    else: ok = got[0] == "Detect" and got[1] == got[2] == "GenericResidualConv"
    assert ok, f"SURGERY MISSING on {where} ({variant}): {got}"

def make_trainer(variant: str):
    from ultralytics.models.yolo.detect import DetectionTrainer
    class SurgeryTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            m = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            added = apply_surgery(m, variant); assert_surgery(m, variant, "get_model")
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
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    t0 = time.time(); model = YOLO(A.MODEL_YAML)
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

# ── commands ──────────────────────────────────────────────────────────────
def cmd_stage(a):
    cal = json.load(open(f"{PROJECT}/trajectory/profiles/calibration.json"))
    r = train(a.variant, cal["s2_epochs"], a.seed, f"exp005_{a.variant}_S2_seed{a.seed}"); r["stage"] = "S2"
    r["delta_vs_baseline_seed"] = (r["primary"] - BASE_S2[a.seed]["primary"]) if a.seed in BASE_S2 else None
    _w(f"{a.variant}_S2_seed{a.seed}.json", r)

def cmd_units(a):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter
    m = YOLO(a.weights); assert_surgery(m.model, a.variant, "units weights")
    units = Yolo26nVisdroneScratchAdapter().per_unit_metric(m); v = [x for _, x in units]
    _w(f"per_unit_S2_{a.variant}_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "weights": a.weights, "n_units": len(v), "mean_f1": sum(v) / len(v),
        "degenerate_frac": sum(1 for x in v if x in (0.0, 1.0)) / len(v), "f1_zero_frac": sum(1 for x in v if x == 0.0) / len(v), "units": units})

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
    import torch
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    net = YOLO(A.MODEL_YAML).model; added = apply_surgery(net, "mech"); net.train()
    out = net(torch.rand(2, 3, 640, 640, generator=torch.Generator().manual_seed(0)))
    loss = sum(out[h][k].float().abs().mean() for h in ("one2one", "one2many") for k in ("scores", "boxes"))
    net.zero_grad(); loss.backward(); det = net.model[-1]
    g = {n: float(m[-1].weight.grad.norm()) for n, m in (("cv2_p3", det.cv2[0]), ("cv3_p3", det.cv3[0]), ("o2o_cv2_p3", det.one2one_cv2[0]), ("o2o_cv3_p3", det.one2one_cv3[0]))}
    ok = all(v > 0 for v in g.values()); _w("gradcheck.json", {"grad_norm": g, "params_added": added, "pass": ok}); print("GRADCHECK", "PASS" if ok else "FAIL", g); sys.exit(0 if ok else 1)

def cmd_dryrun(_a):
    import torch
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    res = {}
    for v in ("mech", "control"):
        net = YOLO(A.MODEL_YAML).model; base = sum(p.numel() for p in net.parameters()); added = apply_surgery(net, v); assert_surgery(net, v, "dryrun"); net.eval()
        with torch.no_grad(): y = net(torch.rand(1, 3, 640, 640))
        y0 = y[0] if isinstance(y, tuple) else y; preds = y[1]["one2one"] if isinstance(y, tuple) else None
        res[v] = {"params_added": added, "classes": surgery_classes(net), "out_shape": list(y0.shape), "n_anchors": int(preds["scores"].shape[-1]) if preds else None}
    ratio = res["control"]["params_added"] / res["mech"]["params_added"]
    ok = 0.9 <= ratio <= 1.1 and res["mech"]["params_added"] <= 0.10 * base and res["mech"]["n_anchors"] == 4 * 6400 + 1600 + 400 and res["control"]["n_anchors"] == 8400
    res.update(ratio_control_over_mech=ratio, **{"pass": ok}); _w("dryrun.json", res); print("DRYRUN", "PASS" if ok else "FAIL", res); sys.exit(0 if ok else 1)

def cmd_tinyprobe(_a):
    imgs = sorted(l.strip() for l in open(f"{PROJECT}/visdrone_train.txt") if l.strip())[:8]
    lst = os.path.join(STATE, "tiny8.txt"); open(lst, "w").write("\n".join(imgs) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]))
    ty = os.path.join(STATE, "tiny.yaml"); open(ty, "w").write(f"path: {os.path.expanduser('~/datasets/VisDrone')}\ntrain: {lst}\nval: {lst}\nnames:\n{names}\n")
    out = {}
    for v in ("control", "mech"):   # control (generic) stands in for 'baseline speed'; both go through the surgery trainer = surgery dry-run
        r = train(v, 200, 42, f"exp005_tinyprobe_{v}", data=ty, val_full=False, batch=8, extra={"mosaic": 0.0, "val": False, "warmup_epochs": 0})
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
