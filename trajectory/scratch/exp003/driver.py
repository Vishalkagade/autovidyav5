"""exp003 driver — divisive normalization on the one2one class logits (site_wrap
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
STATE = os.path.join(PROJECT, "trajectory", "scratch", "exp003", "state"); os.makedirs(STATE, exist_ok=True)
B0 = os.path.join(PROJECT, "trajectory", "scratch", "exp000", "state")
PREREG = json.load(open(os.path.join(PROJECT, "trajectory", "scratch", "exp003", "prereg.json")))
BASE_S2 = {s: json.load(open(f"{B0}/S2_seed{s}.json")) for s in (42, 123, 7, 1000, 2000) if os.path.exists(f"{B0}/S2_seed{s}.json")}
CTL_TARGET = PREREG["control"]["target_params_per_scale"]

from torch import nn
from adapters.yolo26n_visdrone_scratch.modules.divnorm import DivNorm
from adapters.yolo26n_visdrone_scratch.generic_control import GenericResidualConv

def _w(name, obj):
    p = os.path.join(STATE, name); json.dump(obj, open(p, "w"), indent=2); print(f"[driver] wrote {p}")

# ── surgery ───────────────────────────────────────────────────────────────
def apply_surgery(det_model, variant: str) -> int:
    det = det_model.model[-1]; before = sum(p.numel() for p in det.parameters())
    for i in range(det.nl):
        orig = det.one2one_cv3[i]
        if variant == "mech": det.one2one_cv3[i] = nn.Sequential(orig, DivNorm(det.nc))
        elif variant == "control": det.one2one_cv3[i] = GenericResidualConv(orig, target_params=CTL_TARGET)
        else: raise ValueError(variant)
    return sum(p.numel() for p in det.parameters()) - before

def surgery_classes(det_model) -> list[str]:
    det = det_model.model[-1]; out = []
    for i in range(det.nl):
        m = det.one2one_cv3[i]
        out.append(type(m[-1]).__name__ if isinstance(m, nn.Sequential) else type(m).__name__)
    return out

def assert_surgery(det_model, variant: str, where: str):
    want = {"mech": "DivNorm", "control": "GenericResidualConv"}[variant]
    got = surgery_classes(det_model)
    assert all(g == want for g in got), f"SURGERY MISSING on {where}: expected {want} x3, got {got}"

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
        out["dup_stats"] = dup_stats(last)
        if variant == "mech":
            out["divnorm_params"] = [{"sigma": float(m[-1].sigma), "n": float(m[-1].n), "w_sum": float(m[-1].w.sum())} for m in last.model.model[-1].one2one_cv3]
    return out

# ── D1 extras: duplicate / pure-FP fractions at conf .25 (adapter matcher) ─
def dup_stats(model) -> dict:
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

# ── commands ──────────────────────────────────────────────────────────────
def cmd_stage(a):
    cal = json.load(open(f"{PROJECT}/trajectory/profiles/calibration.json"))
    r = train(a.variant, cal["s2_epochs"], a.seed, f"exp003_{a.variant}_S2_seed{a.seed}"); r["stage"] = "S2"
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
    loss = out["one2one"]["scores"].float().abs().mean(); net.zero_grad(); loss.backward()
    g = [sum(float(p.grad.norm()) for p in m[-1].parameters() if p.grad is not None) for m in net.model[-1].one2one_cv3]
    ok = all(x > 0 for x in g); _w("gradcheck.json", {"grad_norm_per_scale": g, "params_added": added, "pass": ok}); print("GRADCHECK", "PASS" if ok else "FAIL", g); sys.exit(0 if ok else 1)

def cmd_dryrun(_a):
    import torch
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    res = {}
    for v in ("mech", "control"):
        net = YOLO(A.MODEL_YAML).model; base = sum(p.numel() for p in net.parameters()); added = apply_surgery(net, v); assert_surgery(net, v, "dryrun"); net.eval()
        with torch.no_grad(): y = net(torch.rand(1, 3, 640, 640))
        res[v] = {"params_added": added, "classes": surgery_classes(net), "out_shape": list(y[0].shape) if isinstance(y, tuple) else list(y.shape)}
    ratio = res["control"]["params_added"] / res["mech"]["params_added"]
    ok = 0.9 <= ratio <= 1.1 and res["mech"]["params_added"] <= 0.10 * base; res.update(ratio_control_over_mech=ratio, **{"pass": ok})
    _w("dryrun.json", res); print("DRYRUN", "PASS" if ok else "FAIL", res); sys.exit(0 if ok else 1)

def cmd_tinyprobe(_a):
    imgs = sorted(l.strip() for l in open(f"{PROJECT}/visdrone_train.txt") if l.strip())[:8]
    lst = os.path.join(STATE, "tiny8.txt"); open(lst, "w").write("\n".join(imgs) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]))
    ty = os.path.join(STATE, "tiny.yaml"); open(ty, "w").write(f"path: {os.path.expanduser('~/datasets/VisDrone')}\ntrain: {lst}\nval: {lst}\nnames:\n{names}\n")
    out = {}
    for v in ("control", "mech"):   # control (generic) stands in for 'baseline speed'; both go through the surgery trainer = surgery dry-run
        r = train(v, 200, 42, f"exp003_tinyprobe_{v}", data=ty, val_full=False, batch=8, extra={"mosaic": 0.0, "val": False, "warmup_epochs": 0})
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
    for n in ("gradcheck", "dryrun", "tinyprobe", "basedup"): sub.add_parser(n)
    a = p.parse_args()
    {"stage": cmd_stage, "units": cmd_units, "screen": cmd_screen, "gradcheck": cmd_gradcheck, "dryrun": cmd_dryrun, "tinyprobe": cmd_tinyprobe, "basedup": cmd_basedup}[a.cmd](a)

if __name__ == "__main__": main()
