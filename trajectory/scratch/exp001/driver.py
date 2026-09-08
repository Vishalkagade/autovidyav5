"""exp001 driver — P2 evidence fusion (topology) vs stride-8 param-matched control.

Topology kind: the model is built straight from a yaml, so plain Model.train
is safe (no in-place surgery to lose). Tripwire instead: Detect must expose
4 strides [4,8,16,32] for the mechanism and 3 for the control, asserted on
the model the TRAINER built (trainer.model), not the one we built.

CLI (scripts/pipeline_local.sh contract):
  stage  --variant mech|control --seed S      -> state/{variant}_S2_seed{S}.json
  units  --variant mech|control --seed S --weights W -> state/per_unit_S2_{variant}_seed{S}.json
  screen --seed 42                            -> state/screen_seed42.json {"decision": KILL|PROMOTE}
  gradcheck | dryrun | tinyprobe              -> pre-screen gates (state/*.json), exit 1 on fail
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT); os.chdir(PROJECT)
STATE = os.path.join(PROJECT, "trajectory", "scratch", "exp001", "state"); os.makedirs(STATE, exist_ok=True)
CFG = os.path.join(PROJECT, "adapters", "yolo26n_visdrone_scratch", "configs")
YAML = {"mech": os.path.join(CFG, "yolo26n-p2-visdrone.yaml"),
        "control": os.path.join(CFG, "yolo26n-p3ctl-visdrone.yaml"),
        "baseline": os.path.join(CFG, "yolo26n-visdrone.yaml")}
STRIDES = {"mech": [4, 8, 16, 32], "control": [8, 16, 32], "baseline": [8, 16, 32]}
PREREG = json.load(open(os.path.join(PROJECT, "trajectory", "scratch", "exp001", "prereg.json")))
BASE_S2 = {s: json.load(open(f"{PROJECT}/trajectory/scratch/exp000/state/S2_seed{s}.json")) for s in (42, 123, 7, 1000, 2000)
           if os.path.exists(f"{PROJECT}/trajectory/scratch/exp000/state/S2_seed{s}.json")}

def _w(name, obj):
    p = os.path.join(STATE, name); json.dump(obj, open(p, "w"), indent=2); print(f"[driver] wrote {p}")

def _detect(net):
    return net.model[-1]

def _strides(net):
    return [int(s) for s in _detect(net).stride.tolist()]

def _nparams(net):
    return sum(p.numel() for p in net.parameters())

def _curve(save_dir):
    try:
        rows = list(csv.DictReader(open(os.path.join(save_dir, "results.csv"))))
        col = next(c for c in rows[0] if c.strip().endswith("mAP50(B)"))
        return [float(r[col]) for r in rows]
    except Exception:
        return []

def train(variant, epochs, seed, name, data=None, val_full=True, batch=None, extra=None):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    t0 = time.time()
    model = YOLO(YAML[variant])
    kw = dict(data=data or A.TRAIN_DATA_YAML, epochs=epochs, imgsz=A.IMG_SIZE, batch=batch or A.BATCH, seed=seed,
              device=0, workers=8, amp=A.AMP, val=True, plots=False, pretrained=False,
              project=os.path.join(PROJECT, "runs_visdrone"), name=name, exist_ok=True, optimizer="MuSGD", cache="ram")
    if extra: kw.update(extra)
    model.train(**kw)
    tr = model.trainer
    # TRIPWIRE on the trainer-built model (the one that actually trained)
    st = _strides(tr.model); assert st == STRIDES[variant], f"tripwire: {variant} strides {st} != {STRIDES[variant]}"
    out = {"variant": variant, "seed": seed, "epochs": epochs, "save_dir": str(tr.save_dir),
           "tripwire": {"trainer_model_strides": st, "trainer_model_params": _nparams(tr.model)},
           "curve_val500_map50": _curve(str(tr.save_dir)), "wall_min": (time.time() - t0) / 60}
    if val_full:
        last = YOLO(str(tr.last))
        f = last.val(data=A.DATA_YAML, imgsz=A.IMG_SIZE, batch=A.BATCH, plots=False, verbose=False, device=0)
        out.update(primary=float(f.box.map50), secondary={"mAP50_95": float(f.box.map), "precision": float(f.box.mp), "recall": float(f.box.mr)})
        out["per_scale_recall"] = A.Yolo26nVisdroneScratchAdapter().per_scale_metric(last)
        out["per_class_ap50"] = A.Yolo26nVisdroneScratchAdapter().per_class_metric(last)
    return out

def cmd_stage(a):
    cal = json.load(open(f"{PROJECT}/trajectory/profiles/calibration.json"))
    r = train(a.variant, cal["s2_epochs"], a.seed, f"exp001_{a.variant}_S2_seed{a.seed}")
    r["stage"] = "S2"; r["delta_vs_baseline_seed"] = (r["primary"] - BASE_S2[a.seed]["primary"]) if a.seed in BASE_S2 else None
    _w(f"{a.variant}_S2_seed{a.seed}.json", r)

def cmd_units(a):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter
    units = Yolo26nVisdroneScratchAdapter().per_unit_metric(YOLO(a.weights)); v = [x for _, x in units]
    _w(f"per_unit_S2_{a.variant}_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "weights": a.weights, "n_units": len(v),
        "mean_f1": sum(v) / len(v), "degenerate_frac": sum(1 for x in v if x in (0.0, 1.0)) / len(v),
        "f1_zero_frac": sum(1 for x in v if x == 0.0) / len(v), "units": units})

def cmd_screen(a):
    m = json.load(open(f"{STATE}/mech_S2_seed{a.seed}.json"))
    d = m["primary"] - BASE_S2[a.seed]["primary"]; bar = PREREG["kill_bar_seed42_delta"]
    small_ok = m["per_scale_recall"]["small_recall"] - PREREG["discriminator"]["baseline_small_recall_seed123"]
    _w(f"screen_seed{a.seed}.json", {"seed": a.seed, "mech": m["primary"], "baseline": BASE_S2[a.seed]["primary"], "delta": d,
        "kill_bar": bar, "decision": "KILL" if d < bar else "PROMOTE", "small_recall_delta_vs_seed123_baseline": small_ok})

def cmd_gradcheck(_a):
    import torch
    from ultralytics import YOLO
    net = YOLO(YAML["mech"]).model; net.train()
    x = torch.rand(2, 3, 640, 640, generator=torch.Generator().manual_seed(0))
    out = net(x); flat = out
    while isinstance(flat, (tuple, list, dict)):
        flat = list(flat.values())[0] if isinstance(flat, dict) else flat[0]
    net.zero_grad(); flat.float().abs().mean().backward()
    # new P2-path layers in yolo26-p2: 17 (Upsample) 18 (Concat) 19 (C3k2 P2) 20 (Conv down) — check 19 and 20
    g = {i: sum(float(p.grad.norm()) for p in net.model[i].parameters() if p.grad is not None) for i in (19, 20)}
    ok = all(v > 0 for v in g.values())
    _w("gradcheck.json", {"grad_norm_by_layer": g, "pass": ok, "strides": _strides(net)}); print("GRADCHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

def cmd_dryrun(_a):
    import torch
    from ultralytics import YOLO
    res = {}
    for v in ("baseline", "mech", "control"):
        net = YOLO(YAML[v]).model.eval()
        with torch.no_grad(): y = net(torch.rand(1, 3, 640, 640))
        res[v] = {"params": _nparams(net), "strides": _strides(net)}
    res["added_mech"] = res["mech"]["params"] - res["baseline"]["params"]; res["added_control"] = res["control"]["params"] - res["baseline"]["params"]
    ok = (res["mech"]["strides"] == STRIDES["mech"] and res["control"]["strides"] == STRIDES["control"]
          and 0.9 <= res["added_control"] / res["added_mech"] <= 1.1 and res["added_mech"] <= 0.10 * res["baseline"]["params"])
    res["pass"] = ok; _w("dryrun.json", res); print("DRYRUN", "PASS" if ok else "FAIL", res); sys.exit(0 if ok else 1)

def cmd_tinyprobe(_a):
    # 8 fixed train images, 200 iterations at batch 8 (1 iter/epoch), baseline vs mechanism; loss curves recorded.
    imgs = sorted(l.strip() for l in open(f"{PROJECT}/visdrone_train.txt") if l.strip())[:8]
    lst = os.path.join(STATE, "tiny8.txt"); open(lst, "w").write("\n".join(imgs) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]))
    ty = os.path.join(STATE, "tiny.yaml"); open(ty, "w").write(f"path: {os.path.expanduser('~/datasets/VisDrone')}\ntrain: {lst}\nval: {lst}\nnames:\n{names}\n")
    out = {}
    for v in ("baseline", "mech"):
        r = train(v, 200, 42, f"exp001_tinyprobe_{v}", data=ty, val_full=False, batch=8, extra={"mosaic": 0.0, "val": False, "warmup_epochs": 0})
        rows = list(csv.DictReader(open(os.path.join(r["save_dir"], "results.csv"))))
        cols = [c for c in rows[0] if "train/" in c]
        loss = [sum(float(x[c]) for c in cols) for x in rows]
        out[v] = {"loss_first": loss[0], "loss_last": loss[-1], "loss_every_20": loss[::20], "wall_min": r["wall_min"]}
    ratio = out["mech"]["loss_last"] / out["baseline"]["loss_last"]; ok = ratio <= 1.5
    out.update(ratio_mech_over_baseline=ratio, bar=1.5, **{"pass": ok}); _w("tinyprobe.json", out); print("TINYPROBE", "PASS" if ok else "FAIL", f"ratio={ratio:.3f}"); sys.exit(0 if ok else 1)

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--variant", choices=["mech", "control"], required=True); s.add_argument("--seed", type=int, required=True)
    u = sub.add_parser("units"); u.add_argument("--variant", choices=["mech", "control"], required=True); u.add_argument("--seed", type=int, required=True); u.add_argument("--weights", required=True)
    sc = sub.add_parser("screen"); sc.add_argument("--seed", type=int, required=True)
    for n in ("gradcheck", "dryrun", "tinyprobe"): sub.add_parser(n)
    a = p.parse_args()
    {"stage": cmd_stage, "units": cmd_units, "screen": cmd_screen, "gradcheck": cmd_gradcheck, "dryrun": cmd_dryrun, "tinyprobe": cmd_tinyprobe}[a.cmd](a)

if __name__ == "__main__":
    main()
