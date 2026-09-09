"""exp007 driver — Horvitz-Thompson loss reweighting (aux_objective, driver-side)
vs the permuted-weight placebo control. No architecture change: both variants
attach E2ELoss(loss_fn=HTLoss) as model.criterion at on_pretrain_routine_end
(after the trainer has built the model and set model.args). Tripwire: the
loss's applied_batches / weight-area correlation are recorded per stage.

CLI: stage --variant mech|control --seed S | units ... | screen --seed 42 |
     dryrun | lossdiff | tinyprobe
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, PROJECT); os.chdir(PROJECT)
STATE = os.path.join(PROJECT, "trajectory", "scratch", "exp007", "state"); os.makedirs(STATE, exist_ok=True)
B0 = os.path.join(PROJECT, "trajectory", "scratch", "exp000", "state")
PREREG = json.load(open(os.path.join(PROJECT, "trajectory", "scratch", "exp007", "prereg.json")))
BASE_S2 = {s: json.load(open(f"{B0}/S2_seed{s}.json")) for s in (42, 123, 7, 1000, 2000) if os.path.exists(f"{B0}/S2_seed{s}.json")}
MARGIN, LAM = PREREG["mechanism"]["margin"], PREREG["mechanism"]["lambda"]
from adapters.yolo26n_visdrone_scratch.modules import siblingmargin as SM

def _w(name, obj):
    p = os.path.join(STATE, name); json.dump(obj, open(p, "w"), indent=2); print(f"[driver] wrote {p}")

def _curve(save_dir):
    try:
        rows = list(csv.DictReader(open(os.path.join(save_dir, "results.csv"))))
        col = next(c for c in rows[0] if c.strip().endswith("mAP50(B)")); return [float(r[col]) for r in rows]
    except Exception: return []

def train(variant, epochs, seed, name, data=None, val_full=True, batch=None, extra=None):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch import adapter as A
    t0 = time.time(); model = YOLO(A.MODEL_YAML); holder = {}
    def attach_cb(trainer):
        holder["stats"] = SM.attach(trainer.model, variant, MARGIN, LAM)
        assert type(trainer.model.criterion).__name__ == "E2ELoss" and type(trainer.model.criterion.one2many).__name__ == "SiblingMarginLoss", "criterion not attached"
        print(f"[siblingmargin] attached ({variant}) on trainer.model")
    model.add_callback("on_pretrain_routine_end", attach_cb)
    kw = dict(data=data or A.TRAIN_DATA_YAML, epochs=epochs, imgsz=A.IMG_SIZE, batch=batch or A.BATCH, seed=seed, device=0, workers=8, amp=A.AMP,
              val=True, plots=False, pretrained=False, project=os.path.join(PROJECT, "runs_visdrone"), name=name, exist_ok=True, optimizer="MuSGD", cache="ram")
    if extra: kw.update(extra)
    model.train(**kw); tr = model.trainer; st = holder["stats"]
    assert st["applied_batches"] > 0 and st["n_pos_paired"] > 0, "TRIPWIRE: sibling margin never applied"
    trip = {"applied_batches": st["applied_batches"], "n_pos_paired": st["n_pos_paired"], "mean_margin_term": st["sum_term"] / max(1, st["applied_batches"]),
            "criterion": type(tr.model.criterion.one2many).__name__, "variant_pairs": SM.SIBLINGS if variant == "mech" else SM.CONTROL_PAIRS}
    out = {"variant": variant, "seed": seed, "epochs": epochs, "save_dir": str(tr.save_dir), "tripwire": trip, "curve_val500_map50": _curve(str(tr.save_dir)), "wall_min": (time.time() - t0) / 60}
    if val_full:
        last = YOLO(str(tr.last))
        f = last.val(data=A.DATA_YAML, imgsz=A.IMG_SIZE, batch=A.BATCH, plots=False, verbose=False, device=0)
        out.update(primary=float(f.box.map50), secondary={"mAP50_95": float(f.box.map), "precision": float(f.box.mp), "recall": float(f.box.mr)})
        ad = A.Yolo26nVisdroneScratchAdapter(); out["per_scale_recall"] = ad.per_scale_metric(last); out["per_class_ap50"] = ad.per_class_metric(last)
        out["sibconf"] = sibconf(last)
    return out

# ── D1 extras: sibling-pair confusion at conf .25 (adapter matcher, class-agnostic localisation) ─
def sibconf(model) -> dict:
    import yaml, collections
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter, _iou, DATA_YAML, IMG_SIZE
    A = Yolo26nVisdroneScratchAdapter(); cfg = yaml.safe_load(open(DATA_YAML)); imgs = A._val_image_list(cfg)
    tab = SM.pair_table(SM.SIBLINGS).tolist(); located = collections.Counter(); confused = collections.Counter()
    for p in imgs:
        r = model.predict(p, imgsz=IMG_SIZE, conf=0.25, verbose=False, device=0)[0]; gt = A._load_gt(p, cfg)
        preds = sorted([(int(r.boxes.cls[i]), float(r.boxes.conf[i]), *[float(v) for v in r.boxes.xyxy[i]]) for i in range(len(r.boxes))], key=lambda t: -t[1])
        matched = set()
        for pc, _, *pb in preds:
            best, bj = 0.0, -1
            for j, (gc, *gb) in enumerate(gt):
                if j in matched: continue
                i = _iou(pb, gb)
                if i > best: best, bj = i, j
            if best >= 0.5:
                matched.add(bj); gc = int(gt[bj][0]); located[gc] += 1
                if tab[gc] == pc: confused[gc] += 1
    per_class = {SM.NAMES[c]: {"located": located[c], "as_sibling": confused[c], "rate": confused[c] / max(1, located[c])} for c in range(len(SM.NAMES))}
    tot_loc = sum(located.values()); tot_conf = sum(confused.values())
    return {"per_class": per_class, "pooled_rate": tot_conf / max(1, tot_loc), "located_total": tot_loc, "confused_total": tot_conf}

def cmd_basesib(_a):
    from ultralytics import YOLO
    for s in (42, 123, 7):
        p = f"{B0}/sibconf_S2_seed{s}.json"
        if os.path.exists(p): continue
        w = BASE_S2[s]["save_dir"] + "/weights/last.pt"; r = sibconf(YOLO(w)); json.dump({"seed": s, "weights": w, **r}, open(p, "w"), indent=2); print("[basesib]", s, "pooled %.4f" % r["pooled_rate"], {k: round(v["rate"], 3) for k, v in r["per_class"].items()})

def cmd_stage(a):
    cal = json.load(open(f"{PROJECT}/trajectory/profiles/calibration.json"))
    r = train(a.variant, cal["s2_epochs"], a.seed, f"exp007_{a.variant}_S2_seed{a.seed}"); r["stage"] = "S2"
    r["delta_vs_baseline_seed"] = (r["primary"] - BASE_S2[a.seed]["primary"]) if a.seed in BASE_S2 else None
    _w(f"{a.variant}_S2_seed{a.seed}.json", r)

def cmd_units(a):
    from ultralytics import YOLO
    from adapters.yolo26n_visdrone_scratch.adapter import Yolo26nVisdroneScratchAdapter
    units = Yolo26nVisdroneScratchAdapter().per_unit_metric(YOLO(a.weights)); v = [x for _, x in units]
    _w(f"per_unit_S2_{a.variant}_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "weights": a.weights, "n_units": len(v), "mean_f1": sum(v) / len(v),
        "degenerate_frac": sum(1 for x in v if x in (0.0, 1.0)) / len(v), "f1_zero_frac": sum(1 for x in v if x == 0.0) / len(v), "units": units})

def cmd_screen(a):
    m = json.load(open(f"{STATE}/mech_S2_seed{a.seed}.json")); d = m["primary"] - BASE_S2[a.seed]["primary"]; bar = PREREG["kill_bar_seed42_delta"]
    _w(f"screen_seed{a.seed}.json", {"seed": a.seed, "mech": m["primary"], "baseline": BASE_S2[a.seed]["primary"], "delta": d, "kill_bar": bar, "decision": "KILL" if d < bar else "PROMOTE"})

def _cpu_batch():
    import torch
    from ultralytics import YOLO
    from ultralytics.data import build_yolo_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from adapters.yolo26n_visdrone_scratch import adapter as A
    model = YOLO(A.MODEL_YAML).model; cfg = get_cfg(); cfg.imgsz = 640; cfg.mosaic = 0.0
    data = check_det_dataset(A.TRAIN_DATA_YAML); ds = build_yolo_dataset(cfg, data["train"], 8, data, mode="train", stride=32)
    from torch.utils.data import DataLoader
    b = next(iter(DataLoader(ds, batch_size=8, shuffle=False, collate_fn=ds.collate_fn))); b["img"] = b["img"].float() / 255
    model.args = cfg; return model, b

def cmd_dryrun(_a):
    import torch
    model, b = _cpu_batch(); res = {}
    with torch.no_grad(): preds = model(b["img"])
    for v in ("mech", "control"):
        st = SM.attach(model, v, MARGIN, LAM); loss, items = model.criterion(preds, b)
        res[v] = {"loss_total": float(loss.sum()), "applied_batches": st["applied_batches"], "n_pos_paired": st["n_pos_paired"], "mean_margin_term": st["sum_term"] / max(1, st["applied_batches"])}
    ok = res["mech"]["applied_batches"] == 2 and res["mech"]["n_pos_paired"] > 0 and res["mech"]["mean_margin_term"] > 0 and res["control"]["n_pos_paired"] > 0
    res["pass"] = ok; _w("dryrun.json", res); print("DRYRUN", "PASS" if ok else "FAIL", res); sys.exit(0 if ok else 1)

def cmd_lossdiff(_a):
    import torch
    model, b = _cpu_batch(); model.train()
    def grad_of(attach):
        model.zero_grad(); model.criterion = None
        if attach: attach()
        preds = model(b["img"]); loss, _ = model.loss(b, preds); loss.sum().backward()
        return float(loss.sum()), torch.cat([p.grad.flatten() for p in model.model[-1].cv3.parameters()]).clone()
    l0, g0 = grad_of(None); l1, g1 = grad_of(lambda: SM.attach(model, "mech", MARGIN, LAM))
    cos = float(torch.nn.functional.cosine_similarity(g0, g1, dim=0)); ok = abs(l1 - l0) > 1e-6 and cos < 0.999
    _w("lossdiff.json", {"vanilla_loss": l0, "margin_loss": l1, "cls_grad_cosine": cos, "pass": ok}); print("LOSSDIFF", "PASS" if ok else "FAIL", {"vanilla": l0, "margin": l1, "cos": cos}); sys.exit(0 if ok else 1)

def cmd_tinyprobe(_a):
    imgs = sorted(l.strip() for l in open(f"{PROJECT}/visdrone_train.txt") if l.strip())[:8]
    lst = os.path.join(STATE, "tiny8.txt"); open(lst, "w").write("\n".join(imgs) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]))
    ty = os.path.join(STATE, "tiny.yaml"); open(ty, "w").write(f"path: {os.path.expanduser('~/datasets/VisDrone')}\ntrain: {lst}\nval: {lst}\nnames:\n{names}\n")
    out = {}
    for v in ("control", "mech"):
        r = train(v, 200, 42, f"exp007_tinyprobe_{v}", data=ty, val_full=False, batch=8, extra={"mosaic": 0.0, "val": False, "warmup_epochs": 0})
        rows = list(csv.DictReader(open(os.path.join(r["save_dir"], "results.csv")))); cols = [c for c in rows[0] if "train/" in c]
        loss = [sum(float(x[c]) for c in cols) for x in rows]
        out[v] = {"loss_first": loss[0], "loss_last": loss[-1], "loss_every_20": loss[::20], "wall_min": r["wall_min"], "tripwire": r["tripwire"]}
    ratio = out["mech"]["loss_last"] / out["control"]["loss_last"]; ok = ratio <= 1.5 and out["mech"]["tripwire"]["applied_batches"] >= 200 and out["mech"]["tripwire"]["n_pos_paired"] > 0
    out.update(ratio_mech_over_control=ratio, bar=1.5, **{"pass": ok}); _w("tinyprobe.json", out); print("TINYPROBE", "PASS" if ok else "FAIL", f"ratio={ratio:.3f}"); sys.exit(0 if ok else 1)

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--variant", choices=["mech", "control"], required=True); s.add_argument("--seed", type=int, required=True)
    u = sub.add_parser("units"); u.add_argument("--variant", choices=["mech", "control"], required=True); u.add_argument("--seed", type=int, required=True); u.add_argument("--weights", required=True)
    sc = sub.add_parser("screen"); sc.add_argument("--seed", type=int, required=True)
    for n in ("dryrun", "lossdiff", "tinyprobe", "basesib"): sub.add_parser(n)
    a = p.parse_args()
    {"stage": cmd_stage, "units": cmd_units, "screen": cmd_screen, "dryrun": cmd_dryrun, "lossdiff": cmd_lossdiff, "tinyprobe": cmd_tinyprobe, "basesib": cmd_basesib}[a.cmd](a)

if __name__ == "__main__": main()
