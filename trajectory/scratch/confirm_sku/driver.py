"""SKU-110K confirmation-track driver (WIN leg for exp011; baseline vs bandgate).
probe | stage --variant baseline|mech --seed S | units --variant --seed --weights | assemble"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, statistics
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, P); os.chdir(P)
HERE = os.path.join(P, "trajectory", "scratch", "confirm_sku"); ST = os.path.join(HERE, "state"); os.makedirs(ST, exist_ok=True)
CFG = os.path.join(P, "adapters", "yolo26n_visdrone_scratch", "configs")
YAML = {"baseline": os.path.join(CFG, "yolo26n-sku.yaml"), "mech": os.path.join(CFG, "yolo26n-p2-sku.yaml"), "bandgate": os.path.join(CFG, "yolo26n-p2-sku.yaml")}
STRIDES = {"baseline": [8, 16, 32], "mech": [4, 8, 16, 32], "bandgate": [4, 8, 16, 32]}
VARIANTS = ("baseline", "mech", "bandgate")   # bandgate = exp011 (band-gated P2 level, q/B_hi from its prereg) via the SurgeryTrainer pattern
from adapters.yolo26n_visdrone_scratch.modules import bandgate as BG
from adapters.yolo26n_visdrone_scratch.modules import chunked_assigner; chunked_assigner.apply()   # SKU-110K density OOMs the assigner every batch with the 4-level head
BG_PREREG = json.load(open(os.path.join(P, "trajectory", "scratch", "exp011", "prereg.json")))["mechanism"]
def _bg_ok(net): i = BG.surgery_present(net); return i["class"] == "BandGatedDetect" and i["selection"] == "band" and i["strides"] == [4.0, 8.0, 16.0, 32.0]
def _make_trainer(seed):
    from ultralytics.models.yolo.detect import DetectionTrainer
    class SurgeryTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            m = super().get_model(cfg=cfg, weights=weights, verbose=verbose); BG.apply_surgery(m, "band", q=BG_PREREG["q"], b_hi=BG_PREREG["b_hi"], seed=seed); assert _bg_ok(m); return m
    return SurgeryTrainer
def _check_cb(tr):
    assert _bg_ok(tr.model) and (tr.ema is None or _bg_ok(tr.ema.ema)), "bandgate surgery missing before epoch 1"
DATA, DATA_S1 = os.path.join(HERE, "sku.yaml"), os.path.join(HERE, "sku_s1.yaml")
sys.path.insert(0, os.path.join(P, "trajectory", "scratch", "exp000")); from driver import pick_budgets   # the pre-registered budget rule
from adapters.yolo26n_visdrone_scratch.adapter import image_f1, _greedy_match, _area_bucket
import yaml as _yaml
def _w(n, o): json.dump(o, open(os.path.join(ST, n), "w"), indent=2); print("[sku] wrote", n)
def _curve(d):
    try:
        rows = list(csv.DictReader(open(os.path.join(d, "results.csv")))); col = next(c for c in rows[0] if c.strip().endswith("mAP50(B)")); return [float(r[col]) for r in rows]
    except Exception: return []
def _gt(p):
    from PIL import Image
    lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(lp): return []
    w, h = Image.open(p).size; out = []
    for line in open(lp):
        c, xc, yc, bw, bh = map(float, line.split()[:5]); out.append([c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return out
def train(variant, epochs, seed, name):
    from ultralytics import YOLO
    t0 = time.time(); m = YOLO(YAML[variant]); kw = {}
    if variant == "bandgate": kw = {"trainer": _make_trainer(seed)}; m.add_callback("on_pretrain_routine_end", _check_cb)
    m.train(data=DATA_S1, epochs=epochs, imgsz=640, batch=32, seed=seed, device=0, workers=8, amp=True, val=True, plots=False, pretrained=False,
            project=os.path.join(P, "runs_sku"), name=name, exist_ok=True, optimizer="MuSGD", cache="ram", **kw)
    tr = m.trainer; st = [int(s) for s in tr.model.model[-1].stride.tolist()]; assert st == STRIDES[variant], (variant, st)
    if variant == "bandgate": assert _bg_ok(tr.model) and _bg_ok(tr.ema.ema), "bandgate surgery missing at end"
    last = YOLO(str(tr.last)); f = last.val(data=DATA, imgsz=640, batch=32, plots=False, verbose=False, device=0)
    if variant == "bandgate": assert _bg_ok(last.model), "bandgate surgery missing on last.pt"
    return {"variant": variant, "seed": seed, "epochs": epochs, "save_dir": str(tr.save_dir), "tripwire_strides": st, "tripwire_class": type(tr.model.model[-1]).__name__,
            "primary": float(f.box.map50), "secondary": {"mAP50_95": float(f.box.map), "precision": float(f.box.mp), "recall": float(f.box.mr)},
            "per_class_ap50": [{"class_name": last.names[int(i)], "value": float(a)} for i, a in zip(f.box.ap_class_index, f.box.ap50)],
            "curve_val500_map50": _curve(str(tr.save_dir)), "wall_min": (time.time() - t0) / 60}
def per_unit_and_scale(weights):
    from ultralytics import YOLO
    m = YOLO(weights); imgs = sorted(l.strip() for l in open(_yaml.safe_load(open(DATA))["val"]) if l.strip())
    units = []; matched = {"small": 0, "medium": 0, "large": 0}; total = dict(matched)
    det = m.model.model[-1]; gated = type(det).__name__ == "BandGatedDetect"; cov = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}; skipped = []
    for p in imgs:
        try: r = m.predict(p, imgsz=640, conf=0.25, verbose=False, device=0)[0]
        except ValueError as e:   # truncated JPEG (SKU-110K test_274.jpg): the validator's scanner ignores it too, so the unit set is the validator's
            skipped.append(os.path.basename(p)); continue
        gt = _gt(p)
        if gated:   # band coverage of GT centres (input scale; letterbox to 640, cell = stride-8 index)
            from PIL import Image
            w0, h0 = Image.open(p).size; rr = min(640 / h0, 640 / w0); px, py = (640 - round(w0 * rr)) / 2, (640 - round(h0 * rr)) / 2; sel = set(det.last_sel[0].tolist())
            for _, x1, y1, x2, y2 in gt:
                side = max(x2 - x1, y2 - y1) * rr; k = "small" if side < 32 else ("medium" if side < 96 else "large"); X, Y = ((x1 + x2) / 2) * rr + px, ((y1 + y2) / 2) * rr + py
                cov[k][1] += 1; cov[k][0] += (int(Y // 8) * 80 + int(X // 8)) in sel
        preds = [(int(r.boxes.cls[i]), float(r.boxes.conf[i]), *[float(v) for v in r.boxes.xyxy[i]]) for i in range(len(r.boxes))]
        units.append((os.path.basename(p), image_f1(preds, gt)))
        if gt:
            mi = _greedy_match(preds, gt)
            for j, (_, x1, y1, x2, y2) in enumerate(gt):
                b = _area_bucket((x2 - x1) * (y2 - y1)); total[b] += 1; matched[b] += j in mi
    ps = {f"{b}_recall": matched[b] / total[b] if total[b] else 0.0 for b in total}
    if gated: ps["band_coverage"] = {k: {"covered": v[0] / max(1, v[1]), "n": v[1]} for k, v in cov.items()}
    ps["skipped_unreadable"] = skipped
    return units, ps
def cmd_probe(_):
    r = train("baseline", 200, 42, "sku_probe_baseline_seed42"); r["budgets"] = pick_budgets(r["curve_val500_map50"]); r["min_per_epoch"] = r["wall_min"] / 200; _w("probe.json", r); print("[sku] budgets", r["budgets"])
def cmd_stage(a):
    ep = json.load(open(os.path.join(ST, "probe.json")))["budgets"]["s2_epochs"]; r = train(a.variant, ep, a.seed, f"sku_{a.variant}_S2_seed{a.seed}"); _w(f"{a.variant}_S2_seed{a.seed}.json", r)
def cmd_finish(a):
    """Recover a stage whose training completed but whose final eval crashed (e.g. the eval label-cache race): evaluate last.pt and write the state file."""
    from ultralytics import YOLO
    d = os.path.join(P, "runs_sku", f"sku_{a.variant}_S2_seed{a.seed}"); last = YOLO(os.path.join(d, "weights", "last.pt"))
    st = [int(s) for s in last.model.model[-1].stride.tolist()]; assert st == STRIDES[a.variant], (a.variant, st)
    rows = list(csv.DictReader(open(os.path.join(d, "results.csv")))); assert len(rows) == json.load(open(os.path.join(ST, "probe.json")))["budgets"]["s2_epochs"], len(rows)
    tcol = next(c for c in rows[0] if c.strip() == "time"); wall = float(rows[-1][tcol]) / 60
    f = last.val(data=DATA, imgsz=640, batch=32, plots=False, verbose=False, device=0)
    _w(f"{a.variant}_S2_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "epochs": len(rows), "save_dir": d, "tripwire_strides": st, "primary": float(f.box.map50),
        "secondary": {"mAP50_95": float(f.box.map), "precision": float(f.box.mp), "recall": float(f.box.mr)},
        "per_class_ap50": [{"class_name": last.names[int(i)], "value": float(ap)} for i, ap in zip(f.box.ap_class_index, f.box.ap50)],
        "curve_val500_map50": _curve(d), "wall_min": wall, "recovered_by": "finish (final eval re-run after label-cache race; training complete)"})

def cmd_units(a):
    u, ps = per_unit_and_scale(a.weights); v = [x for _, x in u]
    _w(f"per_unit_{a.variant}_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "n_units": len(u), "mean_f1": sum(v) / len(v), "degenerate_frac": sum(1 for x in v if x in (0.0, 1.0)) / len(v), "per_scale_recall": ps, "units": u})
def cmd_assemble(a):
    from core.discipline.p10_stats_gate import paired_unit_test
    S = (42, 123, 7); L = lambda n: json.load(open(os.path.join(ST, n))); mv = a.mech_variant
    b = {s: L(f"baseline_S2_seed{s}.json") for s in S}; m = {s: L(f"{mv}_S2_seed{s}.json") for s in S}
    bu = {s: [tuple(u) for u in L(f"per_unit_baseline_seed{s}.json")["units"]] for s in S}; mu = {s: [tuple(u) for u in L(f"per_unit_{mv}_seed{s}.json")["units"]] for s in S}
    floor = max(x["primary"] for x in b.values()) - min(x["primary"] for x in b.values()); d = sum(m[s]["primary"] - b[s]["primary"] for s in S) / 3
    t = paired_unit_test(mu, bu); sig_unfav = t.ci95[1] < 0
    out = {"baseline_map50_by_seed": {str(s): b[s]["primary"] for s in S}, "mech_map50_by_seed": {str(s): m[s]["primary"] for s in S}, "s2_noise_floor": floor,
           "seedavg_delta_map50": d, "per_unit": {"n_units": t.n_units, "mean_diff": t.mean_diff, "ci95": list(t.ci95), "wilcoxon_p": t.wilcoxon_p, "per_seed_means": {str(k): v for k, v in t.per_seed_means.items()}},
           "mech_variant": mv, "per_scale_recall": {str(s): {"baseline": L(f"per_unit_baseline_seed{s}.json")["per_scale_recall"], "mech": L(f"per_unit_{mv}_seed{s}.json")["per_scale_recall"]} for s in S},
           "bar_a_map50": d >= -floor, "bar_b_per_unit_not_sig_unfavorable": not sig_unfav, "wall_min": {"baseline": [b[s]["wall_min"] for s in S], "mech": [m[s]["wall_min"] for s in S]}}
    out["win_a_map50_ge_floor"] = d >= floor; out["win_b_per_unit_pass_every_seed_positive"] = bool(t.passed) and all(v > 0 for v in t.per_seed_means.values())
    out["WIN_PASS"] = out["win_a_map50_ge_floor"] and out["win_b_per_unit_pass_every_seed_positive"]
    out["NO_REGRESSION_PASS"] = out["bar_a_map50"] and out["bar_b_per_unit_not_sig_unfavorable"]; _w("assembly.json" if mv == "mech" else f"assembly_{mv}.json", out); print("[sku] WIN", "PASS" if out["WIN_PASS"] else "FAIL", {k: out[k] for k in ("seedavg_delta_map50", "s2_noise_floor")}, out["per_unit"]["ci95"])
def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True); sub.add_parser("probe"); asm = sub.add_parser("assemble"); asm.add_argument("--mech-variant", choices=["mech", "bandgate"], default="mech")
    s = sub.add_parser("stage"); s.add_argument("--variant", choices=VARIANTS, required=True); s.add_argument("--seed", type=int, required=True)
    fi = sub.add_parser("finish"); fi.add_argument("--variant", choices=VARIANTS, required=True); fi.add_argument("--seed", type=int, required=True)
    u = sub.add_parser("units"); u.add_argument("--variant", choices=VARIANTS, required=True); u.add_argument("--seed", type=int, required=True); u.add_argument("--weights", required=True)
    a = p.parse_args(); {"probe": cmd_probe, "stage": cmd_stage, "units": cmd_units, "assemble": cmd_assemble, "finish": cmd_finish}[a.cmd](a)
if __name__ == "__main__": main()
