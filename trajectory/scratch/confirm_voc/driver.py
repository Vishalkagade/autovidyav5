"""VOC confirmation-track driver (no-regression leg for exp001's P2 head).
probe | stage --variant baseline|mech --seed S | units --variant --seed --weights | assemble"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, statistics
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, P); os.chdir(P)
HERE = os.path.join(P, "trajectory", "scratch", "confirm_voc"); ST = os.path.join(HERE, "state"); os.makedirs(ST, exist_ok=True)
CFG = os.path.join(P, "adapters", "yolo26n_visdrone_scratch", "configs")
YAML = {"baseline": os.path.join(CFG, "yolo26n-voc.yaml"), "mech": os.path.join(CFG, "yolo26n-p2-voc.yaml")}
STRIDES = {"baseline": [8, 16, 32], "mech": [4, 8, 16, 32]}
DATA, DATA_S1 = os.path.join(HERE, "voc.yaml"), os.path.join(HERE, "voc_s1.yaml")
sys.path.insert(0, os.path.join(P, "trajectory", "scratch", "exp000")); from driver import pick_budgets   # the pre-registered budget rule
from adapters.yolo26n_visdrone_scratch.adapter import image_f1, _greedy_match, _area_bucket
import yaml as _yaml
def _w(n, o): json.dump(o, open(os.path.join(ST, n), "w"), indent=2); print("[voc] wrote", n)
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
    t0 = time.time(); m = YOLO(YAML[variant])
    m.train(data=DATA_S1, epochs=epochs, imgsz=640, batch=32, seed=seed, device=0, workers=8, amp=True, val=True, plots=False, pretrained=False,
            project=os.path.join(P, "runs_voc"), name=name, exist_ok=True, optimizer="MuSGD", cache="ram")
    tr = m.trainer; st = [int(s) for s in tr.model.model[-1].stride.tolist()]; assert st == STRIDES[variant], (variant, st)
    last = YOLO(str(tr.last)); f = last.val(data=DATA, imgsz=640, batch=32, plots=False, verbose=False, device=0)
    return {"variant": variant, "seed": seed, "epochs": epochs, "save_dir": str(tr.save_dir), "tripwire_strides": st,
            "primary": float(f.box.map50), "secondary": {"mAP50_95": float(f.box.map), "precision": float(f.box.mp), "recall": float(f.box.mr)},
            "per_class_ap50": [{"class_name": last.names[int(i)], "value": float(a)} for i, a in zip(f.box.ap_class_index, f.box.ap50)],
            "curve_val500_map50": _curve(str(tr.save_dir)), "wall_min": (time.time() - t0) / 60}
def per_unit_and_scale(weights):
    from ultralytics import YOLO
    m = YOLO(weights); imgs = sorted(l.strip() for l in open(_yaml.safe_load(open(DATA))["val"]) if l.strip())
    units = []; matched = {"small": 0, "medium": 0, "large": 0}; total = dict(matched)
    for p in imgs:
        r = m.predict(p, imgsz=640, conf=0.25, verbose=False, device=0)[0]; gt = _gt(p)
        preds = [(int(r.boxes.cls[i]), float(r.boxes.conf[i]), *[float(v) for v in r.boxes.xyxy[i]]) for i in range(len(r.boxes))]
        units.append((os.path.basename(p), image_f1(preds, gt)))
        if gt:
            mi = _greedy_match(preds, gt)
            for j, (_, x1, y1, x2, y2) in enumerate(gt):
                b = _area_bucket((x2 - x1) * (y2 - y1)); total[b] += 1; matched[b] += j in mi
    return units, {f"{b}_recall": matched[b] / total[b] if total[b] else 0.0 for b in total}
def cmd_probe(_):
    r = train("baseline", 200, 42, "voc_probe_baseline_seed42"); r["budgets"] = pick_budgets(r["curve_val500_map50"]); r["min_per_epoch"] = r["wall_min"] / 200; _w("probe.json", r); print("[voc] budgets", r["budgets"])
def cmd_stage(a):
    ep = json.load(open(os.path.join(ST, "probe.json")))["budgets"]["s2_epochs"]; r = train(a.variant, ep, a.seed, f"voc_{a.variant}_S2_seed{a.seed}"); _w(f"{a.variant}_S2_seed{a.seed}.json", r)
def cmd_units(a):
    u, ps = per_unit_and_scale(a.weights); v = [x for _, x in u]
    _w(f"per_unit_{a.variant}_seed{a.seed}.json", {"variant": a.variant, "seed": a.seed, "n_units": len(u), "mean_f1": sum(v) / len(v), "degenerate_frac": sum(1 for x in v if x in (0.0, 1.0)) / len(v), "per_scale_recall": ps, "units": u})
def cmd_assemble(_):
    from core.discipline.p10_stats_gate import paired_unit_test
    S = (42, 123, 7); L = lambda n: json.load(open(os.path.join(ST, n)))
    b = {s: L(f"baseline_S2_seed{s}.json") for s in S}; m = {s: L(f"mech_S2_seed{s}.json") for s in S}
    bu = {s: [tuple(u) for u in L(f"per_unit_baseline_seed{s}.json")["units"]] for s in S}; mu = {s: [tuple(u) for u in L(f"per_unit_mech_seed{s}.json")["units"]] for s in S}
    floor = max(x["primary"] for x in b.values()) - min(x["primary"] for x in b.values()); d = sum(m[s]["primary"] - b[s]["primary"] for s in S) / 3
    t = paired_unit_test(mu, bu); sig_unfav = t.ci95[1] < 0
    out = {"baseline_map50_by_seed": {str(s): b[s]["primary"] for s in S}, "mech_map50_by_seed": {str(s): m[s]["primary"] for s in S}, "voc_s2_noise_floor": floor,
           "seedavg_delta_map50": d, "per_unit": {"n_units": t.n_units, "mean_diff": t.mean_diff, "ci95": list(t.ci95), "wilcoxon_p": t.wilcoxon_p, "per_seed_means": {str(k): v for k, v in t.per_seed_means.items()}},
           "per_scale_recall": {str(s): {"baseline": L(f"per_unit_baseline_seed{s}.json")["per_scale_recall"], "mech": L(f"per_unit_mech_seed{s}.json")["per_scale_recall"]} for s in S},
           "bar_a_map50": d >= -floor, "bar_b_per_unit_not_sig_unfavorable": not sig_unfav, "wall_min": {"baseline": [b[s]["wall_min"] for s in S], "mech": [m[s]["wall_min"] for s in S]}}
    out["NO_REGRESSION_PASS"] = out["bar_a_map50"] and out["bar_b_per_unit_not_sig_unfavorable"]; _w("assembly.json", out); print("[voc] NO_REGRESSION", "PASS" if out["NO_REGRESSION_PASS"] else "FAIL", {k: out[k] for k in ("seedavg_delta_map50", "voc_s2_noise_floor")}, out["per_unit"]["ci95"])
def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True); sub.add_parser("probe"); sub.add_parser("assemble")
    s = sub.add_parser("stage"); s.add_argument("--variant", choices=["baseline", "mech"], required=True); s.add_argument("--seed", type=int, required=True)
    u = sub.add_parser("units"); u.add_argument("--variant", choices=["baseline", "mech"], required=True); u.add_argument("--seed", type=int, required=True); u.add_argument("--weights", required=True)
    a = p.parse_args(); {"probe": cmd_probe, "stage": cmd_stage, "units": cmd_units, "assemble": cmd_assemble}[a.cmd](a)
if __name__ == "__main__": main()
