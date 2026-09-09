"""exp003 classification — runs the gate chain on the finished state files and
writes trajectory/experiments/exp003.json (+ index/cost rows). Judgment text
(insight claim, notes) is passed in via --notes; everything numeric is here."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, subprocess, sys
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, P); os.chdir(P)
from core.discipline.p10_stats_gate import paired_unit_test
from core.discipline.p9_matched_control import attribution_check
from core.discipline.p11_replication import control_must_replicate
ST = "trajectory/scratch/exp003/state"; B = "trajectory/scratch/exp000/state"
L = lambda p: json.load(open(p)); ex = os.path.exists
SEEDS = (42, 123, 7)
prereg = L("trajectory/scratch/exp003/prereg.json"); cal = L("trajectory/profiles/calibration.json"); floor = cal["noise_floor_s2_map50"]

def units(path): return {int(x) if False else x: None for x in []} or L(path)["units"]
def by_seed(fmt, seeds): return {s: [tuple(u) for u in L(fmt.format(s))["units"]] for s in seeds if ex(fmt.format(s))}

def t2d(r):  # P10Result -> schema p10_test
    return {"n_units": r.n_units, "mean_diff": r.mean_diff, "median_diff": r.median_diff, "ci95": list(r.ci95), "wilcoxon_p": r.wilcoxon_p,
            "units_improved": r.units_improved, "per_seed_means": {str(k): v for k, v in r.per_seed_means.items()}, "verdict": "pass" if r.passed else ("direction-consistent-ns" if r.mean_diff > 0 else "unfavorable"), "reason": r.reason}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--notes", default=""); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    mech = {s: L(f"{ST}/mech_S2_seed{s}.json") for s in SEEDS if ex(f"{ST}/mech_S2_seed{s}.json")}
    ctl = {s: L(f"{ST}/control_S2_seed{s}.json") for s in SEEDS if ex(f"{ST}/control_S2_seed{s}.json")}
    base = {s: L(f"{B}/S2_seed{s}.json") for s in SEEDS}
    bps = {s: L(f"{B}/per_scale_S2_seed{s}.json")["per_scale_recall"] for s in SEEDS}
    bdup = {s: L(f"{B}/dup_S2_seed{s}.json")["dup_frac"] for s in SEEDS}
    # D2: GT count per eval unit (>=100 boxes = dense bucket)
    gtc = {}
    for line in open("visdrone_eval.txt"):
        p = line.strip()
        if not p: continue
        lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        gtc[os.path.basename(p)] = sum(1 for _ in open(lp)) if os.path.exists(lp) else 0
    def dense_f1(units): return sum(v for u, v in units if gtc.get(u, 0) >= 100) / max(1, sum(1 for u, _ in units if gtc.get(u, 0) >= 100))
    screen = L(f"{ST}/screen_seed42.json") if ex(f"{ST}/screen_seed42.json") else None
    log = open("trajectory/scratch/exp003/local.log").read()
    out = {"id": "exp003", "phase": "phase-1", "adapter_name": "yolo26n_visdrone_scratch",
           "mechanism": {"name": prereg["mechanism"]["name"], "source_domain": prereg["mechanism"]["source_domain"], "kind": "site_wrap",
                         "family": prereg["mechanism"]["family"], "site": prereg["mechanism"]["site"], "params_count": prereg["mechanism"]["params_added_measured"]},
           "cites_baseline_finding_ids": prereg["cites_baseline_finding_ids"], "novelty_audit": {**prereg["novelty_audit"], "nearest_cv_analog": prereg["novelty_audit"]["nearest_cv_analog"], "nearest_cv_block": prereg["novelty_audit"]["nearest_cv_analog"], "structural_difference": prereg["novelty_audit"]["mechanism_of_difference"], "mechanism_of_difference": prereg["novelty_audit"]["mechanism_of_difference"] + " REGIME ARGUMENT: " + prereg["novelty_audit"]["regime_argument"]},
           "stages_run": ["S2"], "metrics_per_stage_per_seed": {"S2": {str(s): {"primary": m["primary"], "secondary": m["secondary"]} for s, m in mech.items()}},
           "stage1_curve": {str(s): m["curve_val500_map50"] for s, m in mech.items()},
           "provenance": {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                          "data_manifest_sha256": hashlib.sha256(open("visdrone_manifest.json", "rb").read()).hexdigest(),
                          "ultralytics_commit": subprocess.run(["git", "-C", "../ultralytics_src", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
           "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "boldness": "bold"}
    # discriminator
    bu0 = by_seed(B + "/per_unit_S2_seed{}.json", SEEDS); mu0 = by_seed(ST + "/per_unit_S2_mech_seed{}.json", SEEDS)
    disc = {str(s): {"D1_dup_frac_mech": m["dup_stats"]["dup_frac"], "D1_dup_frac_base": bdup[s], "D1_delta": m["dup_stats"]["dup_frac"] - bdup[s],
                     "D2_dense_f1_mech": dense_f1(mu0[s]) if s in mu0 else None, "D2_dense_f1_base": dense_f1(bu0[s]),
                     "D2_delta": (dense_f1(mu0[s]) - dense_f1(bu0[s])) if s in mu0 else None, "delta": m["dup_stats"]["dup_frac"] - bdup[s]} for s, m in mech.items()}
    disc_pass = all(v["D1_delta"] < 0 and (v["D2_delta"] is not None and v["D2_delta"] > 0) for v in disc.values())
    # deltas
    mb = sum(mech[s]["primary"] for s in mech) / len(mech) - sum(base[s]["primary"] for s in mech) / len(mech)
    per_seed_delta = {str(s): mech[s]["primary"] - base[s]["primary"] for s in mech}
    p10 = {}
    bu = by_seed(B + "/per_unit_S2_seed{}.json", SEEDS); mu = by_seed(ST + "/per_unit_S2_mech_seed{}.json", SEEDS); cu = by_seed(ST + "/per_unit_S2_control_seed{}.json", SEEDS)
    if len(mu) >= 2: p10["vs_baseline"] = t2d(paired_unit_test({s: mu[s] for s in mu}, {s: bu[s] for s in mu}))
    if len(cu) >= 2 and len(mu) >= 2:
        p10["vs_control"] = t2d(paired_unit_test({s: mu[s] for s in cu}, {s: cu[s] for s in cu}))
        p10["control_vs_baseline"] = t2d(paired_unit_test({s: cu[s] for s in cu}, {s: bu[s] for s in cu}))
    if p10: out["p10"] = p10
    p9 = None
    if ctl:
        cb = sum(ctl[s]["primary"] for s in ctl) / len(ctl); bb = sum(base[s]["primary"] for s in ctl) / len(ctl); mm = sum(mech[s]["primary"] for s in ctl) / len(ctl)
        r9 = attribution_check(mech_seedavg=mm, control_seedavg=cb, baseline_seedavg=bb, noise_floor_s2=floor)
        p9 = {"control_run_ids": [ctl[s]["save_dir"] for s in ctl], "mech_minus_baseline": r9.mech_minus_baseline, "mech_minus_control": r9.mech_minus_control,
              "control_minus_baseline": r9.control_minus_baseline, "attribution_pass": r9.passed, "reason": r9.reason,
              "control_must_replicate_at_p11": control_must_replicate(r9.mech_minus_control, floor)}
        out["p9"] = p9
    # classification
    if screen and screen["decision"] == "KILL": cls, sub = "Kill", "screened-out"
    elif "FUTILITY_REJECT" in log: cls, sub = "Reject", "noise-floor"
    elif len(mech) < 3: cls, sub = "Op-Fail", "crash"
    else:
        vb = p10.get("vs_baseline", {}); vc = p10.get("vs_control", {})
        same_sign = all(v > 0 for v in vb.get("per_seed_means", {}).values()) if vb else False
        if p9 and p9["attribution_pass"] and vb.get("verdict") == "pass" and vc.get("verdict") == "pass" and same_sign: cls, sub = "Provisional Winner", "winner-verified"
        elif p9 and vb.get("verdict") == "pass" and vc.get("verdict") != "pass": cls, sub = "Hold", "capacity-explains-gain"
        elif vb.get("mean_diff", 0) > 0: cls, sub = "Hold", "direction-consistent-ns"
        else: cls, sub = "Reject", "noise-floor"
    gate = [l for l in log.splitlines() if "P9_CONTROL_GATE" in l]
    evidence = (f"seed deltas mAP50 {per_seed_delta}; seed-avg mech-baseline {mb:+.4f} ({mb/floor:+.1f}x floor); discriminators D1 dup_frac deltas {{{', '.join(f'{k}: {v['D1_delta']:+.4f}' for k, v in disc.items())}}} D2 dense-F1 deltas {{{', '.join(f'{k}: {(v['D2_delta'] if v['D2_delta'] is not None else float('nan')):+.4f}' for k, v in disc.items())}}} -> {'PASS' if disc_pass else 'FAIL (discriminator_failed)'}; "
                + (gate[-1].split('] ')[1] if gate else "no gate line") + ("; " + a.notes if a.notes else ""))
    out["attribution"] = {"type": "scientific" if cls != "Op-Fail" else "operational", "subtype": sub, "evidence": evidence}
    out["classification"] = cls
    out["discriminator"] = {"statement": prereg["discriminators"]["D1"]["statement"] + " AND " + prereg["discriminators"]["D2"]["statement"], "by_seed": disc, "pass": disc_pass}
    gpu = sum(m["wall_min"] for m in mech.values()) / 60 + sum(c["wall_min"] for c in ctl.values()) / 60
    out["cost_gpu_hours"] = gpu; out["notes"] = a.notes
    import jsonschema; errs = [e.message for e in jsonschema.Draft202012Validator(L("core/schemas/experiment.schema.json")).iter_errors(out)]
    print(json.dumps({k: out[k] for k in ("classification", "attribution", "discriminator", "p9", "p10") if k in out}, indent=1)); print("schema errors:", errs)
    if a.write and not errs:
        json.dump(out, open("trajectory/experiments/exp003.json", "w"), indent=2)
        idx = L("trajectory/index.json")
        ps = {s: m["per_scale_recall"]["small_recall"] for s, m in mech.items()}
        idx["scoreboard"].append({"exp": "exp003", "family": prereg["mechanism"]["family"], "site": prereg["mechanism"]["site"], "params_added": prereg["mechanism"]["params_added_measured"],
            "mAP50": sum(m["primary"] for m in mech.values()) / len(mech), "mAP50_delta": mb,
            "mAP50_95_delta": sum(m["secondary"]["mAP50_95"] for m in mech.values()) / len(mech) - sum(base[s]["secondary"]["mAP50_95"] for s in mech) / len(mech),
            "small_recall": sum(ps.values()) / len(ps), "small_recall_delta": sum(ps[s] - bps[s]["small_recall"] for s in ps) / len(ps),
            "f1_zero_mass_delta": (sum(L(f"{ST}/per_unit_S2_mech_seed{s}.json")["f1_zero_frac"] for s in mu) / len(mu) - sum(L(f"{B}/per_unit_S2_seed{s}.json")["f1_zero_frac"] for s in mu) / len(mu)) if mu else None,
            "classification": cls})
        idx["phase"] = f"Phase-1 — exp003 {cls}"
        json.dump(idx, open("trajectory/index.json", "w"), indent=2); open("trajectory/index.json", "a").write("\n")
        with open("trajectory/cost_log.csv", "a") as f:
            cum = float(open("trajectory/cost_log.csv").read().strip().splitlines()[-1].split(",")[-1])
            for tag, d in (("exp003_mech", mech), ("exp003_control", ctl)):
                for s, r in d.items(): cum += r["wall_min"] / 60; f.write(f"{s},{tag},{r['wall_min']/60:.3f},{cum:.3f}\n")
        print("WROTE exp003.json, index.json, cost_log.csv")
if __name__ == "__main__": main()
