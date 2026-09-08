"""exp001 P11 — sealed-seed replication verdict. Updates exp001.json in place."""
from __future__ import annotations
import argparse, json, os, sys
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, P); os.chdir(P)
from core.discipline.p10_stats_gate import paired_unit_test
from core.discipline.p11_replication import replication_check
ST = "trajectory/scratch/exp001/state"; B = "trajectory/scratch/exp000/state"; L = lambda p: json.load(open(p))
W, C = (42, 123, 7), (1000, 2000)
def t2d(r): return {"n_units": r.n_units, "mean_diff": r.mean_diff, "median_diff": r.median_diff, "ci95": list(r.ci95), "wilcoxon_p": r.wilcoxon_p, "units_improved": r.units_improved,
                    "per_seed_means": {str(k): v for k, v in r.per_seed_means.items()}, "verdict": "pass" if r.passed else ("direction-consistent-ns" if r.mean_diff > 0 else "unfavorable"), "reason": r.reason}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    base = {s: L(f"{B}/S2_seed{s}.json") for s in W + C}; mech = {s: L(f"{ST}/mech_S2_seed{s}.json") for s in W + C}
    bu = {s: [tuple(u) for u in L(f"{B}/per_unit_S2_seed{s}.json")["units"]] for s in W + C}
    mu = {s: [tuple(u) for u in L(f"{ST}/per_unit_S2_mech_seed{s}.json")["units"]] for s in W + C}
    cu = {s: [tuple(u) for u in L(f"{ST}/per_unit_S2_control_seed{s}.json")["units"]] for s in W}
    conf_deltas = {s: mech[s]["primary"] - base[s]["primary"] for s in C}
    vb_all = paired_unit_test(mu, bu)                                   # all 5 seeds
    vc_all = paired_unit_test({s: mu[s] for s in W}, cu)               # control not replicated -> working seeds only (recorded)
    r = replication_check(per_seed_confirmation_deltas=conf_deltas, combined_vs_baseline=vb_all, combined_vs_control=vc_all, control_replicated=False)
    out = {"confirmation_seeds": list(C), "per_seed_confirmation_deltas_map50": {str(s): v for s, v in conf_deltas.items()},
           "confirmation_seed_metrics": {str(s): {"baseline": base[s]["primary"], "mech": mech[s]["primary"], "mech_small_recall": mech[s]["per_scale_recall"]["small_recall"]} for s in C},
           "combined_vs_baseline": t2d(vb_all), "combined_vs_control": t2d(vc_all), "control_replicated": False,
           "verdict": r.verdict, "reason": r.reason}
    print(json.dumps(out, indent=1))
    if a.write:
        e = L("trajectory/experiments/exp001.json"); e["p11"] = out; e["classification"] = r.verdict
        e["attribution"]["subtype"] = "winner-verified" if r.verdict == "Winner" else "replication-failed"
        e["attribution"]["evidence"] += f"; P11 {r.verdict}: confirmation deltas {conf_deltas}, combined vs-baseline CI {vb_all.ci95}, {r.reason}"
        e["metrics_per_stage_per_seed"]["S2"].update({str(s): {"primary": mech[s]["primary"], "secondary": mech[s]["secondary"]} for s in C})
        e["cost_gpu_hours"] += sum(mech[s]["wall_min"] for s in C) / 60
        import jsonschema; errs = [x.message for x in jsonschema.Draft202012Validator(L("core/schemas/experiment.schema.json")).iter_errors(e)]
        print("schema errors:", errs)
        if not errs:
            json.dump(e, open("trajectory/experiments/exp001.json", "w"), indent=2)
            idx = L("trajectory/index.json"); idx["scoreboard"][-1]["classification"] = r.verdict; idx["phase"] = f"Phase-1 — exp001 {r.verdict}"
            json.dump(idx, open("trajectory/index.json", "w"), indent=2); open("trajectory/index.json", "a").write("\n")
            with open("trajectory/cost_log.csv", "a") as f:
                cum = float(open("trajectory/cost_log.csv").read().strip().splitlines()[-1].split(",")[-1])
                for tag, d in (("exp000_sealed", base), ("exp001_mech_p11", mech)):
                    for s in C: cum += d[s]["wall_min"] / 60; f.write(f"{s},{tag},{d[s]['wall_min']/60:.3f},{cum:.3f}\n")
            print("WROTE exp001.json (p11), index.json, cost_log.csv")
if __name__ == "__main__": main()
