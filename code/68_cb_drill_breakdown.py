"""Where in the full CB combine data is the coverage information?

(1) Drill by drill: ridge on ONE drill's full feature set (mechanics + curves + phases), same leave-one-draft-class-out
    rep-level scoring as 67 (context-only baseline), with a CB-permutation null (100).
(2) The transition drill at full resolution vs its single frozen number (drive top speed).
(3) Interpretability: ridge on all features fitted on all CBs, coefficients aggregated by drill and by feature
    family; stability = Spearman correlation of the coefficient vectors between leave-one-class-out fits; and the
    transition-drill curve coefficients over time (which part of the drill matters).
Outcomes: separation gained at the break, separation at the throw (all coverage + zone), completion over expected.
"""
import importlib.util, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
spec = importlib.util.spec_from_file_location("m67", f"{ROOT}/code/67_cb_full_models.py")
src = open(f"{ROOT}/code/67_cb_full_models.py").read().split("OUTS = [")[0]
ns = {}; exec(src, ns)
cov, Vw0, Pm0, CBS = ns["cov"], ns["Vw"], ns["Pm"], ns["CBS"]
rng = np.random.default_rng(31)
DRILLS = sorted({c.split("|")[0] for c in Vw0.columns if "|" in c})
OUTS = [("sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)", None),
        ("sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran)", None),
        ("sep_at_throw", "wr_skill + C(route_ran)", "ZONE_COVERAGE"),
        ("cpoe", "rec_target_prop + C(team_coverage_man_zone)", None)]
split = lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())]


def with_features(cols):
    ns["FE"] = cols
    ns["Vw"] = Vw0[["nfl_id", "view"] + cols]


import sys
rows = []
for y, ctx, sub in ([] if '--interp-only' in sys.argv else OUTS):
    d = cov if sub is None else cov[cov.team_coverage_man_zone == sub]
    d = d.dropna(subset=[y] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    blocks = {f"drill: {dr}": [c for c in Vw0.columns if c.startswith(dr + "|")] for dr in DRILLS}
    tcols = blocks["drill: BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"]
    blocks["transition drill: curves only"] = [c for c in tcols if "|curve_" in c]
    blocks["transition drill: mechanics + phases"] = [c for c in tcols if "|curve_" not in c]
    for name, cols in blocks.items():
        if len(cols) < 2:
            continue
        with_features(cols)
        obs = ns["run_split"](d, y, ctx, split(d), ["ridge_all"])["ridge_all"]
        null = []
        for _ in range(100):
            perm = dict(zip(CBS, rng.permutation(CBS)))
            ns["Vw"] = Vw0[["nfl_id", "view"] + cols].assign(nfl_id=Vw0.nfl_id.map(lambda k: perm.get(k, k)))
            null.append(ns["run_split"](d, y, ctx, split(d), ["ridge_all"])["ridge_all"][0])
        rows.append({"outcome": y + ("" if sub is None else " (zone)"), "block": name, "n_features": len(cols),
                     "oos_mse_reduction_%": obs[0], "cb_effect_r": obs[1], "perm_p": (1 + (np.array(null) >= obs[0]).sum()) / 101})
    print(f"{y}{'' if sub is None else ' (zone)'} done", flush=True)
R = pd.DataFrame(rows)
if len(R): R.to_csv(f"{ROOT}/results/68_cb_drill_breakdown.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
if len(R): print(R.sort_values(["outcome", "oos_mse_reduction_%"], ascending=[True, False]).round(3).to_string(index=False))

# ---------------- (3) interpretability of the all-feature ridge
print("\n=== all-feature ridge: where the weight sits, and how stable it is ===")
allc = [c for c in Vw0.columns if c not in ("nfl_id", "view")]
with_features(allc)
for y, ctx, sub in [OUTS[0], OUTS[1], OUTS[3]]:
    d = cov.dropna(subset=[y] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    cm = smf.ols(f"{y} ~ {ctx}", data=d).fit(); d["res"] = cm.resid
    coefs = {}
    for c in ["all"] + sorted(d.cls.unique()):
        dd = d if c == "all" else d[d.cls != c]
        eff = dd.groupby("nfl_id_d").res.agg(["mean", "size"]).rename(columns={"mean": "ybar", "size": "n"})
        out = ns["cb_models"](eff, eff.index.values, ["ridge_all"])
        coefs[c] = out["_ridge_coef"]
    CF = pd.DataFrame(coefs)
    stab = [stats.spearmanr(CF[a], CF[b])[0] for i, a in enumerate(CF.columns[1:]) for b in CF.columns[i + 2:]]
    w = CF["all"]
    by_drill = w.abs().groupby(lambda c: c.split("|")[0]).sum() / w.abs().sum()
    fam = w.abs().groupby(lambda c: "curve" if "|curve_" in c else ("transition phase" if "|trans_" in c else "mechanic")).sum() / w.abs().sum()
    print(f"\n{y}: coefficient stability across leave-one-class-out fits (Spearman, mean of pairs) {np.mean(stab):+.2f}")
    print("  share of |coefficient| mass by drill:", by_drill.sort_values(ascending=False).round(3).to_dict())
    print("  by family:", fam.round(3).to_dict())
    top = w.reindex(w.abs().sort_values(ascending=False).index[:12])
    print("  top features:", ", ".join(f"{k} {v:+.4f}" for k, v in top.items()))
    for sg in ["speed", "turnrate", "along", "lateral"]:
        tc = w[[c for c in w.index if c.startswith(f"BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION|curve_{sg}_")]]
        print(f"  transition-drill {sg:8s} curve coefficients (x1000) over normalized time 0 -> 1:", np.round(tc.values * 1000, 1).tolist())
