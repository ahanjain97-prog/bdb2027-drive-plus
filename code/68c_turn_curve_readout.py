"""Which moment around the transition-drill turn predicts completions allowed? Ridge (grouped-CV alpha) on the v2
turn-anchored curves only (speed, along-track accel, |lateral accel|, turn rate; -1.0 .. +2.0 s around the turn start),
fitted on all CBs and on each leave-one-class-out training set; coefficients over time + stability."""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
ns = {}; exec(open(f"{ROOT}/code/67b_cb_full_models_v2.py").read().split("OUTS = [")[0], ns)
cov, Vw0 = ns["cov"], ns["Vw"]
cols = [c for c in Vw0.columns if "|turn_" in c]
ns["FE"] = cols; ns["Vw"] = Vw0[["nfl_id", "view"] + cols]
for y, ctx in [("cpoe", "rec_target_prop + C(team_coverage_man_zone) + C(cls)"), ("sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran) + C(cls)")]:
    d = cov.dropna(subset=[y] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    d["res"] = smf.ols(f"{y} ~ {ctx}", data=d).fit().resid
    coefs = {}
    for c in ["all"] + sorted(d.cls.unique()):
        dd = d if c == "all" else d[d.cls != c]
        eff = dd.groupby("nfl_id_d").res.agg(["mean", "size"]).rename(columns={"mean": "ybar", "size": "n"})
        coefs[c] = ns["cb_models"](eff, eff.index.values, ["ridge_all"])["_ridge_coef"]
    CF = pd.DataFrame(coefs)
    stab = np.mean([stats.spearmanr(CF[a], CF[b])[0] for i, a in enumerate(CF.columns[1:]) for b in CF.columns[i + 2:]])
    print(f"\n{y}: stability of the coefficient pattern across leave-one-class-out fits (Spearman) {stab:+.2f}")
    w = CF["all"]
    for sg in ["speed", "along", "lateral", "turnrate"]:
        sub = w[[c for c in w.index if f"|turn_{sg}_" in c]]
        times = [c.split("_")[-1] for c in sub.index]
        print(f"  {sg:8s} (t rel. to turn start, -1.0 .. +2.0 s, x1000):", " ".join(f"{v * 1000:+.1f}" for v in sub.values))
    # summarize by phase
    def phase(c):
        t = float(c.split("_")[-1].rstrip("s"))
        return "before turn (-1..0 s)" if t < 0 else ("turn (0..0.5 s)" if t <= 0.5 else ("early drive (0.6..1.2 s)" if t <= 1.2 else "late drive (1.3..2 s)"))
    agg = w.groupby([lambda c: c.split("|turn_")[1].split("_")[0], phase]).sum().unstack()
    print("  summed coefficient by signal x phase (x1000):\n" + (agg * 1000).round(1).to_string())
