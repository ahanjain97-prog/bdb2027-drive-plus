"""Step 5: projection backtest -- what is combine drive speed worth when a team has only k NFL coverage reps on a CB?

Outcome per rep: context-adjusted residual (OLS on receiver skill, route, man/zone) of
  sep_at_throw (separation allowed at the throw)  and  sep_gain_break; also zone-only reps.
For each CB (reps in time order): the first k reps are 'observed', the rest (>= 10) are the target; forecast = predicted
mean of the remaining reps. Everything that is estimated (context model, drive coefficient, variances) is fitted on CBs
from the OTHER draft classes (leave-one-class-out), so each forecast is out-of-sample for that CB.
Forecasters:
  league     0
  reps_eb    early mean shrunk toward 0: w = k / (k + s2 / t2)          (t2 = between-CB variance, s2 = rep variance)
  combine    b * z_drive
  posterior  prior mean b * z_drive with prior variance t2_res (between-CB variance left after drive), updated with the
             k early reps:  (b*z/t2_res + k*ybar/s2) / (1/t2_res + k/s2)
  reps_cv / posterior_cv  the same two forecasters with the shrinkage weight TUNED on the training classes (grid search
             over w in [0, 1] minimizing the same forecasting task there): reps_cv = w*ybar, posterior_cv = b*z + w*(ybar - b*z)
Score: weighted MSE of forecast vs realized future mean (weights = future reps), as % improvement over 'league'.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
cov = cov.sort_values(["nfl_id_d", "season", "week" if "week" in cov else "game_id", "game_id", "play_id"])


def var_parts(d):
    g = d.groupby("nfl_id_d").res; means, n = g.mean(), g.size()
    s2 = d.res.var()
    t2 = max(means.var() - (s2 / n).mean(), 1e-4)
    return s2, t2


def run(y, ctx, sub=None, ks=(0, 5, 10, 20)):
    d0 = cov if sub is None else cov[sub(cov)]
    d0 = d0.dropna(subset=[y, "z_drive"] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    d0["order"] = d0.groupby("nfl_id_d").cumcount()
    out = {k: {"league": [0, 0], "reps_eb": [0, 0], "combine": [0, 0], "posterior": [0, 0], "reps_cv": [0, 0], "posterior_cv": [0, 0]} for k in ks}
    W = np.linspace(0, 1, 21)

    def tune(tr, k, b, with_prior):
        err = np.zeros(len(W)); tot = 0
        for pid, g in tr.groupby("nfl_id_d"):
            fut = g[g.order >= k]
            if len(fut) < 10 or k == 0:
                continue
            ybar = g[g.order < k].res.mean(); pr = b * g.z_drive.iat[0] if with_prior else 0.0
            err += len(fut) * (pr + W * (ybar - pr) - fut.res.mean()) ** 2; tot += len(fut)
        return W[np.argmin(err)] if tot else 0.0
    ncb = {k: 0 for k in ks}
    for c in sorted(d0.cls.dropna().unique()):
        tr, te = d0[d0.cls != c].copy(), d0[d0.cls == c].copy()
        m = smf.ols(f"{y} ~ {ctx}", data=tr).fit()
        tr["res"] = m.resid
        tr["order"] = tr.groupby("nfl_id_d").cumcount()
        te = te[te.route_ran.isin(tr.route_ran.unique())] if "route_ran" in ctx else te
        te["res"] = te[y] - m.predict(te)
        b = smf.ols("res ~ z_drive", data=tr).fit().params["z_drive"]
        s2, t2 = var_parts(tr)
        tr2 = tr.copy(); tr2["res"] = tr.res - b * tr.z_drive
        _, t2r = var_parts(tr2)
        wr_ = {k: tune(tr, k, b, False) for k in ks}; wp_ = {k: tune(tr, k, b, True) for k in ks}
        for pid, g in te.groupby("nfl_id_d"):
            z = g.z_drive.iat[0]
            for k in ks:
                fut = g[g.order >= k]
                if len(fut) < 10:
                    continue
                early = g[g.order < k].res
                ybar = early.mean() if k > 0 else 0.0
                f = {"league": 0.0,
                     "reps_eb": (k / (k + s2 / t2)) * ybar if k > 0 else 0.0,
                     "combine": b * z,
                     "posterior": (b * z / t2r + k * ybar / s2) / (1 / t2r + k / s2),
                     "reps_cv": wr_[k] * ybar, "posterior_cv": b * z + wp_[k] * (ybar - b * z)}
                tgt = fut.res.mean(); w = len(fut)
                for name, v in f.items():
                    out[k][name][0] += w * (v - tgt) ** 2; out[k][name][1] += w
                ncb[k] += 1
    print(f"\n{y} {'' if sub is None else '(zone reps)'}  [drive coefficient fitted out of class; CBs per k: {ncb}]")
    for k in ks:
        L = out[k]["league"][0] / out[k]["league"][1]
        row = {n_: (1 - (v[0] / v[1]) / L) * 100 for n_, v in out[k].items() if n_ != "league"}
        print(f"  k = {k:2d}: reps-only (tuned) {row['reps_cv']:+6.1f}% | combine-only {row['combine']:+6.1f}% | combine prior + reps (tuned) {row['posterior_cv']:+6.1f}%"
              f"   [untuned EB: reps {row['reps_eb']:+.0f}%, prior+reps {row['posterior']:+.0f}%]")


run("sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran)")
run("sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)")
run("sep_gain_break", "wr_skill + C(route_ran)", sub=lambda d: d.team_coverage_man_zone == "ZONE_COVERAGE")
run("sep_at_throw", "wr_skill + C(route_ran)", sub=lambda d: d.team_coverage_man_zone == "ZONE_COVERAGE")
