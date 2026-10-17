"""Step 6 (robust version): how much of the true between-CB difference in coverage does drive speed explain?
Method of moments: rep residuals from an OLS of the outcome on context (receiver skill, route, man/zone) with or without
drive speed; true between-CB variance = variance of CB mean residuals - mean sampling variance (residual var / reps).
Corner-level bootstrap (resample CBs) for a 90% interval of the explained share. Zone and man separately."""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(23)
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")


def between_var(d, y, rhs):
    m = smf.ols(f"{y} ~ {rhs}", data=d).fit()
    r = pd.Series(m.resid.values, index=d.index)
    g = r.groupby(d.nfl_id_d)
    means, n = g.mean(), g.size()
    return max(means.var() - (r.var() / n).mean(), 0.0)


def explained(d, y, ctx):
    v0, v1 = between_var(d, y, ctx), between_var(d, y, "z_drive + " + ctx)
    return (1 - v1 / v0) if v0 > 0 else np.nan, v0


for y, ctx in [("sep_gain_break", "wr_skill + C(route_ran)"), ("sep_at_throw", "wr_skill + C(route_ran)"), ("cpoe", "rec_target_prop")]:
    for cz in ["ZONE_COVERAGE", "MAN_COVERAGE", None]:
        d = cov if cz is None else cov[cov.team_coverage_man_zone == cz]
        c2 = ctx + ("" if cz else " + C(team_coverage_man_zone)")
        d = d.dropna(subset=[y, "z_drive"] + (["wr_skill"] if "wr_skill" in ctx else []))
        share, v0 = explained(d, y, c2)
        ids = d.nfl_id_d.unique(); boots = []
        for _ in range(300):
            pick = rng.choice(ids, len(ids), replace=True)
            bd = pd.concat([d[d.nfl_id_d == i].assign(nfl_id_d=f"{i}_{k}") for k, i in enumerate(pick)])
            try:
                s_, _ = explained(bd, y, c2)
                if np.isfinite(s_): boots.append(min(max(s_, -1), 1))
            except Exception:
                pass
        lo, hi = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
        print(f"{y:15s} {cz or 'all':14s} true between-CB SD {np.sqrt(v0):.3f} | share explained by drive speed {share:+.0%} [90% CI {lo:+.0%}, {hi:+.0%}] "
              f"(CBs {d.nfl_id_d.nunique()}, reps {len(d)})")
