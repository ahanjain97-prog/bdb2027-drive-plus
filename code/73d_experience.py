"""Step 5 -- does experience close the gap? Drive-speed slope by NFL experience (season - draft class: 0 rookie, 1, 2).

(a) slope per experience year (no FE); (b) CB fixed effects + drive x experience (the drive main effect is absorbed;
the interaction says whether the drive slope changes as the same CBs gain experience); player-permutation p.
Caveat: experience 2 exists only for the 2023 class (2025 season).
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(83)
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
v = Z["top"].reindex(cov.nfl_id_d.unique()); cov["c_top"] = cov.nfl_id_d.map((Z["top"] - v.mean()) / v.std())
cov["exp"] = (cov.season - cov.cls).clip(lower=0)
print("reps by experience x class:\n", pd.crosstab(cov.exp, cov.cls).to_string())
print("CBs with reps in >= 2 experience years:", (cov.groupby("nfl_id_d").exp.nunique() >= 2).sum())
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
TASKS = [("zone-matching closing", cov[cov.cell == "zone-matching"], "db_closing", "wr_skill + C(route_ran)"),
         ("all reps sep gained at break", cov, "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
         ("zone-matching sep gained at break", cov[cov.cell == "zone-matching"], "sep_gain_break", "wr_skill + C(route_ran)"),
         ("CPOE (targeted)", cov[cov.targeted == 1], "cpoe", "rec_target_prop + C(team_coverage_man_zone)")]
for lab, d, y, ctx in TASKS:
    d = d.dropna(subset=[y, "c_top"] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    per = []
    for e in [0, 1, 2]:
        de = d[d.exp == e]
        if de.nfl_id_d.nunique() >= 8:
            m = smf.ols(f"{y} ~ c_top + {ctx}", data=de).fit(cov_type="cluster", cov_kwds={"groups": CL(de)})
            per.append(f"yr{e}: {m.params['c_top']:+.3f} (p {m.pvalues['c_top']:.3f}, reps {len(de)}, CBs {de.nfl_id_d.nunique()})")
    f = f"{y} ~ C(nfl_id_d) + c_top:exp + exp + {ctx}"
    t0 = smf.ols(f, data=d).fit().tvalues["c_top:exp"]; b0 = smf.ols(f, data=d).fit().params["c_top:exp"]
    vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").c_top; null = []
    for _ in range(300):
        dd = d.copy(); dd["c_top"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["c_top:exp"])
    p = (1 + (np.abs(null) >= abs(t0)).sum()) / 301
    print(f"\n{lab}:\n  " + " | ".join(per) + f"\n  within-CB (CB fixed effects): drive x experience {b0:+.3f} per year (perm p {p:.3f})")
