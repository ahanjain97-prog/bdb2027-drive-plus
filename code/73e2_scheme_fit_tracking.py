"""Step 6b -- scheme fit with a TRACKING-defined team measure instead of shell labels.

Team-season zone-match rate = share of the team's DB coverage reps (all pool DBs, 41) that are zone + pre-break
alignment >= .7 (70's definition), computed LEAVING OUT the focal CB's own reps (so it is the team's habit, not his).
Requires >= 15 other reps. Test on all of the focal CB's reps: y ~ drive * team_rate_loo + context (player perm p).
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(107)
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
al = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
al = al[al.season_type.isin(["REG", "POST"]) & (al.nullified != "Y")].reset_index(drop=True)
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = al.idx_r.values, al.idx_d.values
vxr, vyr, sr, lr, pr = g(A, "vx"), g(A, "vy"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
vxd, vyd, sd = g(B, "vx"), g(B, "vy"), g(B, "s"); vv = V[A] & V[B]
T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
jb = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1).argmax(1)
w = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
pa = np.where(w.sum(1) >= 3, np.where(w, cosv, 0).sum(1) / np.maximum(w.sum(1), 1), np.nan)
al["zm"] = ((al.team_coverage_man_zone == "ZONE_COVERAGE") & (pa >= 0.7)).astype(float)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")[["game_id", "play_id", "nfl_id", "team_abbr"]].rename(columns={"nfl_id": "nfl_id_d", "team_abbr": "def_team"})
al = al.merge(pp, on=["game_id", "play_id", "nfl_id_d"], how="left")
team = al.groupby(["def_team", "season"]).zm.agg(["sum", "count"])
own = al.groupby(["def_team", "season", "nfl_id_d"]).zm.agg(["sum", "count"]).reset_index()
own = own.join(team, on=["def_team", "season"], rsuffix="_t")
own["n_other"] = own.count_t - own["count"]; own["team_zm_loo"] = (own.sum_t - own["sum"]) / own.n_other.replace(0, np.nan)
own = own[own.n_other >= 15]
print(f"DB reps {len(al)}, team-seasons {len(team)}, CB-team-seasons with >= 15 other reps {len(own)}; team zone-match rate (loo) "
      f"median {own.team_zm_loo.median():.2f}, IQR {own.team_zm_loo.quantile(.25):.2f}-{own.team_zm_loo.quantile(.75):.2f}")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet").merge(pp, on=["game_id", "play_id", "nfl_id_d"], how="left")
cov = cov.merge(own[["def_team", "season", "nfl_id_d", "team_zm_loo"]], on=["def_team", "season", "nfl_id_d"], how="inner")
v = Z["top"].reindex(cov.nfl_id_d.unique()); cov["c_top"] = cov.nfl_id_d.map((Z["top"] - v.mean()) / v.std())
cov["tz"] = (cov.team_zm_loo - cov.team_zm_loo.mean()) / cov.team_zm_loo.std()
print("does the team rate predict the focal CB's own zone-matching share? r",
      round(cov.groupby(["nfl_id_d", "season"]).agg(a=("cell", lambda s: (s == "zone-matching").mean()), b=("team_zm_loo", "first")).corr().iloc[0, 1], 2))
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
for y, d, ctx in [("db_closing", cov, "wr_skill + C(route_ran)"), ("sep_gain_break", cov, "wr_skill + C(route_ran)"),
                  ("sep_at_throw", cov, "wr_skill + C(route_ran)"), ("cpoe", cov[cov.targeted == 1], "rec_target_prop")]:
    d = d.dropna(subset=[y, "c_top", "tz"] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    f = f"{y} ~ c_top * tz + {ctx}"; m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues["c_top:tz"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").c_top; null = []
    for _ in range(400):
        dd = d.copy(); dd["c_top"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["c_top:tz"])
    p = (1 + (np.abs(null) >= abs(t0)).sum()) / 401
    print(f"  {y:14s} drive {m.params['c_top']:+.3f} (p {m.pvalues['c_top']:.3f}) | drive x team zone-match rate {m.params['c_top:tz']:+.3f} per SD (perm p {p:.3f}) | reps {len(d)}, CBs {d.nfl_id_d.nunique()}")
