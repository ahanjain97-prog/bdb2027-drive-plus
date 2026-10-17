"""EPA per target of his man on the registered wide reps (code/89), same estimator and standard as 89/98.
EPA already prices down, distance and field position. Also: winsorized EPA (big-play tails), success rate (EPA > 0),
yards per target, interceptions; by coverage cell and season; mean EPA per target by Drive+ third.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; rng = np.random.default_rng(401)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other")
T = W[W.targeted == 1].merge(pp[["game_id", "play_id", "nfl_id", "expected_points_added", "rec_yards", "pass_result"]].rename(columns={"nfl_id": "nfl_id_r"}),
                              on=["game_id", "play_id", "nfl_id_r"], how="left")
T["epa"] = T.expected_points_added; lo, hi = T.epa.quantile([.01, .99]); T["epa_w"] = T.epa.clip(lo, hi)
T["success"] = (T.epa > 0).astype(float); T["yds"] = T.rec_yards.fillna(0); T["int"] = (T.pass_result == "IN").astype(float)


def est(d, y, sc=1, n_perm=500, ctx="C(zone) + C(route_ran) + is_s", sign=-1):
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"; mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    if used.nfl_id_d.nunique() < 8: return "too few"
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]})
    t0 = sign * mod.fit().tvalues["dp"]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(sign * smf.ols(f, data=dd).fit().tvalues["dp"])
    p = (1 + (np.array(null) >= t0).sum()) / (n_perm + 1); b, se = m_.params["dp"] * sc, m_.bse["dp"] * sc; lo80, hi80 = b - 1.2816 * se, b + 1.2816 * se
    v = "supported" if (np.sign(b) == sign and p < .10 and (hi80 < 0 if sign < 0 else lo80 > 0)) else ("right sign" if np.sign(b) == sign else "wrong sign")
    return f"{b:+7.3f} [{lo80:+7.3f}, {hi80:+7.3f}]  110 vs 90 {2 * b:+7.2f}  p1 {p:.3f}  {v:10s} n {len(used)}, DBs {used.nfl_id_d.nunique()}"


print(f"wide targets {len(T)}, DBs {T.nfl_id_d.nunique()} | mean EPA/target {T.epa.mean():+.3f}, success {T.success.mean():.3f}, yds/target {T.yds.mean():.2f}, INT {T.int.sum():.0f}")
print("\n=== per 10 Drive+ (one-sided p in the direction good for the defense) ===")
for nm, y, sc, sg in [("EPA per target", "epa", 1, -1), ("EPA per target, winsorized 1/99%", "epa_w", 1, -1), ("success rate (EPA > 0), pts", "success", 100, -1),
                      ("yards per target", "yds", 1, -1), ("interception rate, pts", "int", 100, +1)]:
    print(f"  {nm:34s} {est(T, y, sc, sign=sg)}", flush=True)
print("\n=== EPA per target by coverage (per 10 Drive+) ===")
for c in ["zone-matching", "man-matching", "other"]:
    print(f"  {c:16s} {est(T[T.cell == c], 'epa', ctx='C(route_ran) + is_s')}", flush=True)
print(f"  {'all zone':16s} {est(T[T.zone == 1], 'epa', ctx='C(route_ran) + is_s')}", flush=True)
print("\n=== EPA per target by season (per 10 Drive+) ===")
for s in sorted(T.season.unique()):
    print(f"  {s}  {est(T[T.season == s], 'epa')}", flush=True)
q = pd.qcut(T.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp, 3, labels=["bottom", "middle", "top"]); T["third"] = T.nfl_id_d.map(q)
print("\n=== raw means by Drive+ third (DB thirds among targeted DBs) ===")
print(T.groupby("third").agg(targets=("epa", "size"), DBs=("nfl_id_d", "nunique"), epa=("epa", "mean"), success=("success", "mean"), comp=("completion", "mean"), yds=("yds", "mean")).round(3).to_string())
