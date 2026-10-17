"""How much separation before the throw do high-Drive+ DBs take away, and how many catches is that separation worth?
(1) separation at the throw on targets of his man, by Drive+ third: distance to HIM, and NGS nearest-defender separation
(2) league conversion rate: completion by separation at the throw, all 9.8K pool-receiver targets, raw and holding air yards
(3) implied catches from the separation gap vs the actual catch gap
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
T = W[W.targeted == 1].merge(pp[["game_id", "play_id", "nfl_id", "separation_at_pass_forward", "pass_length"]].rename(columns={"nfl_id": "nfl_id_r", "separation_at_pass_forward": "ngs_sep"}),
                              on=["game_id", "play_id", "nfl_id_r"], how="left")
q = pd.qcut(T.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp, 3, labels=["bottom", "middle", "top"]); T["third"] = T.nfl_id_d.map(q)
T["open_ngs"] = (T.ngs_sep >= 2).astype(float)
print("=== (1) targets of his man, by Drive+ third (raw means) ===")
print(T.groupby("third").agg(targets=("completion", "size"), DBs=("nfl_id_d", "nunique"), dist_to_him=("sep_at_throw", "mean"), ngs_nearest=("ngs_sep", "mean"),
                             open_2yd=("open", "mean"), comp=("completion", "mean")).round(2).to_string())
for y in ["sep_at_throw", "ngs_sep"]:
    m = smf.ols(f"{y} ~ dp + C(zone) + C(route_ran) + is_s", data=T).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(T.dropna(subset=[y]).nfl_id_d)[0]})
    print(f"  per 10 Drive+: {y:13s} {m.params['dp']:+.2f} yd (SE {m.bse['dp']:.2f})")

tg = pp[(pp.target == 1) & (pp.team_abbr == pp.possession_team) & pp.pass_result.isin(["C", "I", "IN"])].copy()
tg["comp"] = (tg.pass_result == "C").astype(int); tg["sep"] = tg.separation_at_pass_forward; tg = tg.dropna(subset=["sep", "pass_length"])
tg["bin"] = pd.cut(tg.sep, [0, 1, 2, 3, 4, 6, 99], labels=["<1", "1-2", "2-3", "3-4", "4-6", "6+"], right=False)
print(f"\n=== (2) league conversion: all pool-receiver targets (n {len(tg)}) ===")
print(tg.groupby("bin").agg(targets=("comp", "size"), completion=("comp", "mean"), air_yds=("pass_length", "mean")).round(3).to_string())
lg = smf.logit("comp ~ sep + I(sep**2) + pass_length + I(pass_length**2)", data=tg).fit(disp=0)
base = lg.predict(tg).mean(); minus = lg.predict(tg.assign(sep=(tg.sep - 1).clip(lower=0))).mean()
print(f"  holding air yards: 1 yd less separation at the throw = {100 * (minus - base):+.1f} pts completion (average over all targets)")
for nm, d in [("on the 502 wide targets", T.rename(columns={"ngs_sep": "sep"}).dropna(subset=["sep", "pass_length"]))]:
    for dy in [0.82, 1.64]:
        b0 = lg.predict(d).mean(); b1 = lg.predict(d.assign(sep=(d.sep - dy).clip(lower=0))).mean()
        print(f"  {nm}: {dy:.2f} yd less separation = {100 * (b1 - b0):+.1f} pts")
print("\n=== (3) per 10 Drive+ ===")
m = smf.ols("completion ~ dp + C(zone) + C(route_ran) + is_s", data=T).fit(); print(f"  actual completion {100 * m.params['dp']:+.1f} pts")

# ---------------- (4) mediation inside the 502 targets: how much of the Drive+ catch effect runs through separation before the throw?
rng = np.random.default_rng(611)
def est(d, extra, n_perm=500):
    d = d.dropna(subset=["completion", "dp", "sep_at_throw", "ngs_sep", "pass_length"]).copy()
    f = "completion ~ dp + C(zone) + C(route_ran) + is_s" + (f" + {extra}" if extra else ""); mod = smf.ols(f, data=d)
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]}); t0 = -mod.fit().tvalues["dp"]
    vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(-smf.ols(f, data=dd).fit().tvalues["dp"])
    p = (1 + (np.array(null) >= t0).sum()) / (n_perm + 1); b, se = 100 * m_.params["dp"], 100 * m_.bse["dp"]
    return b, f"{b:+6.2f} [{b - 1.2816 * se:+6.2f}, {b + 1.2816 * se:+6.2f}] p1 {p:.3f}  n {len(d)}"
T["d"] = T.sep_at_throw.clip(upper=12); T["d2"] = T.d ** 2; T["n_"] = T.ngs_sep.clip(upper=12); T["n2"] = T.n_ ** 2
print("\n=== (4) completion per 10 Drive+, holding separation at the throw fixed (same 499 targets) ===")
specs = [("no separation controls", ""), ("+ air yards", "pass_length"), ("+ NGS nearest-defender separation (what the 'before' model used)", "n_ + n2 + pass_length"),
         ("+ distance to HIM at the throw", "d + d2 + pass_length"), ("+ both", "d + d2 + n_ + n2 + pass_length")]
b0 = None
for nm, ex in specs:
    b, s = est(T, ex); b0 = b if b0 is None else b0
    print(f"  {nm:66s} {s}   ({100 * (1 - b / b0):.0f}% of the effect explained)", flush=True)
print("\n  by season, + distance to HIM + NGS separation + air yards:")
for sname in [2024, 2025]:
    d = T[T.season == sname]; ba, sa = est(d, ""); bb, sb = est(d, "d + d2 + n_ + n2 + pass_length")
    print(f"    {sname}: raw {sa}\n          holding separation {sb}", flush=True)
