"""How can the after-throw catch effect be ~0 in 2024 and large in 2025?
(1) catches above / below expectation by Drive+ third and season (expected = league xComp with separation + throw context, code/98)
(2) receiver hands: receiver's catch rate on all his OTHER targets (leave-one-out) as a control; is it unbalanced in 2025?
(3) by throw depth within season
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; rng = np.random.default_rng(709)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
T = pd.read_parquet(f"{ROOT}/data/outcomes/98_wide_targets_cpoe.parquet")
q = pd.qcut(T.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp, 3, labels=["bottom", "middle", "top"]); T["third"] = T.nfl_id_d.map(q)
# receiver hands: leave-one-out catch rate on all his targets in the data (any defender)
tg = pp[(pp.target == 1) & (pp.team_abbr == pp.possession_team) & pp.pass_result.isin(["C", "I", "IN"])][["game_id", "play_id", "nfl_id", "pass_result"]].copy()
tg["c"] = (tg.pass_result == "C").astype(int); s = tg.groupby("nfl_id").c.agg(["sum", "count"])
T = T.merge(tg[["game_id", "play_id", "nfl_id", "c"]].rename(columns={"nfl_id": "nfl_id_r"}), on=["game_id", "play_id", "nfl_id_r"], how="left")
T["rcv_n"] = T.nfl_id_r.map(s["count"]) - 1; T["hands"] = (T.nfl_id_r.map(s["sum"]) - T.c) / T.rcv_n.replace(0, np.nan)
T["hands_c"] = T.hands - T.hands.mean()

print("=== (1) catches vs expectation (expected = league xComp given separation + throw context) ===")
for sname in [2024, 2025]:
    d = T[T.season == sname]
    g = d.groupby("third").agg(targets=("completion", "size"), DBs=("nfl_id_d", "nunique"), caught=("completion", "sum"), expected=("xc_sep", "sum"), hands=("hands", "mean"))
    g["above_exp"] = g.caught - g.expected; g["per_100"] = 100 * g.above_exp / g.targets
    print(f"  -- {sname}\n" + g.round(1).to_string())

def est(d, y, extra=""):
    d = d.dropna(subset=[y, "dp", "hands_c"]).copy(); f = f"{y} ~ dp + C(zone) + C(route_ran) + is_s" + (f" + {extra}" if extra else ""); mod = smf.ols(f, data=d)
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]}); t0 = -mod.fit().tvalues["dp"]
    vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(500):
        dd = d.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(-smf.ols(f, data=dd).fit().tvalues["dp"])
    p = (1 + (np.array(null) >= t0).sum()) / 501; b, se = 100 * m_.params["dp"], 100 * m_.bse["dp"]
    return f"{b:+6.2f} [{b - 1.2816 * se:+6.2f}, {b + 1.2816 * se:+6.2f}] p1 {p:.3f} n {len(d)}"

print("\n=== (2) receiver hands ===")
for sname in [2024, 2025]:
    d = T[T.season == sname].dropna(subset=["hands"])
    m = smf.ols("hands ~ dp + C(zone) + C(route_ran) + is_s", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]})
    print(f"  {sname}: receivers' catch rate on their other targets vs Drive+: {100 * m.params['dp']:+.2f} pts per 10 (SE {100 * m.bse['dp']:.2f}); mean {d.hands.mean():.3f}")
    print(f"        after-throw (completion - xComp_sep):            {est(d, 'cpoe_sep')}")
    print(f"        after-throw, holding receiver hands fixed:       {est(d, 'cpoe_sep', 'hands_c')}", flush=True)

print("\n=== (3) after-throw effect by throw depth (air yards) within season ===")
T["deep"] = np.where(T.pass_length >= 15, "15+ air yds", "< 15 air yds")
for sname in [2024, 2025]:
    for dpth in ["< 15 air yds", "15+ air yds"]:
        d = T[(T.season == sname) & (T.deep == dpth)]
        print(f"  {sname} {dpth:13s} share {len(d) / (T.season == sname).sum():.2f}: {est(d, 'cpoe_sep')}", flush=True)
