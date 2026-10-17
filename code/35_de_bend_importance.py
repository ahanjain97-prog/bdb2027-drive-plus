"""How important is bend for edge rushers (DE/OLB aligned EDGE)?

A) Combine traits vs NFL rush outcomes (player level, >= 50 REG+POST edge rush snaps): hoop turn speed, hoop peak
   lean, pass-rush-drill sustained lean, 40, 10-yd split, vertical, broad, weight, arm length. Univariate r, partial r
   given weight, and r against the corner-turn rate in head-to-head reps vs pool tackles.
B) NFL skills vs NFL results, cross-fitted by halves of games: rusher's corner bend (p75 lateral accel), arc speed,
   get-off (median), speed at 1 s, measured in half A -> pressure / penetration over expected in half B (and B -> A,
   averaged). This is "how much does bend matter once in the league", free of the shared-noise circularity.
C) Variance view: share of between-rusher variance in each half-B outcome explained by each half-A skill alone and by
   all together (in-sample OLS R^2 vs CV R^2).
"""
import hashlib, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cb = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_player.parquet")
gp = pd.read_parquet(f"{ROOT}/data/bend/game_bend_play.parquet")
gp["half"] = gp.game_id.map(lambda g: int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2)
gp["ytg_b"] = pd.cut(gp.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
T = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_turns.parquet"); T = T[T.family == "hoop"]
hoop = T.groupby(["nfl_id", "event_id"]).apply(lambda d: np.average(d.sust_speed, weights=d.dur), include_groups=False).groupby("nfl_id").mean()
rush = gp[gp.nfl_position.isin(["DE", "OLB"]) & (gp.lined_up_position == "EDGE") & (gp.play_type == "DROPBACK") & gp.player_get_off.notna()].copy()
ro = rush[rush.season_type.isin(["REG", "POST"]) & (rush.nullified != "Y")].copy()


def loo_adjust(df, ycol, cells):
    key = df[cells].astype(str).agg("|".join, axis=1)
    tot = df.groupby(key)[ycol].transform("sum"); cnt = df.groupby(key)[ycol].transform("count")
    pk = key + "#" + df.nfl_id.astype(str)
    ptot = df.groupby(pk)[ycol].transform("sum"); pcnt = df.groupby(pk)[ycol].transform("count")
    return df[ycol] - np.where((cnt - pcnt) >= 20, (tot - ptot) / (cnt - pcnt), df[ycol].mean())


ro["poe"] = loo_adjust(ro, "pressure", ["blitzing", "down", "ytg_b"])
rd = ro.dropna(subset=["max_depth_live"]).copy(); rd["pen_oe"] = loo_adjust(rd, "max_depth_live", ["blitzing", "down", "ytg_b"])
ro["pen_oe"] = rd.pen_oe.reindex(ro.index)
n = ro.groupby("nfl_id").size()
out = pd.DataFrame({"pressure_oe": ro.groupby("nfl_id").poe.mean(), "penetration_oe": ro.groupby("nfl_id").pen_oe.mean()})[n >= 50]
rep = pd.read_parquet(f"{ROOT}/data/bend/matchups.parquet")
te = rep[rep.primary & (rep.ol_align == "T") & (rep.dl_align == "EDGE") & rep.season_type.isin(["REG", "POST"])]
cr = te.groupby("dl_id").corner_turned.agg(["mean", "size"])
out["corner_rate_vs_T"] = cr["mean"].where(cr["size"] >= 15)
trait = pd.DataFrame({"hoop_turn_speed": hoop, "hoop_peak_lean": cb.c_hoop_peak_lat, "rushdrill_sust_lean": cb.c_rushdrill_sust_lat,
                      "forty (lower=faster)": base.forty, "ten_yd (lower=faster)": base.ten_yd_split, "vertical": base.vertical,
                      "broad_jump": base.broad_jump, "weight": base.combine_weight, "arm_length": base.arm_length})
print(f"A) edge rushers with >= 50 REG+POST rush snaps: {len(out)}; with hoop data {out.index.isin(hoop.index).sum()}")
rows = []
for t in trait.columns:
    for o_ in out.columns:
        d = pd.concat([trait[t], out[o_], base.combine_weight.rename("w")], axis=1).reindex(out.index).dropna()
        if len(d) < 15:
            continue
        r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
        if t != "weight":
            Z = np.column_stack([np.ones(len(d)), d.w])
            rx = d.iloc[:, 0] - Z @ np.linalg.lstsq(Z, d.iloc[:, 0], rcond=None)[0]; ry = d.iloc[:, 1] - Z @ np.linalg.lstsq(Z, d.iloc[:, 1], rcond=None)[0]
            pr, pp_ = stats.pearsonr(rx, ry)
        else:
            pr, pp_ = np.nan, np.nan
        rows.append({"trait": t, "outcome": o_, "n": len(d), "r": r, "p": p, "r_given_weight": pr, "p_given_weight": pp_})
A = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(A.pivot_table(index="trait", columns="outcome", values="r").round(2).to_string())
print("\n p-values:"); print(A.pivot_table(index="trait", columns="outcome", values="p").round(3).to_string())
print("\n r given weight:"); print(A.pivot_table(index="trait", columns="outcome", values="r_given_weight").round(2).to_string())
A.to_csv(f"{ROOT}/results/35A_de_combine_traits.csv", index=False)

# ---------------- B) NFL skills (half A) -> NFL results (half B), both directions
i10 = None
rush["speed_1s"] = np.nan
def skills(h):
    r_ = rush[rush.half == h]; g = r_.groupby("nfl_id"); nn = g.size()
    S = pd.DataFrame({"corner_bend_p75": g.peak_lat.quantile(.75), "arc_speed": g.sust_speed.mean(),
                      "get_off_median (lower=faster)": g.player_get_off.median(), "apex_speed": g.apex_speed.mean()})
    return S[nn >= 25]
def results(h):
    o = ro[ro.half == h]; g = o.groupby("nfl_id"); nn = g.size()
    return pd.DataFrame({"pressure_oe": g.poe.mean(), "penetration_oe": g.pen_oe.mean()})[nn >= 25]
rows = []
for a, b in [(0, 1), (1, 0)]:
    S, Rr = skills(a), results(b)
    for s_ in S.columns:
        for o_ in Rr.columns:
            d = pd.concat([S[s_], Rr[o_]], axis=1).dropna()
            r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
            rows.append({"dir": f"{a}->{b}", "skill": s_, "outcome": o_, "n": len(d), "r": r, "p": p})
B = pd.DataFrame(rows)
Bm = B.groupby(["skill", "outcome"]).agg(r_mean=("r", "mean"), r_AtoB=("r", "first"), r_BtoA=("r", "last"), n=("n", "min"),
                                         p_max=("p", "max")).reset_index()
print("\nB) NFL skill in one half of games -> result in the other half (edge rushers):")
print(Bm.round(3).to_string(index=False))
B.to_csv(f"{ROOT}/results/35B_de_nfl_skills.csv", index=False)

# ---------------- C) variance explained in half-B pressure / penetration by half-A skills (CV R^2, 5-fold, both directions)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
print("\nC) CV R^2 of half-B outcome from half-A skills (mean of both directions):")
for o_ in ["pressure_oe", "penetration_oe"]:
    res = {}
    for cols in [["corner_bend_p75"], ["arc_speed"], ["get_off_median (lower=faster)"], ["apex_speed"],
                 ["corner_bend_p75", "arc_speed", "get_off_median (lower=faster)", "apex_speed"]]:
        vals = []
        for a, b in [(0, 1), (1, 0)]:
            d = pd.concat([skills(a)[cols], results(b)[o_]], axis=1).dropna()
            pred = cross_val_predict(LinearRegression(), d[cols].values, d[o_].values, cv=KFold(5, shuffle=True, random_state=0))
            vals.append(1 - ((d[o_] - pred) ** 2).sum() / ((d[o_] - d[o_].mean()) ** 2).sum())
        res[" + ".join(c.split(" ")[0] for c in cols)] = np.mean(vals)
    print(f"  {o_}: " + "; ".join(f"{k}: {v:+.3f}" for k, v in res.items()))
