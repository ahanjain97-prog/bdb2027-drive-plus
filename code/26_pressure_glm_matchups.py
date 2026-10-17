"""(1) Play-level pressure model: pressure ~ hoop turn speed (player covariate) + unit + context cells, logistic,
cluster-robust SE by player (uses every rush snap instead of player means). Also penetration as a check.
(2) How many plays contain BOTH a pool DL and a pool OL on opposite teams (true 1-v-1 trajectories)?"""
import warnings, contextlib, io
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/Users/ahanjain/bdb_ssl/code/24_hoop_robustness.py").read().split("ids = base.index")[0])
hs = cb2[["turn_speed"]].dropna()
d = ro.merge(hs, left_on="nfl_id", right_index=True).merge(base[["combine_weight"]], left_on="nfl_id", right_index=True)
d["z_hoop"] = (d.turn_speed - hs.turn_speed.mean()) / hs.turn_speed.std()
d["z_w"] = (d.combine_weight - d.combine_weight.mean()) / d.combine_weight.std()
d["edge"] = (d.unit == "EDGE").astype(int)
d["blitz"] = d.blitzing.astype(str)
for f in ["pressure ~ z_hoop + edge + C(blitz) + C(down) + C(ytg_b)", "pressure ~ z_hoop + z_w + edge + C(blitz) + C(down) + C(ytg_b)"]:
    m = smf.logit(f, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id)[0]})
    b, se, p = m.params["z_hoop"], m.bse["z_hoop"], m.pvalues["z_hoop"]
    base_rate = d.pressure.mean()
    print(f"{f}\n   n snaps {len(d)}, players {d.nfl_id.nunique()}: OR per SD hoop turn speed {np.exp(b):.3f} "
          f"[{np.exp(b - 1.96 * se):.3f}, {np.exp(b + 1.96 * se):.3f}], p {p:.4f}  (base pressure rate {base_rate:.3f})")
dd = d.dropna(subset=["max_depth_live"])
m = smf.ols("max_depth_live ~ z_hoop + z_w + edge + C(blitz) + C(down) + C(ytg_b)", data=dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.nfl_id)[0]})
print(f"penetration (yd) ~ hoop: b per SD {m.params['z_hoop']:.3f} yd (se {m.bse['z_hoop']:.3f}, p {m.pvalues['z_hoop']:.4f})")
# (2) matchups
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")[["game_id", "play_id", "nfl_id", "team_abbr", "possession_team"]]
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
x = pp.merge(pl, on="nfl_id")
x["role"] = np.where(x.nfl_position.isin(["T", "G", "C"]), "OL", np.where(x.nfl_position.isin(["DE", "OLB", "DT", "NT"]), "DL", "other"))
g = x.groupby(["game_id", "play_id"]).role.agg(lambda r: ("OL" in set(r)) and ("DL" in set(r)))
print(f"\nplays with a pool OL AND a pool DL (opposite teams by construction): {g.sum()} of {len(g)}")
