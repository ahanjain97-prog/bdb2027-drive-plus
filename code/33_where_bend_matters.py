"""Where does bend matter most? Play-level, all DL rush snaps (REG + POST): outcome ~ hoop turn speed x condition.
Conditions: edge vs interior alignment; obvious passing down (3rd/4th and >= 7); late in a drive (drive play >= 7);
rusher fatigue (>= 5 consecutive snaps this drive); 4th quarter / OT; quarterback dropback that lasts long
(rusher still live at 2.5 s, i.e. the throw had not come yet) -- where a corner can actually be turned.
Drives are reconstructed from the play sequence in each game (possession change or half boundary starts a new drive).
Outcomes: penetration (max depth before the throw) and pressure. OLS, cluster-robust SE by player; hoop z-scored
across players. Reported: hoop slope in each condition level and the interaction p.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
gp = pd.read_parquet(f"{ROOT}/data/bend/game_bend_play.parquet")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")[["game_id", "play_id", "nfl_id", "possession_team", "quarter", "dropback_duration"]]
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
# drives from the union of plays seen in each game
plays = pp.drop_duplicates(["game_id", "play_id"]).sort_values(["game_id", "play_id"])[["game_id", "play_id", "possession_team", "quarter"]]
half = np.where(plays.quarter <= 2, 1, np.where(plays.quarter <= 4, 2, 3))
new = (plays.possession_team != plays.possession_team.shift()) | (plays.game_id != plays.game_id.shift()) | (pd.Series(half, index=plays.index) != pd.Series(half, index=plays.index).shift())
plays["drive"] = new.cumsum()
plays["drive_play"] = plays.groupby("drive").cumcount() + 1
d = gp[gp.nfl_position.isin(["DE", "OLB", "DT", "NT"]) & gp.lined_up_position.isin(["EDGE", "INTERIOR_LINE"]) &
       (gp.play_type == "DROPBACK") & gp.player_get_off.notna() & gp.season_type.isin(["REG", "POST"]) & (gp.nullified != "Y")].copy()
d = d.merge(plays[["game_id", "play_id", "drive", "drive_play"]], on=["game_id", "play_id"], how="left")
d = d.sort_values(["nfl_id", "drive", "play_id"])
d["consec"] = d.groupby(["nfl_id", "drive"]).cumcount() + 1
d = d.merge(pp[["game_id", "play_id", "nfl_id", "quarter"]], on=["game_id", "play_id", "nfl_id"], how="left")
dbd = pp.groupby(["game_id", "play_id"]).dropback_duration.max().rename("dropback_duration").reset_index()   # filled on OL rows only
d = d.merge(dbd, on=["game_id", "play_id"], how="left")
T = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_turns.parquet"); T = T[T.family == "hoop"]
hoop = T.groupby(["nfl_id", "event_id"]).apply(lambda x: np.average(x.sust_speed, weights=x.dur), include_groups=False).groupby("nfl_id").mean()
d["hoop"] = d.nfl_id.map(hoop)
d = d.dropna(subset=["hoop"])
hv = d.drop_duplicates("nfl_id").hoop
d["z_hoop"] = (d.hoop - hv.mean()) / hv.std()
d["ytg_b"] = pd.cut(d.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
d["cond_edge"] = (d.lined_up_position == "EDGE").astype(int)
d["cond_obvious_pass"] = ((d.down >= 3) & (d.yards_to_go >= 7)).astype(int)
d["cond_late_drive"] = (d.drive_play >= 7).astype(int)
d["cond_fatigue"] = (d.consec >= 5).astype(int)
d["cond_q4"] = (d.quarter >= 4).astype(int)
d["cond_long_dropback"] = np.where(d.dropback_duration.notna(), (d.dropback_duration >= 2.8).astype(float), np.nan)
print(f"rush snaps {len(d)}, rushers {d.nfl_id.nunique()}; condition shares: " +
      ", ".join(f"{c[5:]} {d[c].mean():.2f}" for c in d.columns if c.startswith("cond_")))
rows = []
for y in ["max_depth_live", "pressure"]:
    for c in [c for c in d.columns if c.startswith("cond_")]:
        dd = d.dropna(subset=[y, c, "blitzing", "down", "ytg_b", "lined_up_position"])
        f = f"{y} ~ z_hoop * {c} + C(lined_up_position) + C(blitzing) + C(down) + C(ytg_b)"
        if c == "cond_edge":
            f = f"{y} ~ z_hoop * {c} + C(blitzing) + C(down) + C(ytg_b)"
        mm = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.nfl_id)[0]})
        b0 = mm.params["z_hoop"]; bi = mm.params[f"z_hoop:{c}"]
        cov = mm.cov_params()
        se1 = np.sqrt(cov.loc["z_hoop", "z_hoop"] + cov.loc[f"z_hoop:{c}", f"z_hoop:{c}"] + 2 * cov.loc["z_hoop", f"z_hoop:{c}"])
        rows.append({"outcome": y, "condition": c[5:], "share": dd[c].mean(), "slope_when_no": b0, "p_no": mm.pvalues["z_hoop"],
                     "slope_when_yes": b0 + bi, "p_yes": 2 * (1 - __import__("scipy").stats.norm.cdf(abs((b0 + bi) / se1))),
                     "interaction_p": mm.pvalues[f"z_hoop:{c}"], "mean_y_yes": dd.loc[dd[c] == 1, y].mean()})
R = pd.DataFrame(rows)
R.to_csv(f"{ROOT}/results/33_where_bend_matters.csv", index=False)
pd.set_option("display.width", 220)
print("\nhoop turn speed effect (per SD) by condition: penetration in yards, pressure in probability")
print(R.round(4).to_string(index=False))
