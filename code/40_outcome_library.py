"""Phase 1a: player-level NFL outcome library by position, context-adjusted, with split-half reliability.

REG + POST, plays not nullified. "_oe" = play value minus the leave-player-out mean of its context cell (other
players only), averaged per player. Reliability = Spearman-Brown corrected correlation between the two random halves
of games. Minimum plays per player per outcome given in MIN.

WR/TE (route rows, screens excluded): separation at the throw over expected -- all, vs man, vs zone, vs press
  (cushion <= 2 yd), by route family (in-breaking, out-breaking, stop, vertical); targets per route; YAC over the
  NGS expectation per catch.
DL (DE/OLB/DT/NT rushing from EDGE / INTERIOR_LINE): penetration, depth at 2 s, pressure, get-off (lower = better),
  run TFL, run penetration -- separately for edge and interior alignments.
OL: peak pressure probability allowed, pressure allowed (by alignment group T vs G/C).
DB (CB/SS/FS/DB): tackles on runs / passes over expected; downhill speed after the handoff when in the box.
Rep-based outcomes (coverage, tackling, head-to-head trench) are added by 41/42 and the bend scripts.
Output data/outcomes/player_outcomes.parquet (long: nfl_id, outcome, value, n, half0, half1) + reliability table.
"""
import hashlib, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
os.makedirs(f"{ROOT}/data/outcomes", exist_ok=True)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
g = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season_type"]]
d = pp.merge(pl, on="nfl_id").merge(g, on="game_id")
d = d[(d.season_type != "PRE") & (d.play_nullified_by_penalty != "Y")].copy()
d["half"] = d.game_id.map(lambda x: int(hashlib.md5(f"half-{x}".encode()).hexdigest(), 16) % 2)
d["dropback"] = d.pass_result.isin(["C", "I", "IN", "S", "R"])
d["ytg_b"] = pd.cut(d.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
for c in ["target", "pressure_allowed", "blitzing", "in_motion_at_ball_snap"]:
    d[c] = d[c].astype("object").map({True: 1.0, False: 0.0}).astype(float)
gb = pd.read_parquet(f"{ROOT}/data/bend/game_bend_play.parquet")[["game_id", "play_id", "nfl_id", "max_depth_live", "depth_2s", "depth_1p5s"]]
d = d.merge(gb, on=["game_id", "play_id", "nfl_id"], how="left")


def loo(df, y, cells):
    key = df[cells].astype(str).agg("|".join, axis=1)
    ok = df[y].notna()
    tot = df[y].where(ok, 0).groupby(key).transform("sum"); cnt = ok.groupby(key).transform("sum")
    pk = key + "#" + df.nfl_id.astype(str)
    ptot = df[y].where(ok, 0).groupby(pk).transform("sum"); pcnt = ok.groupby(pk).transform("sum")
    exp = np.where((cnt - pcnt) >= 20, (tot - ptot) / (cnt - pcnt).clip(lower=1), df[y].mean())
    return (df[y] - exp).where(ok)


rows = []


def add(name, df, col, min_n, agg="mean"):
    df = df[df[col].notna()]
    g_ = df.groupby("nfl_id")[col]
    val, n = g_.agg(agg), g_.size()
    h = df.groupby(["nfl_id", "half"])[col].agg(agg).unstack()
    hn = df.groupby(["nfl_id", "half"])[col].size().unstack()
    keep = n[n >= min_n].index
    for pid in keep:
        rows.append({"nfl_id": pid, "outcome": name, "value": val[pid], "n": n[pid],
                     "half0": h.at[pid, 0] if (0 in h.columns and hn.at[pid, 0] >= min_n // 2) else np.nan,
                     "half1": h.at[pid, 1] if (1 in h.columns and hn.at[pid, 1] >= min_n // 2) else np.nan})


# ---------------- WR / TE
r = d[d.nfl_position.isin(["WR", "TE"]) & d.route_ran.notna() & (d.route_ran != "SCREEN")].copy()
FAM = {"SLANT": "in", "IN": "in", "CROSS": "in", "POST": "in", "ANGLE": "in", "OUT": "out", "CORNER": "out", "FLAT": "out",
       "WHEEL": "out", "HITCH": "stop", "GO": "vertical"}
r["fam"] = r.route_ran.map(FAM)
r["press"] = (r.cushion <= 2.0).astype(float).where(r.cushion.notna())
r["sep_oe"] = loo(r, "separation_at_pass_forward", ["route_ran", "lined_up_position", "team_coverage_man_zone", "press"])
r["tgt_oe"] = loo(r, "target", ["route_ran", "lined_up_position", "team_coverage_man_zone"])
r["yac_oe_raw"] = r.yards_after_catch - r.expected_yards_after_catch
add("REC sep_oe all", r, "sep_oe", 60)
add("REC sep_oe vs man", r[r.team_coverage_man_zone == "MAN_COVERAGE"], "sep_oe", 25)
add("REC sep_oe vs zone", r[r.team_coverage_man_zone == "ZONE_COVERAGE"], "sep_oe", 40)
add("REC sep_oe vs press", r[r.press == 1], "sep_oe", 20)
for f in ["in", "out", "stop", "vertical"]:
    add(f"REC sep_oe {f}-breaking" if f in ("in", "out") else f"REC sep_oe {f}", r[r.fam == f], "sep_oe", 15)
add("REC targets per route oe", r, "tgt_oe", 80)
add("REC YAC over NGS expected", r[r.yac_oe_raw.notna()], "yac_oe_raw", 15)

# ---------------- DL
dl = d[d.nfl_position.isin(["DE", "OLB", "DT", "NT"]) & d.lined_up_position.isin(["EDGE", "INTERIOR_LINE"])].copy()
rush = dl[dl.dropback & dl.player_get_off.notna()].copy()
rush["pressure"] = (rush.time_to_pressure.notna() | (rush.sack.fillna(0) > 0)).astype(float)
cells = ["lined_up_position", "blitzing", "down", "ytg_b"]
for y in ["max_depth_live", "depth_2s", "pressure"]:
    rush[y + "_oe"] = loo(rush, y, cells)
rush["get_off_oe"] = loo(rush, "player_get_off", cells)
runs = dl[~dl.dropback].copy()
runs["tfl"] = pd.to_numeric(runs.tackle_for_loss, errors="coerce").fillna(0).clip(0, 1)
runs["tk"] = ((runs.tackle.fillna(0) + runs.assist.fillna(0)) > 0).astype(float)
for y in ["tfl", "tk", "depth_1p5s"]:
    runs[y + "_oe"] = loo(runs, y, ["lined_up_position", "down", "ytg_b"])
for al, tag in [("EDGE", "EDGE"), ("INTERIOR_LINE", "IDL")]:
    ru, rn = rush[rush.lined_up_position == al], runs[runs.lined_up_position == al]
    add(f"{tag} penetration_oe", ru, "max_depth_live_oe", 50)
    add(f"{tag} depth2s_oe", ru, "depth_2s_oe", 50)
    add(f"{tag} pressure_oe", ru, "pressure_oe", 50)
    add(f"{tag} get_off_oe (lower=better)", ru, "get_off_oe", 50)
    add(f"{tag} run TFL_oe", rn, "tfl_oe", 40)
    add(f"{tag} run tackles_oe", rn, "tk_oe", 40)
    add(f"{tag} run penetration_oe", rn, "depth_1p5s_oe", 40)

# ---------------- OL
ol = d[d.nfl_position.isin(["T", "G", "C"]) & d.peak_pressure_probability_allowed.notna()].copy()
ol["algrp"] = np.where(ol.lined_up_position == "T", "T", "IOL")
ol["ppa_oe"] = loo(ol, "peak_pressure_probability_allowed", ["lined_up_position", "down", "offense_formation"])
ol["pa_oe"] = loo(ol, "pressure_allowed", ["lined_up_position", "down", "offense_formation"])
for grp_ in ["T", "IOL"]:
    add(f"{grp_} pressure prob allowed_oe (lower=better)", ol[ol.algrp == grp_], "ppa_oe", 100)
    add(f"{grp_} pressures allowed_oe (lower=better)", ol[ol.algrp == grp_], "pa_oe", 100)

# ---------------- DB
dbp = d[d.nfl_position.isin(["CB", "SS", "FS", "DB"])].copy()
dbp["tk"] = ((dbp.tackle.fillna(0) + dbp.assist.fillna(0)) > 0).astype(float)
dbp["tk_oe"] = loo(dbp, "tk", ["lined_up_position", "dropback", "down"])
dbp["unit"] = np.where(dbp.nfl_position == "CB", "CB", "S")
for u in ["CB", "S"]:
    add(f"{u} run tackles_oe", dbp[(dbp.unit == u) & ~dbp.dropback], "tk_oe", 60)
    add(f"{u} pass tackles_oe", dbp[(dbp.unit == u) & dbp.dropback], "tk_oe", 80)

O = pd.DataFrame(rows)
O.to_parquet(f"{ROOT}/data/outcomes/player_outcomes.parquet", index=False)


def rel(x):
    w = x[["half0", "half1"]].dropna()
    if len(w) < 12:
        return pd.Series({"players": x.nfl_id.nunique(), "split_half_n": len(w), "reliability": np.nan})
    rr = np.corrcoef(w.half0, w.half1)[0, 1]
    return pd.Series({"players": x.nfl_id.nunique(), "split_half_n": len(w), "reliability": 2 * rr / (1 + rr)})


R = O.groupby("outcome").apply(rel, include_groups=False)
R.to_csv(f"{ROOT}/results/40_outcome_reliability.csv")
pd.set_option("display.width", 200)
print(R.round(2).to_string())
