"""Outcome targets per player x data part, plus draft-time baseline covariates.

Targets use REG + POST plays that were not nullified by penalty. Parts:
  h0 / h1   = game-level random halves (hash of game_id, salted so it is unrelated to the SSL val split)
  2023/2024/2025 = seasons
  career    = player_career_successes, normalized per season since draft
Position groups pool similar roles; the subgroup is a baseline dummy.

Outputs data/targets.parquet (long: nfl_id, part, target, value, n) and data/baseline.parquet.
"""
import hashlib
import numpy as np
import pandas as pd

D = "/Users/ahanjain/bdb_ssl/data/raw"
OUT = "/Users/ahanjain/bdb_ssl/data"

GROUP = {"WR": "REC", "TE": "REC", "T": "OL", "G": "OL", "C": "OL", "DE": "DL", "OLB": "DL", "DT": "DL", "NT": "DL",
         "CB": "DB", "SS": "DB", "FS": "DB", "DB": "DB"}
SUB = {"WR": "WR", "TE": "TE", "T": "T", "G": "IOL", "C": "IOL", "DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL",
       "CB": "CB", "SS": "S", "FS": "S", "DB": "S"}


def half(g):
    return "h%d" % (int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2)


players = pd.read_parquet(f"{D}/players.parquet")
players["grp"] = players.nfl_position.map(GROUP)
players["sub"] = players.nfl_position.map(SUB)
games = pd.read_parquet(f"{D}/games.parquet")[["game_id", "season", "season_type"]]
pp = pd.read_parquet(f"{D}/player_play.parquet").merge(games, on="game_id").merge(
    players[["nfl_id", "grp", "sub"]], on="nfl_id")
pp = pp[(pp.season_type != "PRE") & (pp.play_nullified_by_penalty != "Y")].copy()
pp["half"] = pp.game_id.map(half)
pp["dropback"] = pp.pass_result.isin(["C", "I", "IN", "S", "R"])


def per_part(df, part_col):
    rows = []
    for (pid, part), g in df.groupby(["nfl_id", part_col]):
        grp = g.grp.iat[0]
        r = {"nfl_id": pid, "part": str(part)}
        if grp == "REC":
            routes = g[g.route_ran.notna()]
            nos = routes[(routes.route_ran != "SCREEN") & routes.separation_at_pass_forward.notna()]
            r.update(sep=(nos.separation_at_pass_forward.mean(), len(nos)),
                     tprr=(routes.target.astype(float).mean(), len(routes)),
                     yprr=(routes.rec_yards.fillna(0).mean(), len(routes)))
        elif grp == "OL":
            pb = g[g.peak_pressure_probability_allowed.notna()]
            pa = g[g.pressure_allowed.notna()]
            r.update(ppa=(pb.peak_pressure_probability_allowed.mean(), len(pb)),
                     pressure_allowed=(pa.pressure_allowed.astype(float).mean(), len(pa)))
        elif grp == "DL":
            rush = g[g.dropback & g.player_get_off.notna()]
            pressured = rush.time_to_pressure.notna() | (rush.sack.fillna(0) > 0)
            r.update(pressure_rate=(pressured.mean(), len(rush)),
                     get_off=(rush.player_get_off.mean(), len(rush)))
        elif grp == "DB":
            r.update(tackle_rate=((g.tackle.fillna(0) + g.assist.fillna(0)).clip(0, 1).mean(), len(g)))
        r["snaps_per_game"] = (len(g) / g.game_id.nunique(), g.game_id.nunique())
        rows.append(r)
    long = []
    for r in rows:
        for k, v in r.items():
            if k in ("nfl_id", "part"):
                continue
            long.append({"nfl_id": r["nfl_id"], "part": r["part"], "target": k, "value": v[0], "n": v[1]})
    return pd.DataFrame(long)


tg = pd.concat([per_part(pp, "half"), per_part(pp, "season")], ignore_index=True)
tg["part"] = tg.part.astype(str)

# career outcomes, per season since draft (data cover 2023-2025 seasons)
cs = pd.read_parquet(f"{D}/player_career_successes.parquet").merge(players[["nfl_id", "draft_year"]], on="nfl_id")
cs["seasons"] = 2026 - cs.draft_year
cs["snaps"] = cs.career_offensive_snaps + cs.career_defensive_snaps
car = pd.DataFrame({
    "nfl_id": cs.nfl_id,
    "log_snaps_per_season": np.log1p(cs.snaps / cs.seasons),
    "starts_per_season": cs.career_games_started / cs.seasons,
    "honors": ((cs.ap_all_pro_1st_team + cs.ap_all_pro_2nd_team + cs.pro_bowl_original_ballot) > 0).astype(float),
})
car = car.melt(id_vars="nfl_id", var_name="target", value_name="value")
car["part"] = "career"; car["n"] = 1
tg = pd.concat([tg, car], ignore_index=True)
tg.to_parquet(f"{OUT}/targets.parquet", index=False)

# draft-time baseline covariates
cr = pd.read_parquet(f"{D}/combine_results.parquet")
b = players.merge(cr.drop(columns=["draft_year"]), on="nfl_id", how="left")
b["udfa"] = b.draft_overall_pick.isna().astype(float)
b["log_pick"] = np.log(b.draft_overall_pick.fillna(300))
b["age_at_draft"] = (pd.to_datetime(b.draft_year.astype(str) + "-04-25") - pd.to_datetime(b.birth_date)).dt.days / 365.25
P4 = {"Southeastern Conference", "Big Ten Conference", "Big 12 Conference", "Atlantic Coast Conference", "Pac-12 Conference"}
b["power_conf"] = b.college_conference.isin(P4).astype(float)
BASE = ["log_pick", "udfa", "age_at_draft", "power_conf", "combine_height", "combine_weight", "hand_size", "arm_length",
        "wing_span", "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps",
        "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
keep = ["nfl_id", "display_name", "draft_year", "nfl_position", "grp", "sub"] + BASE
b[keep].to_parquet(f"{OUT}/baseline.parquet", index=False)

print(tg.groupby(["part", "target"]).agg(players=("nfl_id", "nunique"), med_n=("n", "median")).to_string())
print(b.groupby("grp", dropna=False).size().to_string())
print("baseline missing share:\n", b[BASE].isna().mean().round(2).to_string())
