"""Bend hypotheses: combine bend -> NFL movement and NFL performance for edge rushers, interior DL and OL.

Player-level, equal weight. Outcomes: REG + POST, plays not nullified. Movement: all game types.
Outcome adjustment: each play's outcome minus the leave-player-out mean of its context cell
  pass rush: alignment x blitz x down x distance bucket;  run: alignment x down x distance bucket;
  OL pass pro: alignment x down x formation;  run yards: formation x down x distance bucket.
Reported per test: n, Pearson r (permutation p, 5000 perms), Spearman, partial r given weight, partial r given
weight + stopwatch (40, 10-yd, vertical, broad; median-imputed + missing flags). Split-half reliability of the
game-side measure (halves of games) is shown as the ceiling.
"""
import hashlib, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cb = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_player.parquet")
gp = pd.read_parquet(f"{ROOT}/data/bend/game_bend_play.parquet")
gp["half"] = gp.game_id.map(lambda g: int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2)
gp["ytg_b"] = pd.cut(gp.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
out_ok = gp.season_type.isin(["REG", "POST"]) & (gp.nullified != "Y")
rng = np.random.default_rng(0)


def z(x):
    x = np.asarray(x, float); return (x - np.nanmean(x)) / np.nanstd(x)


# combine composites
cb["c_dl_bend"] = np.nanmean(np.column_stack([z(cb.c_hoop_peak_lat), z(cb.c_rushdrill_peak_lat)]), 1)
cb["c_dl_bend_L"] = np.nanmean(np.column_stack([z(cb.c_hoop_L_peak_lat), z(cb.c_rushdrill_L_peak_lat)]), 1)
cb["c_dl_bend_R"] = np.nanmean(np.column_stack([z(cb.c_hoop_R_peak_lat), z(cb.c_rushdrill_R_peak_lat)]), 1)
cb["c_dl_bend_asym"] = cb.c_dl_bend_L - cb.c_dl_bend_R
cb["c_dl_apex_speed"] = np.nanmean(np.column_stack([z(cb.c_hoop_apex_speed), z(cb.c_rushdrill_apex_speed)]), 1)
cb["c_pull_asym"] = z(cb.c_pull_L_peak_lat) - z(cb.c_pull_R_peak_lat)


def loo_adjust(df, ycol, cells):
    key = df[cells].astype(str).agg("|".join, axis=1)
    tot = df.groupby(key)[ycol].transform("sum"); cnt = df.groupby(key)[ycol].transform("count")
    pk = key + "#" + df.nfl_id.astype(str)
    ptot = df.groupby(pk)[ycol].transform("sum"); pcnt = df.groupby(pk)[ycol].transform("count")
    exp = (tot - ptot) / (cnt - pcnt)
    glob = df[ycol].mean()
    return df[ycol] - np.where((cnt - pcnt) >= 20, exp, glob)


def player_mean(df, col, min_n, by=None):
    g = df.groupby(["nfl_id"] + ([by] if by else []))[col]
    m, n = g.mean(), g.count()
    return m[n >= min_n]


def split_half(df, col, min_n, agg="mean"):
    g = df.groupby(["nfl_id", "half"])[col]
    w = pd.DataFrame({"v": g.agg(agg), "count": g.count()})
    w = w[w["count"] >= min_n // 2]["v"].unstack()
    w = w.dropna()
    if len(w) < 15:
        return np.nan
    r = np.corrcoef(w[0], w[1])[0, 1]
    return 2 * r / (1 + r)


STOP = ["forty", "ten_yd_split", "vertical", "broad_jump"]


def partial(x, y, Z):
    Z = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


def test(name, xs, ys, ceiling=np.nan, family="primary"):
    d = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).dropna()
    d = d[d.index.isin(base.index)]
    if len(d) < 15:
        return {"test": name, "n": len(d), "family": family}
    x, y = d.x.values, d.y.values
    r = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(5000)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / 5001
    w = base.loc[d.index, ["combine_weight"]].values.astype(float)
    S = base.loc[d.index, STOP].astype(float)
    Sf = np.column_stack([S.fillna(S.median()).values, S.isna().values.astype(float)])
    Sf = Sf[:, Sf.std(0) > 0]
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x)); boot.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"test": name, "family": family, "n": len(d), "r": r, "ci_lo": lo, "ci_hi": hi, "p_perm": p,
            "spearman": pd.Series(x).corr(pd.Series(y), method="spearman"),
            "r_partial_weight": partial(x, y, w), "r_partial_weight_stopwatch": partial(x, y, np.column_stack([w, Sf])),
            "game_side_reliability": ceiling}


res = []
# ======================= EDGE / interior DL pass rush
pos = gp.nfl_position
for unit, mask_pos, align in [("EDGE", pos.isin(["DE", "OLB"]), "EDGE"), ("IDL", pos.isin(["DT", "NT"]), "INTERIOR_LINE")]:
    rush = gp[mask_pos & (gp.play_type == "DROPBACK") & gp.player_get_off.notna() & (gp.lined_up_position == align)].copy()
    rush_o = rush[out_ok.loc[rush.index]].copy()
    rush_o["poe"] = loo_adjust(rush_o, "pressure", ["lined_up_position", "blitzing", "down", "ytg_b"])
    runs = gp[mask_pos & (gp.play_type == "RUN") & (gp.lined_up_position == align)].copy()
    runs_o = runs[out_ok.loc[runs.index]].copy()
    runs_o["tk_oe"] = loo_adjust(runs_o, "tackle_any", ["lined_up_position", "down", "ytg_b"])
    runs_o["tfl_oe"] = loo_adjust(runs_o, "tfl", ["lined_up_position", "down", "ytg_b"])
    # game bend capacity: 75th pct of per-play corner lateral accel on rush snaps (>= 50 snaps)
    gb = rush.groupby("nfl_id").peak_lat.quantile(.75)[rush.groupby("nfl_id").peak_lat.count() >= 50]
    gb_L = rush[rush.turn_dir == "L"].groupby("nfl_id").peak_lat.quantile(.75)
    gb_R = rush[rush.turn_dir == "R"].groupby("nfl_id").peak_lat.quantile(.75)
    nL = rush[rush.turn_dir == "L"].groupby("nfl_id").size(); nR = rush[rush.turn_dir == "R"].groupby("nfl_id").size()
    ok_side = (nL >= 25) & (nR >= 25)
    g_asym = (gb_L - gb_R)[ok_side[ok_side].index]
    rel_gb = split_half(rush, "peak_lat", 50, agg=lambda v: np.nanpercentile(v, 75))
    poe = player_mean(rush_o, "poe", 50); rel_poe = split_half(rush_o, "poe", 50)
    tk = player_mean(runs_o, "tk_oe", 40); rel_tk = split_half(runs_o, "tk_oe", 40)
    tfl = player_mean(runs_o, "tfl_oe", 40)
    # side-specific pressure over expected (pressure when the corner turn goes left vs right)
    poe_side = rush_o.groupby(["nfl_id", "turn_dir"]).poe.agg(["mean", "count"])
    poe_side = poe_side[poe_side["count"] >= 20]["mean"].unstack()
    poe_asym = (poe_side.get("L") - poe_side.get("R")).dropna() if "L" in poe_side and "R" in poe_side else pd.Series(dtype=float)
    # shed proxy on runs
    rel_shed = split_half(runs, "release_gain", 40)
    shed = player_mean(runs, "release_gain", 40)
    stall = player_mean(runs, "stall_speed", 40)
    c = cb
    res += [
        test(f"{unit}: combine bend -> NFL corner bend (p75 lateral accel)", c.c_dl_bend, gb, rel_gb),
        test(f"{unit}: combine bend -> pass-rush pressure over expected", c.c_dl_bend, poe, rel_poe),
        test(f"{unit}: combine bend -> run tackles over expected", c.c_dl_bend, tk, rel_tk),
        test(f"{unit}: combine bend -> run TFL over expected", c.c_dl_bend, tfl, np.nan),
        test(f"{unit}: combine bend asymmetry (L-R) -> NFL bend asymmetry (L-R)", c.c_dl_bend_asym, g_asym, np.nan),
        test(f"{unit}: combine bend asymmetry (L-R) -> pressure asymmetry (L-R turns)", c.c_dl_bend_asym, poe_asym, np.nan),
        test(f"{unit}: combine apex speed through turns -> pressure over expected", c.c_dl_apex_speed, poe, rel_poe, "secondary"),
        test(f"{unit}: hoop bend only -> pressure over expected", c.c_hoop_peak_lat, poe, rel_poe, "secondary"),
        test(f"{unit}: pass-rush-drill bend only -> pressure over expected", c.c_rushdrill_peak_lat, poe, rel_poe, "secondary"),
        test(f"{unit}: combine bend -> shed proxy (speed regained after stall at LOS, runs)", c.c_dl_bend, shed, rel_shed, "shed"),
        test(f"{unit}: shed proxy -> run tackles over expected (validates the proxy)", shed, tk, np.nan, "shed"),
        test(f"{unit}: NFL corner bend -> pressure over expected (same seasons)", gb, poe, np.nan, "chain"),
    ]
    # chain without shared noise: NFL bend in one half of games -> pressure in the other half
    gbh = rush.groupby(["nfl_id", "half"]).peak_lat.quantile(.75).unstack()
    ph = rush_o.groupby(["nfl_id", "half"]).poe.agg(["mean", "count"])
    ph = ph[ph["count"] >= 25]["mean"].unstack()
    res.append(test(f"{unit}: NFL bend (half A games) -> pressure over expected (half B games)",
                    pd.concat([gbh[0], gbh[1]]).groupby(level=0).first() if False else gbh[0], ph[1], np.nan, "chain"))
    res.append(test(f"{unit}: NFL bend (half B games) -> pressure over expected (half A games)", gbh[1], ph[0], np.nan, "chain"))

# ======================= OL
ol = gp[gp.nfl_position.isin(["T", "G", "C"])].copy()
olr = ol[ol.play_type == "RUN"]
pulls = olr[olr.pull]
g_pull = pulls.groupby("nfl_id").move_peak_lat.quantile(.75)[pulls.groupby("nfl_id").size() >= 8]
g_pull_speed = pulls.groupby("nfl_id").move_apex_speed.median()[pulls.groupby("nfl_id").size() >= 8]
gpL = pulls[pulls.move_turn_dir == "L"].groupby("nfl_id").move_peak_lat.quantile(.75)[pulls[pulls.move_turn_dir == "L"].groupby("nfl_id").size() >= 4]
gpR = pulls[pulls.move_turn_dir == "R"].groupby("nfl_id").move_peak_lat.quantile(.75)[pulls[pulls.move_turn_dir == "R"].groupby("nfl_id").size() >= 4]
climbs = olr[olr.climb]
t_climb = climbs.groupby("nfl_id").t_climb.median()[climbs.groupby("nfl_id").size() >= 20]
climb_bend = climbs.groupby("nfl_id").move_peak_lat.quantile(.75)[climbs.groupby("nfl_id").size() >= 20]
rel_tclimb = split_half(climbs, "t_climb", 20, agg="median")
olr_o = olr[out_ok.loc[olr.index]].copy()
olr_o["yds_oe"] = loo_adjust(olr_o, "yards_gained", ["offense_formation", "down", "ytg_b"])
yds_pull = player_mean(olr_o[olr_o.pull], "yds_oe", 8)
yds_climb = player_mean(olr_o[olr_o.climb], "yds_oe", 20)
olp = ol[(ol.play_type == "DROPBACK") & ol.peak_pressure_probability_allowed.notna()]
olp_o = olp[out_ok.loc[olp.index]].copy()
olp_o["ppa_oe"] = loo_adjust(olp_o, "peak_pressure_probability_allowed", ["lined_up_position", "down", "offense_formation"])
olp_o["pa_oe"] = loo_adjust(olp_o.assign(pa=olp_o.pressure_allowed.astype(float)), "pa", ["lined_up_position", "down", "offense_formation"])
ppa = player_mean(olp_o, "ppa_oe", 100); rel_ppa = split_half(olp_o, "ppa_oe", 100)
pa = player_mean(olp_o, "pa_oe", 100)
c = cb
res += [
    test("OL: combine pull bend -> NFL pull bend (p75 lateral accel on pulls)", c.c_pull_peak_lat, g_pull, np.nan),
    test("OL: combine pull apex speed -> NFL pull apex speed", c.c_pull_apex_speed, g_pull_speed, np.nan),
    test("OL: combine pull bend asymmetry -> NFL pull bend asymmetry", c.c_pull_asym, (gpL - gpR).dropna(), np.nan),
    test("OL: combine pull bend -> time to reach 2nd level on climbs (lower = faster)", c.c_pull_peak_lat, t_climb, rel_tclimb),
    test("OL: combine pull bend -> bend while climbing", c.c_pull_peak_lat, climb_bend, np.nan),
    test("OL: combine pull bend -> run yards over expected on his pulls", c.c_pull_peak_lat, yds_pull, np.nan),
    test("OL: combine pull bend -> run yards over expected on his climbs", c.c_pull_peak_lat, yds_climb, np.nan),
    test("OL: combine pull bend -> pass-pro peak pressure prob over expected", c.c_pull_peak_lat, ppa, rel_ppa),
    test("OL: combine pull bend -> pressures allowed over expected", c.c_pull_peak_lat, pa, np.nan),
    test("OL: combine pull apex speed -> time to 2nd level on climbs", c.c_pull_apex_speed, t_climb, rel_tclimb, "secondary"),
    test("OL: combine pull apex speed -> pass-pro peak pressure prob over expected", c.c_pull_apex_speed, ppa, rel_ppa, "secondary"),
]
R = pd.DataFrame(res)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p)
    adj = np.empty(n); adj[o] = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * p[o]))
    return adj


m_ = R.family.eq("primary") & R.p_perm.notna()
R.loc[m_, "p_holm_primary"] = holm(R.loc[m_, "p_perm"])
R.to_csv(f"{ROOT}/results/22_bend_tests.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 80)
cols = ["test", "family", "n", "r", "ci_lo", "ci_hi", "p_perm", "p_holm_primary", "spearman", "r_partial_weight",
        "r_partial_weight_stopwatch", "game_side_reliability"]
print(R[cols].round(3).to_string(index=False))
