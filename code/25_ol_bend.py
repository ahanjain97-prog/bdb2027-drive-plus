"""OL: combine bend / redirect -> NFL pulls, climb blocks, pass protection. Same design as 23 (matrix + both-halves
replication filter + BH), tackle vs interior unit adjustment.
Combine: pull drills (peak / sustained lateral accel, apex / sustained speed through the turn, radius, retention,
turn time), five-yard wave (slide & shuffle) and mirror drills (mean speed, time, peak lateral accel, number of
direction changes per second).
Game: pulls (arc lateral accel p75, arc speed), climbs (time to 2 yd past the LOS, arc speed / lateral accel while
climbing), pass sets (max lateral speed 0-1.5 s, set depth, peak lateral accel); outcomes over expected: peak pressure
probability allowed, pressures allowed, run yards on his pulls / climbs.
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
out_ok = gp.season_type.isin(["REG", "POST"]) & (gp.nullified != "Y")
ol = gp[gp.nfl_position.isin(["T", "G", "C"])].copy()

# ---- combine: wave + mirror drills (lateral redirect)
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
w = ct[(ct.entity_type == "PLAYER") & ct.drill_name.isin(["FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", "PASS_PRO_MIRROR_DRILL", "PASS_PRO-MIRROR_DRILL"])]
w = w.sort_values(["event_id", "time"])
rows = []
for eid, g in w.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; s = g.s.values.astype(float)
    th = np.radians(g.dir.values.astype(float)); vx, vy = s * np.sin(th), s * np.cos(th)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    lat = pd.Series(np.where(s > 0.5, np.abs(vx * ay - vy * ax) / np.maximum(s, 1e-6), 0)).rolling(3, center=True, min_periods=1).mean().values
    on = int(np.argmax(s >= 1.0))
    # direction changes: sign flips of the dominant-axis velocity while moving
    ax_ = vx if np.abs(vx).sum() > np.abs(vy).sum() else vy
    flips = ((np.sign(ax_[1:]) != np.sign(ax_[:-1])) & (s[1:] > 0.3)).sum()
    fam = "wave" if "WAVE" in g.drill_name.iat[0] else "mirror"
    rows.append({"nfl_id": g.nfl_id.iat[0], "fam": fam, "mean_speed": s[on:].mean(), "time": t[-1] - t[on],
                 "peak_lat": lat.max(), "sust_lat": lat[s > 1].mean() if (s > 1).any() else np.nan,
                 "redirects_per_s": flips / max(t[-1] - t[on], 0.5)})
WM = pd.DataFrame(rows).groupby(["nfl_id", "fam"]).mean().unstack("fam")
WM.columns = [f"c_{f}_{m}" for m, f in WM.columns]
C = cb[[c for c in cb.columns if c.startswith("c_pull_") and "_L_" not in c and "_R_" not in c and "asym" not in c and not c.endswith("lean_deg")]].join(WM, how="outer")


def loo_adjust(df, ycol, cells):
    key = df[cells].astype(str).agg("|".join, axis=1)
    tot = df.groupby(key)[ycol].transform("sum"); cnt = df.groupby(key)[ycol].transform("count")
    pk = key + "#" + df.nfl_id.astype(str)
    ptot = df.groupby(pk)[ycol].transform("sum"); pcnt = df.groupby(pk)[ycol].transform("count")
    return df[ycol] - np.where((cnt - pcnt) >= 20, (tot - ptot) / (cnt - pcnt), df[ycol].mean())


runs = ol[ol.play_type == "RUN"]
ro = runs[out_ok.loc[runs.index]].copy()
ro["yds_oe"] = loo_adjust(ro, "yards_gained", ["offense_formation", "down", "ytg_b"])
pb = ol[(ol.play_type == "DROPBACK") & ol.peak_pressure_probability_allowed.notna()]
po = pb[out_ok.loc[pb.index]].copy()
po["ppa_oe"] = loo_adjust(po, "peak_pressure_probability_allowed", ["lined_up_position", "down", "offense_formation"])
po["pa_oe"] = loo_adjust(po.assign(pa=po.pressure_allowed.astype(float)), "pa", ["lined_up_position", "down", "offense_formation"])
q75 = lambda v: np.nanpercentile(v, 75)


def game_metrics(half=None):
    f = (lambda d: d) if half is None else (lambda d: d[d.half == half])
    k = 1 if half is None else 2
    G = {}
    pu = f(runs[runs.pull]); g = pu.groupby("nfl_id"); n = g.size()
    G["g_pull_bend_p75"] = g.move_peak_lat.agg(q75)[n >= 8 // k]; G["g_pull_speed"] = g.move_apex_speed.mean()[n >= 8 // k]
    cl = f(runs[runs.climb]); g = cl.groupby("nfl_id"); n = g.size()
    G["g_climb_time"] = g.t_climb.median()[n >= 20 // k]; G["g_climb_speed"] = g.move_apex_speed.mean()[n >= 20 // k]
    G["g_climb_bend_p75"] = g.move_peak_lat.agg(q75)[n >= 20 // k]
    ps = f(pb); g = ps.groupby("nfl_id"); n = g.size()
    G["g_set_lat_speed"] = g.set_lat_speed.mean()[n >= 100 // k]; G["g_set_depth"] = g.set_depth.mean()[n >= 100 // k]
    G["g_set_peak_lat"] = g.set_peak_lat.mean()[n >= 100 // k]
    oo = f(po); g = oo.groupby("nfl_id"); n = g.size()
    G["o_ppa_oe"] = g.ppa_oe.mean()[n >= 100 // k]; G["o_pressures_oe"] = g.pa_oe.mean()[n >= 100 // k]
    rr = f(ro);
    g = rr[rr.pull].groupby("nfl_id"); n = g.size(); G["o_yds_on_pulls_oe"] = g.yds_oe.mean()[n >= 8 // k]
    g = rr[rr.climb].groupby("nfl_id"); n = g.size(); G["o_yds_on_climbs_oe"] = g.yds_oe.mean()[n >= 20 // k]
    return pd.DataFrame(G)


GM = game_metrics(); GH = {h: game_metrics(h) for h in (0, 1)}
tackle = pd.Series((base.nfl_position == "T").astype(float), index=base.index)


def pcorr(x, y, ctrl):
    d = pd.concat([x.rename("x"), y.rename("y")] + [c.rename(f"c{i}") for i, c in enumerate(ctrl)], axis=1).dropna()
    if len(d) < 15:
        return np.nan, np.nan, len(d)
    Z = np.column_stack([np.ones(len(d))] + [d[f"c{i}"].values for i in range(len(ctrl))])
    rx = d.x - Z @ np.linalg.lstsq(Z, d.x, rcond=None)[0]; ry = d.y - Z @ np.linalg.lstsq(Z, d.y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry); return r, p, len(d)


ids = base.index[base.nfl_position.isin(["T", "G", "C"])]
rows = []
for cm in C.columns:
    x = C[cm].reindex(ids)
    for gm in GM.columns:
        r, p, n = pcorr(x, GM[gm].reindex(ids), [tackle.loc[ids]])
        r0, p0, _ = pcorr(x, GH[0][gm].reindex(ids), [tackle.loc[ids]])
        r1, p1, _ = pcorr(x, GH[1][gm].reindex(ids), [tackle.loc[ids]])
        rw, _, _ = pcorr(x, GM[gm].reindex(ids), [tackle.loc[ids], base.combine_weight.reindex(ids)])
        rows.append({"combine": cm, "game": gm, "n": n, "r": r, "p": p, "r_half0": r0, "p_half0": p0, "r_half1": r1, "p_half1": p1, "r_plus_weight": rw})
M = pd.DataFrame(rows)
pp_ = M.p.values; o = np.argsort(pp_); q = np.empty(len(pp_)); nn = np.isfinite(pp_).sum()
q[o] = np.minimum.accumulate((pp_[o] * nn / np.arange(1, len(pp_) + 1))[::-1])[::-1]; M["q_bh"] = np.minimum(q, 1)
M["replicates"] = (np.sign(M.r_half0) == np.sign(M.r_half1)) & (M.p_half0 < .10) & (M.p_half1 < .10)
M.to_csv(f"{ROOT}/results/25_ol_bend_matrix.csv", index=False)
pd.set_option("display.width", 250)
print(f"OL matrix: {M.p.notna().sum()} tests, p<.05: {(M.p < .05).sum()} (expected {.05 * M.p.notna().sum():.1f}), q<.10: {(M.q_bh < .10).sum()}, replicating: {M.replicates.sum()}")
cols = ["combine", "game", "n", "r", "p", "q_bh", "r_half0", "r_half1", "r_plus_weight"]
print("\nreplicating in both halves:"); print(M[M.replicates].sort_values("p")[cols].round(3).to_string(index=False))
print("\ntop 12 by p:"); print(M.sort_values("p").head(12)[cols].round(3).to_string(index=False))
print("\noutcome rows (pass pro / run yards), best per outcome:")
print(M[M.game.str.startswith("o_")].sort_values("p").groupby("game").head(2)[cols].round(3).to_string(index=False))
