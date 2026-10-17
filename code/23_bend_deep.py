"""Bend, second pass: sustained metrics, penetration outcome, full combine x game matrix for DL, replication filter.

1) Play-level validation: does in-game penetration (max depth into the backfield before the throw, depth at 2.0 s)
   predict pressure on the same rush? AUC + split-half player reliability of each candidate outcome.
2) Matrix: every combine bend metric (hoop, pass-rush drill: peak / sustained lateral accel, apex / sustained
   speed, radius, retention, heading change, turn time; hoop completion time and mean speed) vs every game
   metric (rush: corner bend p75, sustained arc lateral accel, arc speed, apex speed, radius, penetration;
   outcomes: pressure over expected, run tackles / TFL over expected, run penetration at 1.5 s).
   DL pooled (partial on EDGE/IDL unit), and each unit alone. BH q within each matrix.
3) Replication filter: a combine->game link counts only if the game metric computed on EACH half of games
   separately gives the same sign and p < .10 in both halves (pooled DL).
"""
import hashlib, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cb = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_player.parquet")
gp = pd.read_parquet(f"{ROOT}/data/bend/game_bend_play.parquet")
gp["half"] = gp.game_id.map(lambda g: int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2)
gp["ytg_b"] = pd.cut(gp.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
out_ok = gp.season_type.isin(["REG", "POST"]) & (gp.nullified != "Y")
dl = gp[gp.nfl_position.isin(["DE", "OLB", "DT", "NT"]) & gp.lined_up_position.isin(["EDGE", "INTERIOR_LINE"])].copy()
dl["unit"] = np.where(dl.nfl_position.isin(["DE", "OLB"]), "EDGE", "IDL")


def loo_adjust(df, ycol, cells):
    key = df[cells].astype(str).agg("|".join, axis=1)
    tot = df.groupby(key)[ycol].transform("sum"); cnt = df.groupby(key)[ycol].transform("count")
    pk = key + "#" + df.nfl_id.astype(str)
    ptot = df.groupby(pk)[ycol].transform("sum"); pcnt = df.groupby(pk)[ycol].transform("count")
    return df[ycol] - np.where((cnt - pcnt) >= 20, (tot - ptot) / (cnt - pcnt), df[ycol].mean())


rush = dl[(dl.play_type == "DROPBACK") & dl.player_get_off.notna()].copy()
runs = dl[dl.play_type == "RUN"].copy()
ro = rush[out_ok.loc[rush.index]].copy()
ro["poe"] = loo_adjust(ro, "pressure", ["lined_up_position", "blitzing", "down", "ytg_b"])
for c_ in ["max_depth_live", "depth_2s"]:
    ro[c_ + "_oe"] = loo_adjust(ro.dropna(subset=[c_]), c_, ["lined_up_position", "blitzing", "down", "ytg_b"]).reindex(ro.index)
uo = runs[out_ok.loc[runs.index]].copy()
uo["tk_oe"] = loo_adjust(uo, "tackle_any", ["lined_up_position", "down", "ytg_b"])
uo["tfl_oe"] = loo_adjust(uo, "tfl", ["lined_up_position", "down", "ytg_b"])
uo["rdepth_oe"] = loo_adjust(uo.dropna(subset=["depth_1p5s"]), "depth_1p5s", ["lined_up_position", "down", "ytg_b"]).reindex(uo.index)

# ---------------- 1) penetration as a rush outcome
print("=== 1) penetration vs pressure (play level, REG+POST rush snaps) ===")
for c_ in ["max_depth_live", "depth_2s", "peak_lat", "sust_lat"]:
    d = ro[[c_, "pressure"]].dropna()
    auc = stats.mannwhitneyu(d[c_][d.pressure == 1], d[c_][d.pressure == 0]).statistic / ((d.pressure == 1).sum() * (d.pressure == 0).sum())
    print(f"  {c_:15s} n {len(d):6d}  AUC for pressure {auc:.3f}")


def half_rel(df, col, min_n, agg="mean"):
    g = df.groupby(["nfl_id", "half"])[col]
    w = pd.DataFrame({"v": g.agg(agg), "n": g.count()})
    w = w[w.n >= min_n // 2].v.unstack().dropna()
    r = np.corrcoef(w[0], w[1])[0, 1]
    return 2 * r / (1 + r), len(w)


print("  player split-half reliability (Spearman-Brown):")
for c_, df_, mn in [("poe", ro, 50), ("max_depth_live_oe", ro, 50), ("depth_2s_oe", ro, 50), ("tk_oe", uo, 40),
                    ("tfl_oe", uo, 40), ("rdepth_oe", uo, 40)]:
    print(f"    {c_:18s} {half_rel(df_, c_, mn)[0]:.2f}")

# ---------------- game-side player metrics (optionally restricted to one half of games)
q75 = lambda v: np.nanpercentile(v, 75)


def game_metrics(half=None):
    r_ = rush if half is None else rush[rush.half == half]
    o_ = ro if half is None else ro[ro.half == half]
    u_ = uo if half is None else uo[uo.half == half]
    mn = 50 if half is None else 25
    G = {}
    g = r_.groupby("nfl_id")
    n = g.size()
    G["g_corner_bend_p75"] = g.peak_lat.agg(q75)
    G["g_arc_sust_lat"] = g.sust_lat.mean()
    G["g_arc_speed"] = g.sust_speed.mean()
    G["g_apex_speed"] = g.apex_speed.mean()
    G["g_radius_med"] = g.radius.median()
    G = {k: v[n >= mn] for k, v in G.items()}
    go = o_.groupby("nfl_id"); no = go.size()
    G["o_pressure_oe"] = go.poe.mean()[no >= mn]
    G["o_penetration_oe"] = go.max_depth_live_oe.mean()[no >= mn]
    G["o_depth2s_oe"] = go.depth_2s_oe.mean()[no >= mn]
    gu = u_.groupby("nfl_id"); nu = gu.size(); mu = 40 if half is None else 20
    G["o_run_tackle_oe"] = gu.tk_oe.mean()[nu >= mu]
    G["o_run_tfl_oe"] = gu.tfl_oe.mean()[nu >= mu]
    G["o_run_depth_oe"] = gu.rdepth_oe.mean()[nu >= mu]
    return pd.DataFrame(G)


GM = game_metrics(); GH = {h: game_metrics(h) for h in (0, 1)}
CM = [c for c in cb.columns if (c.startswith("c_hoop_") or c.startswith("c_rushdrill_")) and "_L_" not in c and "_R_" not in c and "asym" not in c]
CM = [c for c in CM if not c.endswith("lean_deg")]                  # monotone in peak_lat
unit = pd.Series(np.where(base.nfl_position.isin(["DE", "OLB"]), 1.0, 0.0), index=base.index)


def pcorr(x, y, ctrl):
    d = pd.concat([x.rename("x"), y.rename("y")] + [c_.rename(f"c{i}") for i, c_ in enumerate(ctrl)], axis=1).dropna()
    if len(d) < 15:
        return np.nan, np.nan, len(d)
    Z = np.column_stack([np.ones(len(d))] + [d[f"c{i}"].values for i in range(len(ctrl))])
    rx = d.x - Z @ np.linalg.lstsq(Z, d.x, rcond=None)[0]; ry = d.y - Z @ np.linalg.lstsq(Z, d.y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(d)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


rows = []
for scope in ["DL", "EDGE", "IDL"]:
    ids = base.index[base.nfl_position.isin(["DE", "OLB", "DT", "NT"] if scope == "DL" else (["DE", "OLB"] if scope == "EDGE" else ["DT", "NT"]))]
    ctrl = [unit.loc[ids]] if scope == "DL" else []
    for cmet in CM:
        x = cb[cmet].reindex(ids)
        for gmet in GM.columns:
            r, p, n = pcorr(x, GM[gmet].reindex(ids), ctrl)
            r0, p0, _ = pcorr(x, GH[0][gmet].reindex(ids), ctrl)
            r1, p1, _ = pcorr(x, GH[1][gmet].reindex(ids), ctrl)
            rw, pw, _ = pcorr(x, GM[gmet].reindex(ids), ctrl + [base.combine_weight.reindex(ids)])
            rows.append({"scope": scope, "combine": cmet, "game": gmet, "n": n, "r": r, "p": p, "r_half0": r0, "p_half0": p0,
                         "r_half1": r1, "p_half1": p1, "r_plus_weight": rw})
M = pd.DataFrame(rows)
M["q_bh"] = M.groupby("scope").p.transform(bh)
M["replicates"] = (np.sign(M.r_half0) == np.sign(M.r_half1)) & (M.p_half0 < .10) & (M.p_half1 < .10)
M.to_csv(f"{ROOT}/results/23_bend_matrix.csv", index=False)
pd.set_option("display.width", 250)
print("\n=== 2) matrix summary: tests, p<.05 (expected), q<.10, replicating in both halves ===")
print(M.groupby("scope").agg(tests=("p", "size"), p05=("p", lambda p: int((p < .05).sum())),
                             expected=("p", lambda p: round(.05 * p.notna().sum(), 1)), q10=("q_bh", lambda q: int((q < .10).sum())),
                             replicating=("replicates", "sum")).to_string())
print("\n=== 3) links that replicate in both halves of games (any scope) ===")
cols = ["scope", "combine", "game", "n", "r", "p", "q_bh", "r_half0", "r_half1", "r_plus_weight"]
print(M[M.replicates].sort_values(["scope", "p"])[cols].round(3).to_string(index=False))
print("\n=== top 15 by p, pooled DL ===")
print(M[M.scope == "DL"].sort_values("p").head(15)[cols].round(3).to_string(index=False))
