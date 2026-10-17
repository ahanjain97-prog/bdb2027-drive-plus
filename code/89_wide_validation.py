"""Loosened validation for thin data, applied symmetrically.

Data: reps rebuilt with PRE-BREAK-ONLY selection (mean receiver-DB distance <= radius from 0.5 s after the snap to the
receiver's break, >= 4 pre-break frames), receivers WR/TE/RB, all graded DBs (position control). Radius ladder 3/4/5 yd;
4 yd is the "wide" default. Zone-matching cell at alignment >= .5 (was .7), plus all zone.
Evidence standard (pre-stated): directional one-sided player-permutation p < .10, 80% interval (clustered SE), and sign
consistency across seasons. Reported for every claim, including the ones that were null under the strict standard.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(223)
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
GR = GR[GR.nfl_position.isin(["CB", "SS", "FS", "DB"])]
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
DIR = {"sep_gain_break": -1, "db_closing": +1, "open": -1, "completion": -1, "sep_at_throw": -1}
LAB = {"sep_gain_break": "separation gained at the key move (yd)", "db_closing": "closing after it (yd/s)", "open": "receiver open at the throw (pts)",
       "completion": "completion when targeted (pts)", "sep_at_throw": "separation at the throw (yd)"}
SC = {"sep_gain_break": 1, "db_closing": 1, "open": 100, "completion": 100, "sep_at_throw": 1}

# ---------------- build all pairs once
m = meta.merge(pl, on="nfl_id"); db = (m.play_type == "DROPBACK") & m.season_type.isin(["REG", "POST"]) & (m.nullified != "Y")
rec = m[db & m.nfl_position.isin(["WR", "TE", "RB", "FB"]) & m.route_ran.notna()]
dbk = m[db & m.nfl_id.isin(GR.index)]
P0 = rec[["game_id", "play_id", "idx", "nfl_id", "route_ran", "team_coverage_man_zone", "season"]].merge(
    dbk[["game_id", "play_id", "idx", "nfl_id"]], on=["game_id", "play_id"], suffixes=("_r", "_d")).reset_index(drop=True)
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = P0.idx_r.values, P0.idx_d.values
xr, yr, sr, lr, pr = g(A, "x_rel"), g(A, "y_field"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
xd, yd, sd = g(B, "x_rel"), g(B, "y_field"), g(B, "s"); vxr, vyr, vxd, vyd = g(A, "vx"), g(A, "vy"), g(B, "vx"), g(B, "vy")
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]; r0 = np.arange(len(P0))
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1); hb = a.max(1) > 0
dist = np.hypot(xr - xd, yr - yd)
pre_w = vv & (k >= 5) & (cols < jb[:, None]); n_pre = pre_w.sum(1)
sep_pre = np.where(n_pre > 0, np.where(pre_w, dist, 0).sum(1) / np.maximum(n_pre, 1), np.nan)
at = lambda arr, j: np.where((j < T) & (j >= 0) & vv[r0, np.clip(j, 0, T - 1)], arr[r0, np.clip(j, 0, T - 1)], np.nan)
P0["sep_gain_break"] = at(dist, jb + 5) - at(dist, jb - 3); P0["db_closing"] = (at(dist, jb) - at(dist, jb + 7)) / 0.7
thr = (pr >= .5) & vv & (cols >= PRE); jt = thr.argmax(1); P0["sep_at_throw"] = np.where(thr.any(1), dist[r0, jt], np.nan)
P0["open"] = np.where(P0.sep_at_throw.notna(), (P0.sep_at_throw >= 2).astype(float), np.nan)
w = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
P0["pa"] = np.where(w.sum(1) >= 3, np.where(w, cosv, 0).sum(1) / np.maximum(w.sum(1), 1), np.nan)
P0["zone"] = (P0.team_coverage_man_zone == "ZONE_COVERAGE").astype(int)
kr = pd.MultiIndex.from_arrays([P0.game_id, P0.play_id, P0.nfl_id_r])
P0["targeted"] = pp.target.reindex(kr).values; res = pp.pass_result.reindex(kr).values
P0["completion"] = np.where(P0.targeted == 1, (res == "C").astype(float), np.nan)
P0["dp"] = P0.nfl_id_d.map((GR.drive_plus - 100) / 10); P0["is_s"] = (P0.nfl_id_d.map(GR.nfl_position) != "CB").astype(int)
P0["sep_pre"] = sep_pre; P0["n_pre"] = n_pre; P0["hb"] = hb
P0 = P0[P0.hb & (P0.n_pre >= 4)].copy()
tot = P0.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
P0["wr_skill"] = ((P0.nfl_id_r.map(tot["sum"]) - P0.sep_gain_break.fillna(0)) / (P0.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)).fillna(P0.sep_gain_break.mean())


def est(d, y, ctx, n_perm=250):
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"
    mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    if used.nfl_id_d.nunique() < 8 or len(used) < 40:
        return None
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]})
    t0 = mod.fit().tvalues["dp"] * DIR[y]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"] * DIR[y])
    p1 = (1 + (np.array(null) >= t0).sum()) / (n_perm + 1)
    b, se = m_.params["dp"] * SC[y], m_.bse["dp"] * SC[y]
    return {"b": b, "lo80": b - 1.2816 * se, "hi80": b + 1.2816 * se, "p1": p1, "n": len(used), "DBs": used.nfl_id_d.nunique()}


def verdict(r, y):
    if r is None: return "too few"
    right = np.sign(r["b"]) == DIR[y]; ci_ok = right and (r["lo80"] > 0 if DIR[y] > 0 else r["hi80"] < 0)
    return "supported" if (right and r["p1"] < .10 and ci_ok) else ("right sign" if right else "wrong sign")


def fmt(r, y):
    return "too few" if r is None else f"{r['b']:+.3f} [{r['lo80']:+.3f}, {r['hi80']:+.3f}] p1 {r['p1']:.3f} · {verdict(r, y):10s} · n {r['n']}, DBs {r['DBs']}"


def run(P, match=.5, label=""):
    P = P.copy(); P["cell"] = np.where(P.pa >= match, np.where(P.zone == 1, "zone-matching", "man-matching"), "other")
    tasks = [("sep_gain_break", P, "wr_skill + C(zone) + C(route_ran) + is_s"), ("db_closing", P[P.cell == "zone-matching"], "wr_skill + C(route_ran) + is_s"),
             ("db_closing", P[P.zone == 1], "wr_skill + C(route_ran) + is_s"), ("open", P, "C(zone) + C(route_ran) + is_s"),
             ("sep_at_throw", P, "wr_skill + C(zone) + C(route_ran) + is_s"), ("completion", P[P.targeted == 1], "C(zone) + C(route_ran) + is_s")]
    names = ["sep gained (all)", f"closing, zone-matching (align ≥ {match})", "closing, all zone", "open at throw (all)", "sep at throw (all)", "completion (targets)"]
    print(f"  [{label}] reps {len(P)}, DBs {P.nfl_id_d.nunique()}, targets {int((P.targeted == 1).sum())}, zone-matching {int((P.cell == 'zone-matching').sum())}")
    out = {}
    for nm, (y, d, ctx) in zip(names, tasks):
        r = est(d, y, ctx); out[nm] = (r, y); print(f"    {nm:38s} {fmt(r, y)}", flush=True)
    return out


pd.set_option("display.width", 240)
print("=== radius ladder (pre-break-only selection, WR/TE/RB routes, all graded DBs) ===")
for rad in [3.0, 4.0, 5.0]:
    run(P0[P0.sep_pre <= rad], .5, f"radius {rad:.0f} yd")
W = P0[P0.sep_pre <= 4.0].copy()
print("\n=== wide default (4 yd): corners only ===")
run(W[W.is_s == 0], .5, "4 yd, CBs only")
print("\n=== wide default (4 yd): matching threshold .7 (the registered one) ===")
run(W, .7, "4 yd, align ≥ .7")
print("\n=== wide default (4 yd): by season ===")
for s in [2024, 2025]:
    run(W[W.season == s], .5, f"4 yd, {s} season")
print("\n=== wide default (4 yd): without GO routes ===")
run(W[W.route_ran != "GO"], .5, "4 yd, no GO")
W.to_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet", index=False)
print("\nstandard: one-sided player-permutation p < .10, 80% interval excluding 0, expected sign -> 'supported'.")
