"""Phase 1b: coverage reps -- a pool WR/TE running a route and a pool DB covering him on the same dropback, both tracked.

Rep = mean receiver-DB distance <= 3 yd over >= 1 s of the route (0.5 s after the snap .. the throw, max 3.5 s).
Per rep (DB perspective = separation ALLOWED, receiver perspective = separation CREATED):
  sep_mean        mean distance over the route window
  sep_at_break    distance at the receiver's break (frame of his peak smoothed |lateral accel| at > 2 yd/s, 0.5-3 s)
  sep_gain_break  distance 0.5 s after the break minus 0.3 s before it (how much the break opened him up)
  sep_at_throw    distance at the pass-forward frame (checked against NGS separation_at_pass_forward)
  db_speed_deficit  receiver speed minus DB speed over the 0.5 s after the break (DB slow to re-accelerate)
Plus context: route, man/zone, DB assignment, cushion, targeted, completion.
Output data/outcomes/coverage_reps.parquet
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
m = meta.merge(pl, on="nfl_id")
db = m.play_type == "DROPBACK"
rec = m[db & m.nfl_position.isin(["WR", "TE"]) & m.route_ran.notna()]
dbk = m[db & m.nfl_position.isin(["CB", "SS", "FS", "DB"])]
P = rec[["game_id", "play_id", "idx", "nfl_id", "route_ran", "team_coverage_man_zone", "lined_up_position", "season", "season_type", "nullified", "down", "yards_to_go"]].merge(
    dbk[["game_id", "play_id", "idx", "nfl_id", "coverage_assignment", "lined_up_position"]], on=["game_id", "play_id"], suffixes=("_r", "_d"))
get = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = P.idx_r.values, P.idx_d.values
xr, yr, sr, lr, pr = get(A, "x_rel"), get(A, "y_field"), get(A, "s"), get(A, "a_lat"), get(A, "post_pass")
xd, yd, sd = get(B, "x_rel"), get(B, "y_field"), get(B, "s")
vv = V[A] & V[B]
k = np.arange(X.shape[1])[None, :] - PRE
route = vv & (k >= 5) & (k <= 35) & (pr < .5)
dist = np.hypot(xr - xd, yr - yd)
n_live = route.sum(1)
sep_mean = np.where(n_live > 0, np.where(route, dist, 0).sum(1) / np.maximum(n_live, 1), np.nan)
keep = (n_live >= 10) & (sep_mean <= 3.0)
P, dist, route, vv = P[keep].reset_index(drop=True), dist[keep], route[keep], vv[keep]
xr, yr, sr, lr, pr, sd = xr[keep], yr[keep], sr[keep], lr[keep], pr[keep], sd[keep]
P["sep_mean"] = sep_mean[keep]
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
bw = vv & (k >= 5) & (k <= 30) & (sr > 2.0) & (pr < .5)
a = np.where(bw, np.abs(ls), -1); jb = a.argmax(1); has_b = a.max(1) > 0
r = np.arange(len(P)); T = X.shape[1]
at = lambda arr, j: arr[r, np.clip(j, 0, T - 1)]
okj = lambda j: vv[r, np.clip(j, 0, T - 1)]
P["sep_at_break"] = np.where(has_b, at(dist, jb), np.nan)
P["sep_gain_break"] = np.where(has_b & okj(jb + 5) & okj(jb - 3), at(dist, jb + 5) - at(dist, jb - 3), np.nan)
P["t_break"] = np.where(has_b, k[0, jb] / 10.0, np.nan)
P["break_lat"] = np.where(has_b, a.max(1), np.nan)
post_w = (np.arange(T)[None, :] > jb[:, None]) & (np.arange(T)[None, :] <= jb[:, None] + 5) & vv
cnt = post_w.sum(1)
P["db_speed_deficit"] = np.where(has_b & (cnt >= 3), np.where(post_w, sr - sd, 0).sum(1) / np.maximum(cnt, 1), np.nan)
thr = (pr >= .5) & vv & (k >= 0)
jt = thr.argmax(1); has_t = thr.any(1)
P["sep_at_throw"] = np.where(has_t, at(dist, jt), np.nan)
P["t_throw"] = np.where(has_t, k[0, jt] / 10.0, np.nan)
o = pp.set_index(["game_id", "play_id", "nfl_id"])
kr = pd.MultiIndex.from_arrays([P.game_id, P.play_id, P.nfl_id_r])
P["ngs_sep"] = o.separation_at_pass_forward.reindex(kr).values
P["cushion"] = o.cushion.reindex(kr).values
P["targeted"] = o.target.reindex(kr).astype("object").map({True: 1.0, False: 0.0}).astype(float).values
P["completion"] = np.where(P.targeted == 1, (o.pass_result.reindex(kr).values == "C").astype(float), np.nan)
P["rec_yards"] = o.rec_yards.reindex(kr).values
P.to_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet", index=False)
print(f"coverage reps {len(P)}; receivers {P.nfl_id_r.nunique()}, DBs {P.nfl_id_d.nunique()}; with break {P.sep_at_break.notna().mean():.2f}, with throw {P.sep_at_throw.notna().mean():.2f}")
v = P[["sep_at_throw", "ngs_sep"]].dropna()
print(f"tracking separation at the throw vs NGS nearest-defender separation: r {v.corr().iloc[0, 1]:.2f}, "
      f"share where the pool DB IS the nearest defender (|diff| < 0.5 yd): {(np.abs(v.sep_at_throw - v.ngs_sep) < .5).mean():.2f}")
print(P.groupby("team_coverage_man_zone")[["sep_mean", "sep_at_break", "sep_gain_break", "sep_at_throw", "db_speed_deficit"]].median().round(2).to_string())
print("targeted:", int(P.targeted.sum()), " completion rate when targeted:", round(P.completion.mean(), 3))
c = P.dropna(subset=["sep_at_throw", "targeted"])
print("sep at throw by targeted:", c.groupby("targeted").sep_at_throw.mean().round(2).to_dict(),
      "| completion by sep at throw tercile:", P.dropna(subset=["completion"]).groupby(pd.qcut(P.sep_at_throw, 3)).completion.mean().round(2).to_dict())
