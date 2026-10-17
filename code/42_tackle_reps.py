"""Phase 1c: open-field tackle reps -- a pool receiver/back catches the ball and a pool defender records the tackle or
assist on the same play; both tracked.

Events (catch, first contact, tackle / out of bounds / touchdown) come from the raw tracking event column.
Per rep, from the catch to the end of the play (frames are snap-aligned, max 6.9 s after the snap):
  dist_at_catch      tackler-carrier distance at the catch
  closing_speed      mean rate the distance shrinks from the catch until the tackler first gets within 1 yd
  pursuit_eff        mean cos(angle between the tackler's velocity and the direction to the carrier) during approach
  arrival_speed      tackler speed when he first gets within 1 yd
  breakdown          tackler's peak approach speed minus his arrival speed (controlled approach)
  yds_after_arrival  carrier's progress (toward +x) from the tackler's arrival to the end of the play (lower = better)
  t_arrival_to_end   time from arrival to the play ending
Output data/outcomes/tackle_reps.parquet
"""
import json, warnings
import duckdb
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").merge(pl, on="nfl_id")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
catch = pp[pp.nfl_position.isin(["WR", "TE", "RB", "FB"]) & (pp.pass_result == "C") & pp.target.fillna(False).astype(bool)][["game_id", "play_id", "nfl_id"]]
tk = pp[~pp.nfl_position.isin(["WR", "TE", "RB", "FB", "T", "G", "C"]) & ((pp.tackle.fillna(0) + pp.assist.fillna(0)) > 0)][["game_id", "play_id", "nfl_id", "nfl_position", "tackle", "assist"]]
R = catch.merge(tk, on=["game_id", "play_id"], suffixes=("_bc", "_tk"))
mi = meta.set_index(["game_id", "play_id", "nfl_id"]).idx
R["idx_bc"] = mi.reindex(pd.MultiIndex.from_arrays([R.game_id, R.play_id, R.nfl_id_bc])).values
R["idx_tk"] = mi.reindex(pd.MultiIndex.from_arrays([R.game_id, R.play_id, R.nfl_id_tk])).values
R = R.dropna(subset=["idx_bc", "idx_tk"]).reset_index(drop=True)
R[["idx_bc", "idx_tk"]] = R[["idx_bc", "idx_tk"]].astype(int)
# event frames from the raw tracking of the ball carrier
con = duckdb.connect(); con.register("keys", R[["game_id", "play_id", "nfl_id_bc"]].drop_duplicates())
ev = con.execute(f"""
  SELECT t.game_id, t.play_id, t.nfl_id,
    min(t."time") FILTER (WHERE event = 'ball_snap') ts,
    min(t."time") FILTER (WHERE event IN ('pass_outcome_caught', 'pass_arrived')) tc,
    min(t."time") FILTER (WHERE event IN ('tackle', 'out_of_bounds', 'touchdown', 'fumble')) te
  FROM '{ROOT}/data/raw/game_tracking_*.parquet' t JOIN keys k ON t.game_id = k.game_id AND t.play_id = k.play_id AND t.nfl_id = k.nfl_id_bc
  GROUP BY ALL""").df()
ev["k_catch"] = np.round((ev.tc - ev.ts).dt.total_seconds() * 10).astype("Int64")
ev["k_end"] = np.round((ev.te - ev.ts).dt.total_seconds() * 10).astype("Int64")
R = R.merge(ev[["game_id", "play_id", "nfl_id", "k_catch", "k_end"]].rename(columns={"nfl_id": "nfl_id_bc"}), on=["game_id", "play_id", "nfl_id_bc"], how="left")
R = R.dropna(subset=["k_catch"]).reset_index(drop=True)
get = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = R.idx_bc.values, R.idx_tk.values
xb, yb = get(A, "x_rel"), get(A, "y_field")
xt, yt, st, vxt, vyt = get(B, "x_rel"), get(B, "y_field"), get(B, "s"), get(B, "vx"), get(B, "vy")
vv = V[A] & V[B]
T = X.shape[1]; cols = np.arange(T)[None, :]
jc = (R.k_catch.astype(int).values + PRE).clip(0, T - 1)
je = np.where(R.k_end.isna(), T - 1, (R.k_end.fillna(0).astype(int).values + PRE)).clip(0, T - 1)
win = vv & (cols >= jc[:, None]) & (cols <= je[:, None])
dist = np.hypot(xb - xt, yb - yt)
r = np.arange(len(R))
arrive = win & (dist <= 1.0)
has_a = arrive.any(1); ja = np.where(has_a, arrive.argmax(1), -1)
R["dist_at_catch"] = np.where(vv[r, jc], dist[r, jc], np.nan)
appr = win & (cols <= np.where(has_a, ja, je)[:, None])
napp = appr.sum(1)
dt = (np.where(has_a, ja, je) - jc) / 10.0
d_end = dist[r, np.where(has_a, ja, je)]
R["closing_speed"] = np.where((napp >= 3) & (dt > 0), (R.dist_at_catch.values - d_end) / np.maximum(dt, .1), np.nan)
ux, uy = (xb - xt), (yb - yt)
nrm = np.hypot(ux, uy) + 1e-6
spd = np.hypot(vxt, vyt) + 1e-6
cosang = (vxt * ux + vyt * uy) / (nrm * spd)
R["pursuit_eff"] = np.where(napp >= 3, np.where(appr & (st > 1), cosang, 0).sum(1) / np.maximum((appr & (st > 1)).sum(1), 1), np.nan)
R["arrival_speed"] = np.where(has_a, st[r, ja], np.nan)
R["breakdown"] = np.where(has_a, np.where(appr, st, 0).max(1) - st[r, np.maximum(ja, 0)], np.nan)
R["yds_after_arrival"] = np.where(has_a, xb[r, je] - xb[r, np.maximum(ja, 0)], np.nan)
R["t_arrival_to_end"] = np.where(has_a, (je - ja) / 10.0, np.nan)
R["arrived"] = has_a
o = pp.drop_duplicates(["game_id", "play_id", "nfl_id"]).set_index(["game_id", "play_id", "nfl_id"])
kb = pd.MultiIndex.from_arrays([R.game_id, R.play_id, R.nfl_id_bc])
R["yac"] = o.yards_after_catch.reindex(kb).values; R["xyac"] = o.expected_yards_after_catch.reindex(kb).values
R = R.merge(meta[["idx", "season", "season_type", "nullified"]].rename(columns={"idx": "idx_bc"}), on="idx_bc", how="left")
R.to_parquet(f"{ROOT}/data/outcomes/tackle_reps.parquet", index=False)
print(f"tackle reps {len(R)}; tacklers {R.nfl_id_tk.nunique()}; arrived within 1 yd {R.arrived.mean():.2f}")
print(R[["dist_at_catch", "closing_speed", "pursuit_eff", "arrival_speed", "breakdown", "yds_after_arrival", "t_arrival_to_end"]].describe().round(2).T[["count", "mean", "50%", "std"]].to_string())
print("validation: yards after arrival vs YAC over expected r =", round(R[["yds_after_arrival"]].assign(yoe=R.yac - R.xyac).corr().iloc[0, 1], 2),
      "| closing speed vs YAC over expected r =", round(R[["closing_speed"]].assign(yoe=R.yac - R.xyac).corr().iloc[0, 1], 2))
