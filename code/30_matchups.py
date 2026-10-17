"""Head-to-head pass-rush reps: a pool DL (DE/OLB/DT/NT aligned EDGE or INTERIOR_LINE) vs a pool OL (T/G/C) on the
same dropback, both tracked. Positions are comparable within a play (x relative to the LOS, y = field width).

Engagement: |dy| <= 4 yd at the snap and minimum distance <= 1.5 yd within 0-2.5 s (before the throw / play end).
One rep per DL per play: the closest engaged OL. Per rep:
  corner_turned  rusher >= 0.5 yd deeper (toward the QB, -x) than the blocker while within 2 yd laterally, by 2.5 s
  t_corner       when (NaN if never)
  rusher bend    peak / sustained |lateral accel| and speed within +/-0.3 s of the corner (else closest approach)
  blocker set    depth and width of the set at 1.0 s; side from the set direction (offense faces +x: +y = left)
  outcomes       blocker's peak pressure probability allowed + pressure allowed; rusher pressure / sack
Output data/bend/matchups.parquet (all candidate pairs, `engaged` flag)
"""
import json
import numpy as np
import pandas as pd

ROOT = "/Users/ahanjain/bdb_ssl"
voc = json.load(open(f"{ROOT}/data/seq/vocab.json"))
F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r")
V = np.load(f"{ROOT}/data/seq/valid.npy")
m = meta.merge(pl, on="nfl_id", how="left")
db = m.play_type == "DROPBACK"
dl = m[db & m.nfl_position.isin(["DE", "OLB", "DT", "NT"]) & m.lined_up_position.isin(["EDGE", "INTERIOR_LINE"])]
ol = m[db & m.nfl_position.isin(["T", "G", "C"]) & m.lined_up_position.isin(["T", "G", "C"])]
pairs = dl[["game_id", "play_id", "idx", "nfl_id", "lined_up_position"]].merge(
    ol[["game_id", "play_id", "idx", "nfl_id", "lined_up_position"]], on=["game_id", "play_id"], suffixes=("_dl", "_ol"))
print("candidate DL-OL pairs on the same dropback:", len(pairs), flush=True)
ci = {n: F.index(n) for n in ["x_rel", "y_field", "s", "a_lat", "post_pass", "post_dead"]}
A, B = pairs.idx_dl.values, pairs.idx_ol.values
get = lambda idx, n: np.asarray(X[idx][:, :, ci[n]], dtype=np.float32)
xd, yd, sd, ld = get(A, "x_rel"), get(A, "y_field"), get(A, "s"), get(A, "a_lat")
xo, yo = get(B, "x_rel"), get(B, "y_field")
vv = V[A] & V[B]
k = np.arange(X.shape[1])[None, :] - PRE
live = vv & (k >= 0) & (k <= 25) & (get(A, "post_pass") < .5) & (get(A, "post_dead") < .5)
dist = np.hypot(xd - xo, yd - yo)
dy0 = np.abs(yd[:, PRE] - yo[:, PRE])
mind = np.where(live, dist, np.inf).min(1)
lat_s = ld.copy(); lat_s[:, 1:-1] = (ld[:, :-2] + ld[:, 1:-1] + ld[:, 2:]) / 3
corner = live & (xd < xo - 0.5) & (np.abs(yd - yo) <= 2.0)
has_c = corner.any(1)
jc = np.where(has_c, corner.argmax(1), np.where(live, dist, np.inf).argmin(1))
win = (np.abs(np.arange(X.shape[1])[None, :] - jc[:, None]) <= 3) & vv & (k >= 0)
cnt = win.sum(1).clip(1)
i10 = PRE + 10
rep = pd.DataFrame({
    "game_id": pairs.game_id.values, "play_id": pairs.play_id.values, "dl_id": pairs.nfl_id_dl.values, "ol_id": pairs.nfl_id_ol.values,
    "dl_align": pairs.lined_up_position_dl.values, "ol_align": pairs.lined_up_position_ol.values,
    "idx_dl": A, "idx_ol": B, "dy_snap": dy0, "min_dist": mind,
    "corner_turned": has_c, "t_corner": np.where(has_c, k[0, jc] / 10.0, np.nan),
    "rush_peak_lat": np.where(win, np.abs(lat_s), 0).max(1), "rush_sust_lat": np.where(win, np.abs(lat_s), 0).sum(1) / cnt,
    "rush_speed_at_corner": np.where(win, sd, 0).sum(1) / cnt,
    "ol_set_depth_1s": -(xo[:, i10] - xo[:, PRE]), "ol_set_width_1s": np.abs(yo[:, i10] - yo[:, PRE]),
    "ol_set_dir": np.where(yo[:, i10] - yo[:, PRE] > 0, "left", "right"),
    "dl_wider": np.abs(yd[:, PRE]) > np.abs(yo[:, PRE]),
    "dl_depth_2s": np.where(vv[:, PRE + 20], -xd[:, PRE + 20], np.nan)})
rep["engaged"] = (rep.dy_snap <= 4.0) & (rep.min_dist <= 1.5)
# one rep per DL per play: closest engaged OL
rep = rep.sort_values("min_dist")
rep["primary"] = False
first = rep[rep.engaged].drop_duplicates(["game_id", "play_id", "dl_id"]).index
rep.loc[first, "primary"] = True
o = pp.set_index(["game_id", "play_id", "nfl_id"])
kol = pd.MultiIndex.from_arrays([rep.game_id, rep.play_id, rep.ol_id]); kdl = pd.MultiIndex.from_arrays([rep.game_id, rep.play_id, rep.dl_id])
rep["ppa"] = o.peak_pressure_probability_allowed.reindex(kol).values
rep["ol_pressure_allowed"] = o.pressure_allowed.reindex(kol).astype(float).values
rep["dl_pressure"] = (o.time_to_pressure.reindex(kdl).notna().values | (o.sack.reindex(kdl).fillna(0).values > 0)).astype(float)
rep["dl_get_off"] = o.player_get_off.reindex(kdl).values
rep["blitzing"] = o.blitzing.reindex(kdl).astype(str).values
mm = m.set_index("idx")
for c in ["season", "season_type", "down", "yards_to_go", "offense_formation", "nullified", "quarter"]:
    rep[c] = mm[c].reindex(rep.idx_dl).values
rep.to_parquet(f"{ROOT}/data/bend/matchups.parquet", index=False)
e = rep[rep.primary]
print(f"primary engaged reps: {len(e)}; DL {e.dl_id.nunique()}, OL {e.ol_id.nunique()}; REG+POST {e.season_type.isin(['REG', 'POST']).sum()}")
print(e.groupby(["ol_align", "dl_align"]).size().to_string())
print(f"corner turned {e.corner_turned.mean():.3f}; ppa mean {e.ppa.mean():.3f}; OL pressure allowed {e.ol_pressure_allowed.mean():.3f}; DL pressure {e.dl_pressure.mean():.3f}")
print("corner turned -> outcomes:\n", e.groupby("corner_turned")[["ppa", "ol_pressure_allowed", "dl_pressure"]].mean().round(3).to_string())
