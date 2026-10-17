"""In-game bend and related mechanics, per play, plus the play's outcome. Same lateral-accel formula and
3-frame smoothing as the combine (20_bend_combine.py). Frame of reference from 01_build_sequences
(offense toward +x; x relative to LOS; a_lat sign: + = turning left / counter-clockwise).

DL pass rush (EDGE / INTERIOR_LINE alignment, dropback with a recorded get-off), window snap .. min(3.5 s,
  pass thrown / play dead): corner = frame of peak |a_lat| with s > 2.5 -> peak_lat, lean_deg, apex_speed,
  radius, turn dir (L/R), t_corner. Outcome: pressure (time_to_pressure recorded or sack), time_to_pressure, sack.
DL run plays: same corner metrics + block-shed proxy: after get-off (s >= 2), stall = lowest speed while within
  2 yd of the LOS in 0-2.5 s; release = speed gained in the 0.7 s after the stall. Outcome: tackle / assist / TFL.
OL run plays: pull = lateral travel >= 3 yd by 1.2 s at >= 3 yd/s; climb = reaches 2 yd past the LOS by 3 s.
  Bend while pulling / climbing = peak |a_lat| with s > 2. Outcome: yards gained, EPA.
OL dropbacks (pressure probability recorded): pass set = max lateral speed and depth in 1.5 s, peak |a_lat|
  (s > 1) in 2.5 s. Outcome: peak_pressure_probability_allowed, pressure_allowed.
Movement uses all game types; outcome analyses later restrict to REG + POST.
Output data/bend/game_bend_play.parquet
"""
import json
import numpy as np
import pandas as pd

ROOT = "/Users/ahanjain/bdb_ssl"
G = 10.73
voc = json.load(open(f"{ROOT}/data/seq/vocab.json"))
F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
keep = ["game_id", "play_id", "nfl_id", "player_get_off", "time_to_pressure", "sack", "tackle", "assist", "tackle_for_loss",
        "peak_pressure_probability_allowed", "pressure_allowed", "yards_gained", "expected_points_added", "blitzing",
        "offense_formation", "pass_result"]
m = meta.merge(pp[[c for c in keep if c not in ("blitzing", "offense_formation", "pass_result")]], on=["game_id", "play_id", "nfl_id"], how="left").merge(pl, on="nfl_id", how="left")
assert (m.idx.values == np.arange(len(m))).all()
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r")
V = np.load(f"{ROOT}/data/seq/valid.npy")

DLPOS, OLPOS = {"DE", "OLB", "DT", "NT"}, {"T", "G", "C"}
sel = np.where(m.nfl_position.isin(DLPOS | OLPOS).values & m.lined_up_position.notna().values)[0]
print("plays to process", len(sel), flush=True)
ch = {n: np.asarray(X[sel][:, :, F.index(n)], dtype=np.float32) for n in
      ["x_rel", "y_rel", "s", "a_lat", "vy", "post_pass", "post_dead"]}
v = V[sel]
lat = ch["a_lat"].copy()
lat_s = lat.copy()
lat_s[:, 1:-1] = (lat[:, :-2] + lat[:, 1:-1] + lat[:, 2:]) / 3        # same 3-frame smoothing as the combine
s, xr, yr = ch["s"], ch["x_rel"], ch["y_rel"]
k = np.arange(X.shape[1])[None, :] - PRE                              # frames relative to snap
live = v & (k >= 0) & (ch["post_pass"] < .5) & (ch["post_dead"] < .5)


def corner(window, smin):
    a = np.where(window & (s > smin), np.abs(lat_s), -1.0)
    j = a.argmax(1)
    ok = a.max(1) > 0
    r = np.arange(len(a))
    pk = np.where(ok, a[r, j], np.nan)
    sp = np.where(ok, s[r, j], np.nan)
    return {"peak_lat": pk, "lean_deg": np.degrees(np.arctan(pk / G)), "apex_speed": sp,
            "radius": sp ** 2 / pk, "turn_dir": np.where(ok, np.where(lat_s[r, j] > 0, "L", "R"), None),
            "t_corner": np.where(ok, k[0, j] / 10.0, np.nan)}


out = pd.DataFrame({"idx": sel})
win_c = live & (k <= 35)
cr = corner(win_c, 2.5)
for c_, val in cr.items():
    out[c_] = val
# sustained arc: mean |a_lat| and speed over +/-0.3 s around the corner peak (live frames, s > 2)
a_c = np.where(win_c & (s > 2.5), np.abs(lat_s), -1.0); jc = a_c.argmax(1); okc = a_c.max(1) > 0
cols_ = np.arange(X.shape[1])[None, :]
near_c = (np.abs(cols_ - jc[:, None]) <= 3) & win_c & (s > 2.0) & okc[:, None]
cnt = near_c.sum(1)
out["sust_lat"] = np.where(cnt >= 3, np.where(near_c, np.abs(lat_s), 0).sum(1) / np.maximum(cnt, 1), np.nan)
out["sust_speed"] = np.where(cnt >= 3, np.where(near_c, s, 0).sum(1) / np.maximum(cnt, 1), np.nan)
# penetration (defense moves toward -x; QB sits ~5-7 yd behind the LOS)
i20 = PRE + 20
out["depth_2s"] = np.where(live[:, i20], -xr[:, i20], np.nan)
out["max_depth_live"] = np.where((live & (k <= 30)).any(1), -np.where(live & (k <= 30), xr, np.inf).min(1), np.nan)
i15 = PRE + 15
out["depth_1p5s"] = np.where(v[:, i15], -xr[:, i15], np.nan)
# block-shed proxy (DL, meaningful on runs)
got_off = live & (s >= 2.0)
first_go = np.where(got_off.any(1), got_off.argmax(1), -1)
after = (k >= 0) & (k <= 25) & v & (np.arange(X.shape[1])[None, :] > first_go[:, None]) & (first_go[:, None] >= 0)
near = after & (np.abs(xr) <= 2.0)
sm = np.where(near, s, np.inf)
js = sm.argmin(1)
has = np.isfinite(sm.min(1))
r = np.arange(len(sel))
out["stall_speed"] = np.where(has, s[r, js], np.nan)
rel = np.full(len(sel), np.nan)
for i in np.where(has)[0]:
    w = slice(js[i], min(js[i] + 8, X.shape[1]))
    vv = v[i, w]
    if vv.sum() > 1:
        rel[i] = s[i, w][vv].max() - s[i, js[i]]
out["release_gain"] = rel
# OL pull / climb (run plays) and pass set (dropbacks)
w12 = v & (k >= 0) & (k <= 12)
lat_trav = np.where(w12, np.abs(yr), 0).max(1)
fast = np.where(w12, s, 0).max(1)
out["pull"] = (lat_trav >= 3.0) & (fast >= 3.0)
w30 = v & (k >= 0) & (k <= 30)
reach = w30 & (xr >= 2.0)
out["climb"] = reach.any(1)
out["t_climb"] = np.where(reach.any(1), k[0, reach.argmax(1)] / 10.0, np.nan)
cb = corner(w30, 2.0)
out["move_peak_lat"] = cb["peak_lat"]; out["move_apex_speed"] = cb["apex_speed"]; out["move_turn_dir"] = cb["turn_dir"]
w15 = v & (k >= 0) & (k <= 15)
out["set_lat_speed"] = np.where(w15, np.abs(ch["vy"]), 0).max(1)
out["set_depth"] = -(np.where(w15, xr, np.inf).min(1) - xr[:, PRE])
cs = corner(v & (k >= 0) & (k <= 25), 1.0)
out["set_peak_lat"] = cs["peak_lat"]

info = m.iloc[sel][["idx", "game_id", "play_id", "nfl_id", "nfl_position", "season", "season_type", "lined_up_position",
                    "play_type", "down", "yards_to_go", "nullified", "blitzing", "offense_formation"] + [c for c in keep[3:] if c not in ("blitzing", "offense_formation", "pass_result")]].reset_index(drop=True)
out = info.merge(out, on="idx")
for c_ in ["sack", "tackle", "assist", "tackle_for_loss", "pressure_allowed", "player_get_off", "time_to_pressure", "peak_pressure_probability_allowed", "yards_gained", "expected_points_added"]:
    out[c_] = pd.to_numeric(out[c_].astype("object").replace({True: 1, False: 0}), errors="coerce").astype(float)
out["pressure"] = (out.time_to_pressure.notna() | (out.sack.fillna(0) > 0)).astype(float)
out["tackle_any"] = ((out.tackle.fillna(0) + out.assist.fillna(0)) > 0).astype(float)
out["tfl"] = pd.to_numeric(out.tackle_for_loss, errors="coerce").fillna(0).clip(0, 1).astype(float)
out["grp"] = np.where(out.nfl_position.isin(OLPOS), "OL", "DL")
out.to_parquet(f"{ROOT}/data/bend/game_bend_play.parquet", index=False)

dl = out[out.grp == "DL"]
rush = dl[(dl.play_type == "DROPBACK") & dl.player_get_off.notna() & dl.lined_up_position.isin(["EDGE", "INTERIOR_LINE"])]
print(f"DL pass-rush snaps {len(rush)}; corner found {rush.peak_lat.notna().mean():.2f}")
print(rush.groupby("lined_up_position")[["peak_lat", "lean_deg", "apex_speed", "radius", "t_corner", "pressure"]].median().round(2).to_string())
print("turn dir share L by alignment:", rush.groupby("lined_up_position").turn_dir.apply(lambda x: (x == "L").mean()).round(2).to_dict())
ol = out[out.grp == "OL"]
runs = ol[ol.play_type == "RUN"]
print(f"OL run snaps {len(runs)}; pull rate {runs.pull.mean():.3f}; climb rate {runs.climb.mean():.3f}")
print("OL pull plays medians:", runs[runs.pull][["move_peak_lat", "move_apex_speed", "yards_gained"]].median().round(2).to_dict())
dlr = dl[dl.play_type == "RUN"]
print("DL run: stall/release medians", dlr[["stall_speed", "release_gain"]].median().round(2).to_dict(), "tackle_any", dlr.tackle_any.mean().round(3))
