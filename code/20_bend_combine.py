"""Combine BEND, by turn direction, from the drills that contain real turns at speed.

Drills: DL Run the Hoop (figure-8: one loop each direction), DL Pass Rush Drill (one attempt per side),
OL pull drills (fold-block right, long-pull right, deep-short-pull left).
Per frame: lateral (centripetal) accel a_lat = (vx*ay - vy*ax)/s, signed (+ = turning left / counter-clockwise),
smoothed with a 3-frame rolling mean so a single noisy frame cannot set the peak.
Per turn (contiguous frames with |a_lat| > 1.5 and s > 1.5 yd/s, same sign, >= 0.3 s):
  peak_lat   max |a_lat| (yd/s^2)          lean_deg  atan(peak_lat / g), g = 10.73 yd/s^2
  apex_speed speed at the peak             radius    apex_speed^2 / peak_lat (yd)
  entry_speed speed 0.3 s before the turn  retention apex_speed / entry_speed
Per player x drill-family x direction: best turn (max peak_lat) averaged over attempts.
Output data/bend/combine_bend_turns.parquet (all turns) and combine_bend_player.parquet (wide).
"""
import os
import numpy as np
import pandas as pd

ROOT = "/Users/ahanjain/bdb_ssl"
os.makedirs(f"{ROOT}/data/bend", exist_ok=True)
G = 10.73
FAMILY = {"RUN_THE_HOOP_DRILL": "hoop", "PASS_RUSH_DRILL": "rushdrill", "OL_PULL_DRILL_FOLD_BLOCK_RIGHT": "pull",
          "OL_PULL_DRILL_LONG_PULL_RIGHT": "pull", "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT": "pull"}
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[(ct.entity_type == "PLAYER") & ct.drill_name.isin(FAMILY) & ct.drill_type.isin(["SKILL_DRILLS_DL", "SKILL_DRILLS_OL"])]
ct = ct.sort_values(["event_id", "time"])


def lat_series(t, s, dirdeg):
    th = np.radians(dirdeg)
    vx, vy = s * np.sin(th), s * np.cos(th)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    lat = np.where(s > 0.5, (vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    return pd.Series(lat).rolling(3, center=True, min_periods=1).mean().values


def turns(t, s, lat, min_len=3):
    out = []
    on = (np.abs(lat) > 1.5) & (s > 1.5)
    sign = np.sign(lat)
    i = 0
    while i < len(s):
        if not on[i]:
            i += 1; continue
        j = i
        while j + 1 < len(s) and on[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        if j - i + 1 >= min_len:
            seg = np.arange(i, j + 1)
            k = seg[np.argmax(np.abs(lat[seg]))]
            e = max(0, i - 3)
            pk = abs(lat[k])
            out.append({"dir": "L" if lat[k] > 0 else "R", "peak_lat": pk, "lean_deg": np.degrees(np.arctan(pk / G)),
                        "sust_lat": np.abs(lat[seg]).mean(), "sust_speed": s[seg].mean(),
                        "heading_change": np.degrees(np.abs(lat[seg] / np.maximum(s[seg], 0.5)).sum() * np.median(np.diff(t))),
                        "apex_speed": s[k], "radius": s[k] ** 2 / pk, "entry_speed": s[e],
                        "retention": s[k] / max(s[e], 0.5), "t_start": t[i], "dur": t[j] - t[i]})
        i = j + 1
    return out


rows = []
for eid, g in ct.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 8:
        continue
    s = g.s.values.astype(float)
    lat = lat_series(t, s, g.dir.values.astype(float))
    for tr in turns(t, s, lat):
        rows.append({"nfl_id": g.nfl_id.iat[0], "event_id": eid, "drill": g.drill_name.iat[0],
                     "family": FAMILY[g.drill_name.iat[0]], **tr})
T = pd.DataFrame(rows)
T.to_parquet(f"{ROOT}/data/bend/combine_bend_turns.parquet", index=False)
# best turn per attempt x direction, then mean over attempts
best = T.loc[T.groupby(["nfl_id", "event_id", "dir"]).peak_lat.idxmax()]
MET = ["peak_lat", "lean_deg", "apex_speed", "radius", "retention", "sust_lat", "sust_speed", "heading_change", "dur"]
P = best.groupby(["nfl_id", "family", "dir"])[MET].mean()
W = P.unstack(["family", "dir"])
W.columns = [f"c_{fam}_{d}_{m}" for m, fam, d in W.columns]
for fam in ["hoop", "rushdrill", "pull"]:
    for m in MET:
        cl, cr = f"c_{fam}_L_{m}", f"c_{fam}_R_{m}"
        if cl in W and cr in W:
            W[f"c_{fam}_{m}"] = W[[cl, cr]].mean(axis=1)
            W[f"c_{fam}_asym_{m}"] = W[cl] - W[cr]
# whole-attempt hoop summaries: time to complete and mean speed while moving (fixed hoop geometry)
hp = []
for eid, g in ct[ct.drill_name == "RUN_THE_HOOP_DRILL"].groupby("event_id"):
    s_ = g.s.values.astype(float); t_ = (g.time - g.time.iat[0]).dt.total_seconds().values
    on = int(np.argmax(s_ >= 1.0)); mv = s_ >= 1.0
    hp.append({"nfl_id": g.nfl_id.iat[0], "c_hoop_time": t_[-1] - t_[on], "c_hoop_mean_speed": s_[on:].mean()})
W = W.join(pd.DataFrame(hp).groupby("nfl_id").mean())
W.to_parquet(f"{ROOT}/data/bend/combine_bend_player.parquet")

print("turns per attempt by drill:\n", T.groupby("drill").apply(lambda d: d.groupby("event_id").size().median(), include_groups=False).to_string())
print("\nmedian turn metrics by family x direction:")
print(best.groupby(["family", "dir"])[["peak_lat", "lean_deg", "apex_speed", "radius", "retention"]].median().round(2).to_string())
# reliability across directions / drills (parallel forms) for the overall bend score
for a, b in [("c_hoop_L_peak_lat", "c_hoop_R_peak_lat"), ("c_rushdrill_L_peak_lat", "c_rushdrill_R_peak_lat"),
             ("c_hoop_peak_lat", "c_rushdrill_peak_lat"), ("c_pull_L_peak_lat", "c_pull_R_peak_lat"),
             ("c_hoop_apex_speed", "c_rushdrill_apex_speed")]:
    if a in W and b in W:
        m = W[[a, b]].dropna()
        print(f"r({a}, {b}) = {np.corrcoef(m[a], m[b])[0, 1]:.2f}  n {len(m)}")
