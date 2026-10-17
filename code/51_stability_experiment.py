"""Can better estimation make combine mechanics repeatable? Test-retest (attempt 1 vs 2, drills with >= 15 repeaters)
for each feature under three estimators:
  V0  vendor speed, 3-frame rolling mean, frame differences (current factory)
  V1  Savitzky-Golay (window 7, order 2) on x/y -> velocity, speed, acceleration analytically
  V2  V1 + multi-frame line fits: braking rate = slope of speed over the frames between 85% and 35% of entry speed;
      re-acceleration = slope over 0.1-0.6 s after the minimum; start acceleration = slope over the first 1.0 s
Reports median test-retest per feature x estimator and the per-drill table.
"""
import warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[ct.entity_type == "PLAYER"].copy(); ct["time"] = pd.to_datetime(ct.time)
ct["drill"] = ct.drill_type.str.replace("SKILL_DRILLS_", "") + ":" + ct.drill_name
rep_drills = ct.groupby(["drill", "nfl_id"]).event_id.nunique().reset_index()
rep_drills = rep_drills[rep_drills.event_id >= 2].groupby("drill").size()
rep_drills = rep_drills[rep_drills >= 15].index
ct = ct[ct.drill.isin(rep_drills)].sort_values(["event_id", "time"])
print("drills with >= 15 repeaters:", list(rep_drills))


def slope(t, v):
    if len(t) < 3:
        return np.nan
    return np.polyfit(t, v, 1)[0]


def mech(t, x, y, s_vendor, mode):
    if mode == "V0":
        s = pd.Series(s_vendor).rolling(3, center=True, min_periods=1).mean().values
        al = np.gradient(s, t)
        sx = sy = None
    else:
        w = 7 if len(t) >= 9 else (len(t) // 2) * 2 - 1
        if w < 5:
            return None
        dt = np.median(np.diff(t))
        vx = savgol_filter(x, w, 2, deriv=1, delta=dt); vy = savgol_filter(y, w, 2, deriv=1, delta=dt)
        ax = savgol_filter(x, w, 2, deriv=2, delta=dt); ay = savgol_filter(y, w, 2, deriv=2, delta=dt)
        s = np.hypot(vx, vy)
        al = (vx * ax + vy * ay) / np.maximum(s, 1e-6)
    if s.max() < 1:
        return None
    on = int(np.argmax(s >= 1.0)); f = {}
    f["top_speed"] = s.max()
    j1 = np.searchsorted(t, t[on] + 1.0)
    f["start_accel"] = slope(t[on:j1 + 1], s[on:j1 + 1]) if mode == "V2" else ((s[min(j1, len(s) - 1)] - s[on]) / 1.0)
    # first break
    pk = on
    for j in range(on + 1, len(s) - 1):
        if s[j] > s[pk]:
            pk = j
        if s[j] <= s[j - 1] and s[j] < s[j + 1] and s[pk] >= 3.0 and s[j] <= 0.6 * s[pk]:
            e = s[pk]; f["entry_speed"] = e; f["min_speed"] = s[j]
            if mode == "V2":
                sel = np.arange(pk, j + 1); sel = sel[(s[sel] <= .85 * e) & (s[sel] >= .35 * e)]
                f["brake_rate"] = -slope(t[sel], s[sel]) if len(sel) >= 3 else np.nan
                sel2 = np.where((t >= t[j] + .1) & (t <= t[j] + .6))[0]
                f["reaccel_rate"] = slope(t[sel2], s[sel2]) if len(sel2) >= 3 else np.nan
            else:
                b0 = pk + int(np.argmax(s[pk:j + 1] <= .9 * e)); f["brake_rate"] = (e - s[j]) / max(t[j] - t[b0], .1)
                je = min(np.searchsorted(t, t[j] + .5), len(s) - 1); f["reaccel_rate"] = (s[je] - s[j]) / .5
            f["brake_dist_per_speed"] = np.hypot(np.diff(x[pk:j + 1]), np.diff(y[pk:j + 1])).sum() / e
            break
    return f


rows = []
for eid, g in ct.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 9:
        continue
    for mode in ("V0", "V1", "V2"):
        f = mech(t, g.x.values.astype(float), g.y.values.astype(float), g.s.values.astype(float), mode)
        if f:
            rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "t0": g.time.iat[0], "mode": mode, **f})
A = pd.DataFrame(rows).sort_values(["mode", "nfl_id", "drill", "t0"])
A["att"] = A.groupby(["mode", "nfl_id", "drill"]).cumcount()
FE = ["top_speed", "start_accel", "entry_speed", "min_speed", "brake_rate", "reaccel_rate", "brake_dist_per_speed"]
out = []
for (mode, d_), g in A[A.att <= 1].groupby(["mode", "drill"]):
    w = g.pivot_table(index="nfl_id", columns="att", values=FE)
    for f in FE:
        if (f, 0) in w and (f, 1) in w:
            mm = w[[(f, 0), (f, 1)]].dropna()
            if len(mm) >= 15:
                out.append({"mode": mode, "drill": d_, "feature": f, "n": len(mm), "r": np.corrcoef(mm.iloc[:, 0], mm.iloc[:, 1])[0, 1]})
O = pd.DataFrame(out)
O.to_csv(f"{ROOT}/results/51_stability_experiment.csv", index=False)
pd.set_option("display.width", 200)
print("\nmedian test-retest across drills (feature x estimator):")
print(O.pivot_table(index="feature", columns="mode", values="r", aggfunc="median").round(2).to_string())
print("\nper drill, V2:")
print(O[O["mode"] == "V2"].pivot_table(index="drill", columns="feature", values="r").round(2).to_string())
