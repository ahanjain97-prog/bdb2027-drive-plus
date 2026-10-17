"""FROZEN METRIC (2026-10-12): Hoop Turn Speed, from NFL Combine Run-the-Hoop tracking.

Do not edit. The pre-registered 2026 test (PREREG_hoop_turn_speed.md) uses this exact file; its SHA-256 is recorded
there. Input: combine_tracking.csv/parquet rows (columns time, s, dir, nfl_id, event_id, drill_name, entity_type).

Definition
  1. Keep PLAYER rows of drill RUN_THE_HOOP_DRILL; sort each attempt (event_id) by time.
  2. Lateral (centripetal) acceleration per frame: vx = s*sin(dir), vy = s*cos(dir) (dir in degrees, 0 = +y,
     clockwise); ax, ay = numpy.gradient of vx, vy over time; a_lat = (vx*ay - vy*ax)/s for s > 0.5, else 0;
     then a centered 3-frame rolling mean.
  3. Turn segments: maximal runs of consecutive frames with |a_lat| > 1.5 yd/s^2, s > 1.5 yd/s and the same sign
     of a_lat, at least 3 frames long.
  4. Attempt value = mean speed over the turn-segment frames, segments weighted by duration (frames - 1) * dt.
  5. Player value = mean over his attempts with >= 1 turn segment.  Units: yd/s.
"""
import numpy as np
import pandas as pd

LAT_MIN, SPEED_MIN, MIN_FRAMES = 1.5, 1.5, 3


def _lateral(t, s, dir_deg):
    th = np.radians(dir_deg)
    vx, vy = s * np.sin(th), s * np.cos(th)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    lat = np.where(s > 0.5, (vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    return pd.Series(lat).rolling(3, center=True, min_periods=1).mean().values


def attempt_value(t, s, dir_deg):
    lat = _lateral(t, s, dir_deg)
    on = (np.abs(lat) > LAT_MIN) & (s > SPEED_MIN)
    sign = np.sign(lat)
    speeds, weights = [], []
    i = 0
    while i < len(s):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and on[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        if j - i + 1 >= MIN_FRAMES:
            speeds.append(s[i:j + 1].mean())
            weights.append(t[j] - t[i])
        i = j + 1
    if not speeds or sum(weights) <= 0:
        return np.nan
    return float(np.average(speeds, weights=weights))


def hoop_turn_speed(combine: pd.DataFrame) -> pd.Series:
    d = combine[(combine.entity_type == "PLAYER") & (combine.drill_name == "RUN_THE_HOOP_DRILL")].copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values(["event_id", "time"])
    vals = []
    for eid, g in d.groupby("event_id"):
        if len(g) < 8:
            continue
        t = (g.time - g.time.iat[0]).dt.total_seconds().values
        v = attempt_value(t, g.s.values.astype(float), g.dir.values.astype(float))
        if np.isfinite(v):
            vals.append((g.nfl_id.iat[0], v))
    return pd.DataFrame(vals, columns=["nfl_id", "hoop_turn_speed"]).groupby("nfl_id").hoop_turn_speed.mean()
