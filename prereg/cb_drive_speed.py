"""FROZEN METRIC (2026-10-12): CB Transition Drive Speed, from the NFL Combine DB drill
BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION.

Do not edit. The pre-registered 2026 test (PREREG_cb_drive_speed.md) uses this exact file; its SHA-256 is recorded there.

Definition (per attempt, PLAYER rows sorted by time):
  1. Savitzky-Golay (window 7, order 2) derivatives of x and y -> vx, vy; speed s = |v|; heading = atan2(vx, vy),
     unwrapped, in degrees.
  2. Backpedal line = median heading of the first 4 frames after onset (first frame >= 1 yd/s) moving at >= 2 yd/s.
  3. Turn start j0 = first later frame at >= 3 yd/s whose heading deviates > 10 deg from the line. Within the next
     1.5 s the maximum deviation must be 25-90 deg (a ~45-degree break; otherwise the attempt is skipped).
     Turn end j1 = first frame reaching 90% of that maximum deviation.
  4. Drive: from j1 forward up to 20 frames, track the running maximum speed; stop when speed falls 0.5 yd/s below it.
     Attempt value = that running maximum (yd/s).
  5. Player value = mean over his valid attempts.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DRILL = "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"


def attempt_value(t, x, y):
    if len(t) < 12:
        return np.nan
    dt = np.median(np.diff(t))
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    on = int(np.argmax(s >= 1.0))
    run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
    if len(run) < 4:
        return np.nan
    dev = np.abs(hd - np.median(hd[run[:4]]))
    st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
    if not len(st):
        return np.nan
    j0 = st[0]
    win = slice(j0, min(j0 + 15, len(s)))
    dmax = dev[win].max()
    if dmax < 25 or dmax > 90:
        return np.nan
    j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax))
    k, jend, top = j1, min(j1 + 20, len(s) - 1), s[j1]
    while k < jend:
        k += 1
        if s[k] > top:
            top = s[k]
        elif s[k] < top - 0.5:
            break
    return float(top)


def cb_drive_speed(combine: pd.DataFrame) -> pd.Series:
    d = combine[(combine.entity_type == "PLAYER") & (combine.drill_name == DRILL)].copy()
    d["time"] = pd.to_datetime(d["time"]); d = d.sort_values(["event_id", "time"])
    vals = []
    for eid, g in d.groupby("event_id"):
        t = (g.time - g.time.iat[0]).dt.total_seconds().values
        v = attempt_value(t, g.x.values.astype(float), g.y.values.astype(float))
        if np.isfinite(v):
            vals.append((g.nfl_id.iat[0], v))
    return pd.DataFrame(vals, columns=["nfl_id", "cb_drive_speed"]).groupby("nfl_id").cb_drive_speed.mean()
