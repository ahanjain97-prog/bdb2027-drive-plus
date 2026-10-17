"""Generic transition-phase decomposition for one combine attempt (generalizes the frozen CB drive logic).

Initial line = median heading of the first 4 frames at >= 2 yd/s after onset; turn start j0 = first later frame at
>= start_speed whose heading deviates > 10 deg; the max deviation within 1.5 s must lie in [dev_lo, dev_hi];
turn end j1 = first frame at 90% of it. Drive top = running max from j1 (<= 2 s, stop 0.5 yd/s below the max).
Returns speeds at fixed times after j1 (NaN past the clip end), plus the clip time left after j1.
"""
import numpy as np
from scipy.signal import savgol_filter

TAUS = (0.2, 0.4, 0.6, 0.8, 1.0)


def phase(t, x, y, dev_lo=25, dev_hi=90, start_speed=3.0):
    if len(t) < 12:
        return None
    dt = np.median(np.diff(t))
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    if not (s >= 1.0).any():
        return None
    on = int(np.argmax(s >= 1.0))
    run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
    if len(run) < 4:
        return None
    dev = np.abs(hd - np.median(hd[run[:4]]))
    st = np.where((np.arange(len(s)) > run[3]) & (s >= start_speed) & (dev > 10))[0]
    if not len(st):
        return None
    j0 = st[0]; win = slice(j0, min(j0 + 15, len(s))); dmax = dev[win].max()
    if dmax < dev_lo or dmax > dev_hi:
        return None
    j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax))
    k, jend, top = j1, min(j1 + 20, len(s) - 1), s[j1]
    while k < jend:
        k += 1
        if s[k] > top:
            top = s[k]
        elif s[k] < top - 0.5:
            break
    r = {"v_entry": s[max(j0 - 3, 0):j0].mean() if j0 > 0 else np.nan, "v_out": s[j1], "top": top,
         "turn_angle": dmax, "clip_after": t[-1] - t[j1]}
    for tau in TAUS:
        r[f"v_{int(round(tau * 10)):02d}"] = np.interp(t[j1] + tau, t, s, right=np.nan)
    return r
