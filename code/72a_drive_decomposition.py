"""Where inside the transition drill does CB Transition Drive Speed live?

Per attempt of BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION, with the FROZEN metric's own phase marks (turn start j0,
turn end j1; prereg/cb_drive_speed.py logic re-run here, untouched):
  entry        backpedal speed, mean over the 0.3 s before the turn starts
  turn_time    j0 -> j1 (s)                      turn_min    min speed from j0 to j1 + 0.2 s
  v_out        speed at the turn's end (j1)      -- "coming out of the break"
  v_02 .. v_15 speed 0.2 / 0.4 / 0.6 / 0.8 / 1.0 / 1.5 s after j1 (one step ~ 0.25 s, so v_04 ~ after two steps)
  step2_gain   v_05 - v_out (first two steps)    late_gain   top - v_05 (after the second step)
  top          frozen drive top speed            t_top       time from j1 to that top
  d_05, d_10   ground covered 0.5 / 1.0 s after j1 (yd)      d0_15  ground covered from the turn START, 1.5 s
  straight     path length / straight-line distance over j1 .. j1 + 1.0 s (1 = dead straight)
Reliability: attempt 1 vs attempt 2 (r) and of the 2-attempt mean (Spearman-Brown). Every player value is the mean
over valid attempts, then z-scored WITHIN combine year (clip-length trap). Truncation check: share of attempts whose
clip ends before j1 + 1.5 s, by year. Output data/features/cb_drive_parts.parquet (player) + _attempts.parquet.
"""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
ROOT = "/Users/ahanjain/bdb_ssl"
DRILL = "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
d = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == DRILL)].copy()
d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"])


def parts(t, x, y):
    if len(t) < 12:
        return None
    dt = np.median(np.diff(t))
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    on = int(np.argmax(s >= 1.0))
    run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
    if len(run) < 4:
        return None
    dev = np.abs(hd - np.median(hd[run[:4]]))
    st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
    if not len(st):
        return None
    j0 = st[0]; win = slice(j0, min(j0 + 15, len(s))); dmax = dev[win].max()
    if dmax < 25 or dmax > 90:
        return None
    j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax))
    k, jend, top, ktop = j1, min(j1 + 20, len(s) - 1), s[j1], j1
    while k < jend:
        k += 1
        if s[k] > top:
            top, ktop = s[k], k
        elif s[k] < top - 0.5:
            break
    vt = lambda tau: np.interp(t[j1] + tau, t, s, right=np.nan)
    pre = (t >= t[j0] - 0.3) & (t < t[j0])
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    dist = lambda ta, tb: (np.interp(tb, t, cum, right=np.nan) - np.interp(ta, t, cum))
    e10 = t[j1] + 1.0
    if e10 <= t[-1]:
        xe, ye = np.interp(e10, t, x), np.interp(e10, t, y)
        straight = dist(t[j1], e10) / max(np.hypot(xe - x[j1], ye - y[j1]), 1e-6)
    else:
        straight = np.nan
    r = {"entry": s[pre].mean() if pre.any() else np.nan, "turn_time": t[j1] - t[j0], "turn_min": s[j0:j1 + 3].min(),
         "v_out": s[j1], "top": top, "t_top": t[ktop] - t[j1], "d_05": dist(t[j1], t[j1] + 0.5), "d_10": dist(t[j1], t[j1] + 1.0),
         "d0_15": dist(t[j0], t[j0] + 1.5), "straight": straight, "turn_angle": dmax, "clip_after_j1": t[-1] - t[j1]}
    for tau in (0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5):
        r[f"v_{int(round(tau * 10)):02d}"] = vt(tau)
    r["step2_gain"] = r["v_05"] - r["v_out"]; r["late_gain"] = top - r["v_05"]
    return r


rows = []
for eid, g in d.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    p = parts(t, g.x.values.astype(float), g.y.values.astype(float))
    if p:
        rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "yr": g.draft_year.iat[0], "t0": g.time.iat[0], **p})
A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"]); A["att"] = A.groupby("nfl_id").cumcount()
A.to_parquet(f"{ROOT}/data/features/cb_drive_parts_attempts.parquet", index=False)
FE = ["entry", "turn_time", "turn_min", "v_out", "v_02", "v_04", "v_05", "v_06", "v_08", "v_10", "v_15", "step2_gain", "late_gain",
      "top", "t_top", "d_05", "d_10", "d0_15", "straight", "turn_angle"]
print(f"valid attempts {len(A)}, players {A.nfl_id.nunique()} | clip ends before j1 + 1.5 s, share by year:",
      A.assign(tr=A.clip_after_j1 < 1.5).groupby("yr").tr.mean().round(3).to_dict())
print("median by year (artifact check):\n", A.groupby("yr")[["entry", "turn_time", "v_out", "v_05", "top", "t_top", "d_10"]].median().round(2).to_string())
two = A[A.att <= 1].pivot(index="nfl_id", columns="att", values=FE).dropna(how="any", subset=[("top", 0), ("top", 1)])
rel = {}
for f in FE:
    h = two[f].dropna()
    if len(h) >= 20:
        r = np.corrcoef(h[0], h[1])[0, 1]; rel[f] = (round(r, 2), round(2 * r / (1 + r), 2), len(h))
R = pd.DataFrame(rel, index=["attempt_r", "reliability_of_mean", "n"]).T
print("\nreliability (attempt 1 vs 2; Spearman-Brown for the 2-attempt mean):\n", R.to_string())
P = A.groupby("nfl_id")[FE].mean()
yr = base.draft_year.reindex(P.index)
Z = (P - P.groupby(yr).transform("mean")) / P.groupby(yr).transform("std")
Z["n_att"] = A.groupby("nfl_id").size()
Z.to_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet"); R.to_csv(f"{ROOT}/results/72a_reliability.csv")
cbs = base.index[base.nfl_position == "CB"]
print("\ncorrelation with frozen top (CBs, within-year z):", Z.reindex(cbs).corr()["top"].drop("top").round(2).to_dict())
st = base.reindex(cbs)[["forty", "ten_yd_split", "three_cone", "short_shuttle", "combine_weight"]]
print("correlation with stopwatch/size (CBs):\n", Z.reindex(cbs)[["entry", "turn_time", "v_out", "v_05", "step2_gain", "late_gain", "top", "d_10"]].join(st).corr().loc[
    ["entry", "turn_time", "v_out", "v_05", "step2_gain", "late_gain", "top", "d_10"], st.columns].round(2).to_string())
