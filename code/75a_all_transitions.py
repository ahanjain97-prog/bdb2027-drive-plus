"""Drive+ step 1: every transition in every DB drill -> post-turn drive speed.

Turn detection (all DB drills, CBs + safeties): Savitzky-Golay heading; turn frames = |turn rate| > 60 deg/s at > 1 yd/s
(1-frame gaps bridged); a transition = a turn segment with >= 25 deg total heading change, entered at >= 2 yd/s
(moving, not a standing start). For each: turn end j1; the straight run lasts until the next transition starts, 1.5 s,
or the clip end.
  v_out          speed at the turn end        v_04 / v_06 / v_08 / v_10   speed 0.4..1.0 s after it, only if the run lasts
  post_top       max speed in the straight run after the turn (the drive)   run_time  length of that run
  angle, side    heading change and its direction (left/right)
Per drill: z within drill x combine year x angle tercile. Player value per drill = mean over his transitions.
Reliability per drill: left-turn mean vs right-turn mean within the same player (Spearman-Brown) -- avoids the
alternating-turn artifact of odd/even events; for the 45-degree drill also attempt 1 vs 2 of the frozen metric.
Convergent validity: per-drill post_top vs the frozen 45-degree drive speed (CBs). Output data/features/db_transitions.parquet
and db_drive_by_drill.parquet.
"""
import warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
dbs = base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]
DR = ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "W_DRILL", "BOX_DRILL", "TERYL_AUSTIN_DRILL_1",
      "TERYL_AUSTIN_DRILL_2", "GAUNTLET_DRILL", "LINE_DRILL", "SHORT_SHUTTLE", "THREE_CONE_DRILL"]
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[(ct.entity_type == "PLAYER") & ct.nfl_id.isin(dbs)].copy()
ct["drill"] = ct.drill_name.replace({"LINE": "LINE_DRILL"}); ct = ct[ct.drill.isin(DR)]
ct["time"] = pd.to_datetime(ct.time); ct = ct.sort_values(["event_id", "time"])
rows = []
for eid, g in ct.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 15:
        continue
    dt = np.median(np.diff(t)); x, y = g.x.values.astype(float), g.y.values.astype(float)
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    tr = np.abs(np.gradient(hd, t)); tr = np.convolve(tr, np.ones(3) / 3, mode="same")
    cand = (tr > 60) & (s > 1.0)
    for i in range(1, len(cand) - 1):                                  # bridge 1-frame gaps
        if not cand[i] and cand[i - 1] and cand[i + 1]:
            cand[i] = True
    segs, i = [], 0
    while i < len(cand):
        if cand[i]:
            j = i
            while j + 1 < len(cand) and cand[j + 1]:
                j += 1
            a_, b_ = max(i - 1, 0), min(j + 1, len(s) - 1)
            if abs(hd[b_] - hd[a_]) >= 25 and s[max(a_ - 5, 0):a_ + 1].max() >= 2.0:
                segs.append((a_, b_, hd[b_] - hd[a_]))
            i = j + 1
        else:
            i += 1
    for n_, (a_, b_, dh) in enumerate(segs):
        nxt = segs[n_ + 1][0] if n_ + 1 < len(segs) else len(s) - 1
        end = min(nxt, b_ + 15, len(s) - 1)
        if end - b_ < 3:
            continue
        r = {"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "yr": g.draft_year.iat[0], "k": n_, "angle": abs(dh),
             "side": np.sign(dh), "v_entry": s[max(a_ - 3, 0)], "v_out": s[b_], "post_top": s[b_ + 1:end + 1].max(), "run_time": (end - b_) * dt}
        for kk in (4, 6, 8, 10):
            r[f"v_{kk:02d}"] = s[b_ + kk] if b_ + kk <= end else np.nan
        rows.append(r)
T = pd.DataFrame(rows)
T["ang3"] = T.groupby("drill").angle.transform(lambda a: pd.qcut(a, 3, labels=False, duplicates="drop"))
FE = ["v_out", "v_04", "v_06", "v_08", "v_10", "post_top"]
for f in FE:
    grp = T.groupby(["drill", "yr", "ang3"])[f]; T[f + "_z"] = (T[f] - grp.transform("mean")) / grp.transform("std")
T.to_parquet(f"{ROOT}/data/features/db_transitions.parquet", index=False)
pd.set_option("display.width", 240)
print("transitions per drill (median per attempt, angle median, run time median, share with a 1.0 s run):")
print(T.groupby("drill").agg(transitions=("k", "size"), per_attempt=("event_id", lambda e: e.value_counts().median()), angle=("angle", "median"),
                             run=("run_time", "median"), has_v10=("v_10", lambda v: v.notna().mean())).round(2).to_string())
import sys
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
drive = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))
yr = base.draft_year.reindex(drive.index); frozen = (drive - drive.groupby(yr).transform("mean")) / drive.groupby(yr).transform("std")
cbs = base.index[base.nfl_position == "CB"]
out = []
for dr, g in T.groupby("drill"):
    rec = {"drill": dr}
    for f in ["post_top", "v_08", "v_04"]:
        lr = g.dropna(subset=[f + "_z"]).groupby(["nfl_id", "side"])[f + "_z"].mean().unstack().dropna()
        if len(lr) >= 15 and lr.shape[1] == 2:
            r = np.corrcoef(lr.iloc[:, 0], lr.iloc[:, 1])[0, 1]; rec[f"rel_LR_{f}"] = 2 * r / (1 + r)
        pm = g.groupby("nfl_id")[f + "_z"].mean()
        j = pd.concat([pm, frozen], axis=1, join="inner").reindex([i for i in cbs if i in pm.index and i in frozen.index]).dropna()
        rec[f"r_with_45_{f}"] = j.corr().iloc[0, 1] if len(j) >= 15 else np.nan
    rec["CBs"] = g[g.nfl_id.isin(cbs)].nfl_id.nunique()
    out.append(rec)
Rel = pd.DataFrame(out).set_index("drill")
print("\nper drill: left-vs-right-turn reliability (Spearman-Brown) and correlation with the frozen 45-degree drive speed (CBs):")
print(Rel.round(2).to_string())
Rel.to_csv(f"{ROOT}/results/75a_drill_reliability.csv")
W = T.groupby(["nfl_id", "drill"])[[f + "_z" for f in FE]].mean().unstack("drill")
W.columns = [f"{d}|{f[:-2]}" for f, d in W.columns]
W.to_parquet(f"{ROOT}/data/features/db_drive_by_drill.parquet")
print("\ncross-drill correlations of post-turn drive (post_top), CBs:")
pt = W[[c for c in W.columns if c.endswith("|post_top")]].reindex(cbs)
pt.columns = [c.split("|")[0][:14] for c in pt.columns]
print(pt.corr(min_periods=15).round(2).to_string())
