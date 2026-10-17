"""Full CB combine features, v2 -- comparable across combine years.

Why: clip lengths differ by combine year for some drills (45-degree transition drill: 11.3 s in 2023, 5.7 s in 2024,
9.8 s in 2025), so v1's onset->clip-end normalized curves and whole-clip mechanics encoded the YEAR, and the 2024 CB
class differs in completion over expected. v2:
  1. Every attempt is truncated to a drill-specific common window after onset = 0.9 x the shortest yearly median clip
     length for that drill, so every year contributes the same part of the drill.
  2. All 41 factory mechanics are recomputed on the truncated attempt.
  3. Curves on REAL time: speed / turn rate / along-track accel / |lateral accel| every 0.2 s from onset to the
     common window; for the transition drill also every 0.1 s from 1.0 s before to 2.0 s after the turn start.
  4. Every feature is z-scored WITHIN combine year (over all DBs who ran it that year).
Output data/features/cb_full_views_v2.parquet, cb_full_player_v2.parquet
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import DRILL
src = open(f"{ROOT}/code/50_feature_factory.py").read()
ns = {}
exec(src.split("rows, EV = []")[0].split("ct = pd.read_parquet")[0], ns)        # imports + helper defs
fdef = src[src.index("def roll("):src.index("rows, EV = []")]
exec(fdef, ns); ns["EVENTS"] = []
features = ns["features"]
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
dbs = base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[ct.nfl_id.isin(dbs)].copy(); ct["time"] = pd.to_datetime(ct.time)
ct["drill"] = np.where(ct.drill_type.str.startswith("SKILL_DRILLS"), ct.drill_name, ct.drill_name)
ct["drill"] = ct.drill.replace({"LINE": "LINE_DRILL"})
ct = ct.sort_values(["event_id", "entity_type", "time"])
pl = ct[ct.entity_type == "PLAYER"]
# common window per drill
dur = []
for eid, g in pl.groupby("event_id"):
    s = g.s.values
    on = int(np.argmax(s >= 1.0)) if (s >= 1.0).any() else 0
    dur.append({"drill": g.drill.iat[0], "yr": g.draft_year.iat[0], "d": (g.time.iat[-1] - g.time.iat[on]).total_seconds()})
D = pd.DataFrame(dur)
WIN = (D.groupby(["drill", "yr"]).d.median().groupby("drill").min() * 0.9).round(1)
print("common window after onset (s):", WIN.to_dict())
balls = {e: b for e, b in ct[ct.entity_type == "BALL"].groupby("event_id")}
rows = []
for eid, g in pl.groupby("event_id"):
    if len(g) < 9:
        continue
    s0 = g.s.values
    if not (s0 >= 1.0).any():
        continue
    on = int(np.argmax(s0 >= 1.0)); t_on = g.time.iat[on]
    gt = g[(g.time <= t_on + pd.Timedelta(seconds=float(WIN[g.drill.iat[0]])))]
    if len(gt) < 9:
        continue
    ns["EVENTS"].clear()
    f = features(gt, balls.get(eid)) or {}
    # real-time curves from the full clip (sampled only inside the common window)
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; dt = np.median(np.diff(t))
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    ax = savgol_filter(x, 7, 2, deriv=2, delta=dt); ay = savgol_filter(y, 7, 2, deriv=2, delta=dt)
    s = np.hypot(vx, vy); al = (vx * ax + vy * ay) / np.maximum(s, 1e-6)
    lat = np.where(s > 0.5, np.abs(vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    hd = np.unwrap(np.arctan2(vx, vy)); tr = np.where(s > 0.5, np.abs(np.gradient(hd, t)) * 180 / np.pi, 0.0)
    ton = t[on]; W = float(WIN[g.drill.iat[0]])
    grid = np.arange(0, W + 1e-9, 0.2)
    for nm, sig in (("speed", s), ("turnrate", tr), ("along", al), ("lateral", lat)):
        vals = np.interp(ton + grid, t, sig, left=np.nan, right=np.nan)
        for i, v in enumerate(vals):
            f[f"rt_{nm}_{i * 0.2:.1f}s"] = v
    if g.drill.iat[0] == DRILL:
        run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
        if len(run) >= 4:
            hdd = np.degrees(hd); dev = np.abs(hdd - np.median(hdd[run[:4]]))
            st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
            if len(st) and 25 <= dev[st[0]:st[0] + 15].max() <= 90:
                j0 = st[0]
                for nm, sig in (("speed", s), ("along", al), ("lateral", lat), ("turnrate", tr)):
                    vals = np.interp(t[j0] + np.arange(-1.0, 2.0001, 0.1), t, sig, left=np.nan, right=np.nan)
                    for i, v in enumerate(vals):
                        f[f"turn_{nm}_{-1.0 + i * 0.1:+.1f}s"] = v
    rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "yr": g.draft_year.iat[0], "t0": g.time.iat[0], **f})
A = pd.DataFrame(rows)
FE = [c for c in A.columns if c not in ("event_id", "nfl_id", "drill", "yr", "t0")]
# within-year z-scores per drill x feature
for f in FE:
    grp = A.groupby(["drill", "yr"])[f]
    A[f] = (A[f] - grp.transform("mean")) / grp.transform("std")
A = A.sort_values(["nfl_id", "drill", "t0"]); A["att"] = A.groupby(["nfl_id", "drill"]).cumcount()
nviews = A.groupby("nfl_id").att.max() + 1
out = []
for pid, g in A.groupby("nfl_id"):
    for v in range(int(nviews[pid])):
        rec = {"nfl_id": pid, "view": v}
        for d_, gd in g.groupby("drill"):
            a = gd[gd.att == min(v, gd.att.max())].iloc[0]
            for f in FE:
                if pd.notna(a[f]) and np.isfinite(a[f]):
                    rec[f"{d_}|{f}"] = float(a[f])
        out.append(rec)
Vw = pd.DataFrame(out)
Vw = Vw[Vw.nfl_id.isin(base.index[base.nfl_position == "CB"])]
Vw = Vw.loc[:, Vw.notna().sum() >= 10]
Vw.to_parquet(f"{ROOT}/data/features/cb_full_views_v2.parquet", index=False)
Vw.drop(columns="view").groupby("nfl_id").mean().to_parquet(f"{ROOT}/data/features/cb_full_player_v2.parquet")
cols = Vw.columns[2:]
print(f"CBs {Vw.nfl_id.nunique()}, views {len(Vw)}, features {len(cols)} (mechanics {sum(('|rt_' not in c) and ('|turn_' not in c) for c in cols)}, "
      f"real-time curves {sum('|rt_' in c for c in cols)}, turn-anchored curves {sum('|turn_' in c for c in cols)})")
# leak check: can the features still identify the combine year?
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
P = Vw.drop(columns="view").groupby("nfl_id").mean()
Xp = P.fillna(0).values; yr = base.loc[P.index, "draft_year"].values
acc = cross_val_score(LogisticRegression(C=0.1, max_iter=3000), Xp, yr, cv=5).mean()
print(f"leak check: 5-fold accuracy predicting combine year from v2 features {acc:.2f} (chance {pd.Series(yr).value_counts(normalize=True).max():.2f})")
