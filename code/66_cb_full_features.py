"""Full, unshrunk CB combine feature table (no reliability filter, no selection, no PCA).

Per CB drill attempt (11 drills: 8 DB drills + 40 + shuttle + 3-cone):
  mechanics  all 41 per-attempt features of the feature factory (Savitzky-Golay kinematics)
  curves     the attempt resampled onto 20 time-normalized points (onset -> end) for speed, turn rate |d heading/dt|,
             along-track accel and |lateral accel|  (80 values per drill)
  transition the 45-degree transition drill phase metrics (backpedal speed, slowest speed in the turn, turn time,
             turn angle, drive acceleration in 1 s, drive top speed = frozen metric, turn ratio)
Attempts are kept as VIEWS: view v takes each drill's attempt v (or the player's last attempt if he has fewer), so a
player with two attempts contributes two rows. Output data/features/cb_full_views.parquet (+ player means).
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import DRILL
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cbs = base.index[base.nfl_position == "CB"]
A = pd.read_parquet(f"{ROOT}/data/features/attempt_features.parquet")
A = A[A.nfl_id.isin(cbs)].copy()
MECH = [c for c in A.columns if c not in ("event_id", "nfl_id", "drill", "group", "t0")]
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[(ct.entity_type == "PLAYER") & ct.event_id.isin(A.event_id)].copy(); ct["time"] = pd.to_datetime(ct.time)
ct = ct.sort_values(["event_id", "time"])
NPT = 20
curves, trans = {}, {}
for eid, g in ct.groupby("event_id"):
    if len(g) < 9:
        continue
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; dt = np.median(np.diff(t))
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    ax = savgol_filter(x, 7, 2, deriv=2, delta=dt); ay = savgol_filter(y, 7, 2, deriv=2, delta=dt)
    s = np.hypot(vx, vy)
    if s.max() < 1:
        continue
    al = (vx * ax + vy * ay) / np.maximum(s, 1e-6)
    lat = np.where(s > 0.5, np.abs(vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    hd = np.unwrap(np.arctan2(vx, vy)); tr = np.where(s > 0.5, np.abs(np.gradient(hd, t)) * 180 / np.pi, 0.0)
    on = int(np.argmax(s >= 1.0)); tt = (t[on:] - t[on]) / max(t[-1] - t[on], 1e-6)
    grid = np.linspace(0, 1, NPT)
    curves[eid] = np.concatenate([np.interp(grid, tt, sig[on:]) for sig in (s, tr, al, lat)])
    if g.drill_name.iat[0] == DRILL:
        run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
        if len(run) >= 4:
            hdd = np.degrees(hd); dev = np.abs(hdd - np.median(hdd[run[:4]]))
            st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
            if len(st):
                j0 = st[0]; win = slice(j0, min(j0 + 15, len(s))); dmax = dev[win].max()
                if 25 <= dmax <= 90:
                    j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax)); jmin = j0 + int(np.argmin(s[j0:j1 + 1]))
                    k, top = j1, s[j1]
                    while k + 1 < min(j1 + 20, len(s)):
                        k += 1
                        if s[k] > top: top = s[k]
                        elif s[k] < top - 0.5: break
                    j_1s = min(np.searchsorted(t, t[j1] + 1.0), len(s) - 1)
                    trans[eid] = {"bp_speed": s[j0], "trans_min": s[jmin], "trans_time": t[j1] - t[j0], "turn_deg": dmax,
                                  "drive_accel_1s": (s[j_1s] - s[j1]) / max(t[j_1s] - t[j1], .1), "drive_top": top, "turn_ratio": s[j1] / max(s[j0], .5)}
SIG = ["speed", "turnrate", "along", "lateral"]
CN = [f"curve_{sg}_{i:02d}" for sg in SIG for i in range(NPT)]
C = pd.DataFrame.from_dict(curves, orient="index", columns=CN)
TRN = pd.DataFrame.from_dict(trans, orient="index").add_prefix("trans_")
A = A.set_index("event_id").join(C).join(TRN).reset_index()
A = A.sort_values(["nfl_id", "drill", "t0"]); A["att"] = A.groupby(["nfl_id", "drill"]).cumcount()
FEAT = MECH + CN + list(TRN.columns)
nviews = A.groupby("nfl_id").att.max() + 1
rows = []
for pid, g in A.groupby("nfl_id"):
    for v in range(int(nviews[pid])):
        rec = {"nfl_id": pid, "view": v}
        for d_, gd in g.groupby("drill"):
            a = gd[gd.att == min(v, gd.att.max())].iloc[0]
            tag = d_.replace("DB:", "")
            for f in FEAT:
                val = a[f]
                if pd.notna(val):
                    rec[f"{tag}|{f}"] = float(val)
        rows.append(rec)
Vw = pd.DataFrame(rows)
Vw = Vw.loc[:, Vw.notna().sum() >= 10]
Vw.to_parquet(f"{ROOT}/data/features/cb_full_views.parquet", index=False)
Pm = Vw.drop(columns="view").groupby("nfl_id").mean()
Pm.to_parquet(f"{ROOT}/data/features/cb_full_player.parquet")
print(f"CBs {Vw.nfl_id.nunique()}, view rows {len(Vw)} (views per CB {nviews.value_counts().to_dict()}), features {Vw.shape[1] - 2}")
fam = pd.Series([c.split("|")[1].split("_")[0] if "|" in c else c for c in Vw.columns[2:]]).value_counts()
print("feature families:", {"curves": int(sum(c.split('|')[1].startswith('curve_') for c in Vw.columns[2:])),
                            "transition phases": int(sum(c.split('|')[1].startswith('trans_') for c in Vw.columns[2:])),
                            "mechanics": int(sum(not c.split('|')[1].startswith(('curve_', 'trans_')) for c in Vw.columns[2:]))})
print("coverage (share of CBs with each drill):", (Pm.notna().groupby(lambda c: c.split("|")[0], axis=1).mean().mean().round(2)).to_dict())
