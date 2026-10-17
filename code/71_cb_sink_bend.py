"""Sink, bend, hip flip and drive for CBs: combine -> NFL movement -> coverage outcomes, zone-matching vs man.

Combine (DB drills, year-comparable: common window per drill from 66b, Savitzky-Golay kinematics, within-year z):
  SINK (plants: speed minimum after a >= 3 yd/s peak with a >= 40% drop)
    sink_rate     braking slope fitted over the frames between 85% and 35% of entry speed (yd/s^2, higher = sharper)
    sink_dist     yards travelled from 90% of entry speed to the minimum, per yd/s of entry speed (lower = sharper)
    plant_depth   minimum speed / entry speed (lower = had to stop more)
    exit_rate     re-acceleration slope 0.1-0.6 s after the plant
  BEND (speed turns: |lateral accel| > 1.5 at > 1.5 yd/s, >= 3 frames, same side)
    bend_speed    mean speed through turns       bend_lat  mean |lateral accel| in turns     bend_radius  median apex radius
  HIP FLIP (heading change >= 90 deg within 0.6 s while staying >= 1.0 yd/s)
    flip_speed    min speed through the flip     flip_time   time to 90% of the rotation    flip_rate  flips per attempt
  DRIVE           frozen transition-drill drive speed
NFL (CB coverage reps, around the receiver's break): the CB's own direction change = his peak smoothed |lateral accel|
  within 0.3 s before .. 1.0 s after the receiver's break (at > 1.5 yd/s):
    g_sink_rate   -slope of his speed over the 0.5 s before it (braking into it)   g_min_speed  his slowest speed around it
    g_bend_lat    |lateral accel| at it   g_bend_speed  speed at it               g_flip  his max heading change within +/-0.6 s
Tests: (1) reliability (odd/even events); (2) combine -> NFL same trait (player level, CBs >= 10 reps); (3) each combine
trait -> separation gained / at throw / closing in zone-matching and man-matching reps, alone and with drive speed;
player-permutation p; (4) the scheme split for each trait.
"""
import json, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
rng = np.random.default_rng(47)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
dbs = base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]
WIN = {'BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL': 4.8, 'BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION': 5.1, 'BOX_DRILL': 6.2,
       'GAUNTLET_DRILL': 7.3, 'LINE_DRILL': 5.5, 'TERYL_AUSTIN_DRILL_1': 5.6, 'TERYL_AUSTIN_DRILL_2': 6.9, 'W_DRILL': 6.7,
       'SHORT_SHUTTLE': 4.0, 'THREE_CONE_DRILL': 6.3}
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[(ct.entity_type == "PLAYER") & ct.nfl_id.isin(dbs)].copy(); ct["time"] = pd.to_datetime(ct.time)
ct["drill"] = ct.drill_name.replace({"LINE": "LINE_DRILL"})
ct = ct[ct.drill.isin(WIN)].sort_values(["event_id", "time"])


def slope(t, v):
    return np.polyfit(t, v, 1)[0] if len(t) >= 3 else np.nan


ev = []
for eid, g in ct.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 12:
        continue
    s0 = g.s.values
    if not (s0 >= 1).any():
        continue
    on = int(np.argmax(s0 >= 1.0)); keep = t <= t[on] + WIN[g.drill.iat[0]]
    t, x, y = t[keep], g.x.values[keep].astype(float), g.y.values[keep].astype(float)
    if len(t) < 12:
        continue
    dt = np.median(np.diff(t))
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    ax = savgol_filter(x, 7, 2, deriv=2, delta=dt); ay = savgol_filter(y, 7, 2, deriv=2, delta=dt)
    s = np.hypot(vx, vy); lat = np.where(s > 0.5, (vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    base_rec = {"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "yr": g.draft_year.iat[0]}
    # plants (sink)
    pk = 0
    for j in range(1, len(s) - 1):
        if s[j] > s[pk]:
            pk = j
        if s[j] <= s[j - 1] and s[j] < s[j + 1] and s[pk] >= 3.0 and s[j] <= 0.6 * s[pk]:
            e = s[pk]; seg = np.arange(pk, j + 1)
            sel = seg[(s[seg] <= .85 * e) & (s[seg] >= .35 * e)]
            b0 = pk + int(np.argmax(s[pk:j + 1] <= 0.9 * e))
            ex = np.where((t >= t[j] + .1) & (t <= t[j] + .6))[0]
            ev.append({**base_rec, "kind": "plant", "sink_rate": -slope(t[sel], s[sel]) if len(sel) >= 3 else np.nan,
                       "sink_dist": np.hypot(np.diff(x[b0:j + 1]), np.diff(y[b0:j + 1])).sum() / e, "plant_depth": s[j] / e,
                       "exit_rate": slope(t[ex], s[ex]) if len(ex) >= 3 else np.nan})
            pk = j
    # speed turns (bend)
    onm = (np.abs(lat) > 1.5) & (s > 1.5); sg = np.sign(lat); i = 0
    while i < len(s):
        if not onm[i]:
            i += 1; continue
        jj = i
        while jj + 1 < len(s) and onm[jj + 1] and sg[jj + 1] == sg[i]:
            jj += 1
        if jj - i + 1 >= 3:
            kk = i + int(np.argmax(np.abs(lat[i:jj + 1])))
            ev.append({**base_rec, "kind": "turn", "bend_speed": s[i:jj + 1].mean(), "bend_lat": np.abs(lat[i:jj + 1]).mean(),
                       "bend_radius": s[kk] ** 2 / max(abs(lat[kk]), 1e-6)})
        i = jj + 1
    # hip flips
    j = 0
    while j < len(s) - 6:
        w = slice(j, j + 7)
        if s[w].min() >= 1.0 and abs(hd[j + 6] - hd[j]) >= 90:
            tot = hd[j + 6] - hd[j]; k90 = j + int(np.argmax(np.abs(hd[j:j + 7] - hd[j]) >= 0.9 * abs(tot)))
            ev.append({**base_rec, "kind": "flip", "flip_speed": s[w].min(), "flip_time": t[k90] - t[j]})
            j += 7
        else:
            j += 1
    ev.append({**base_rec, "kind": "attempt"})
E = pd.DataFrame(ev)
E["k"] = E.groupby(["event_id", "kind"]).cumcount()
TR = ["sink_rate", "sink_dist", "plant_depth", "exit_rate", "bend_speed", "bend_lat", "bend_radius", "flip_speed", "flip_time"]
print("events:", E.kind.value_counts().to_dict(), "| flips per attempt by drill:", E[E.kind == "flip"].groupby("drill").size().div(E[E.kind == "attempt"].groupby("drill").size()).round(2).dropna().to_dict())
# within-year z per drill x trait, then per player mean across drills (= composite); reliability odd/even events
TR = [f for f in TR if f in E.columns]
for f in TR:
    grp = E.groupby(["drill", "yr"])[f]
    E[f + "_z"] = (E[f] - grp.transform("mean")) / grp.transform("std")
E["par"] = E.k % 2
P = E.groupby("nfl_id")[[f + "_z" for f in TR]].mean()
P.columns = TR
fl = E[E.kind == "flip"].groupby("nfl_id").size().div(E[E.kind == "attempt"].groupby("nfl_id").size()).rename("flip_rate")
P = P.join(fl)
d_ = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")); yr = base.draft_year.reindex(d_.index)
P["drive"] = (d_ - d_.groupby(yr).transform("mean")) / d_.groupby(yr).transform("std")
rel = {}
for f in TR:
    h = E.dropna(subset=[f + "_z"]).groupby(["nfl_id", "par"])[f + "_z"].mean().unstack().dropna()
    if len(h) >= 15:
        r = np.corrcoef(h[0], h[1])[0, 1]; rel[f] = round(2 * r / (1 + r), 2)
print("combine reliability (odd vs even events, Spearman-Brown):", rel)
cbs = base.index[base.nfl_position == "CB"]
print("correlations among combine traits (CBs):\n", P.reindex(cbs).corr().round(2).to_string())

# ---------------- NFL counterparts in CB coverage reps
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
gx = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = cov.idx_r.values, cov.idx_d.values
sr, lr, pr = gx(A, "s"), gx(A, "a_lat"), gx(A, "post_pass")
sd, ld, vxd, vyd = gx(B, "s"), gx(B, "a_lat"), gx(B, "vx"), gx(B, "vy")
vxr, vyr = gx(A, "vx"), gx(A, "vy")
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]; r_ = np.arange(len(cov))
sm = lambda a_: np.concatenate([a_[:, :1], (a_[:, :-2] + a_[:, 1:-1] + a_[:, 2:]) / 3, a_[:, -1:]], 1)
lrs, lds = sm(lr), sm(ld)
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(lrs), -1); jb = a.argmax(1)
win = vv & (cols >= (jb - 3)[:, None]) & (cols <= (jb + 10)[:, None]) & (sd > 1.5)
pdl = np.where(win, np.abs(lds), -1); jd = pdl.argmax(1); okd = pdl.max(1) > 1.0
cov["g_bend_lat"] = np.where(okd, np.abs(lds[r_, jd]), np.nan)
cov["g_bend_speed"] = np.where(okd, sd[r_, jd], np.nan)
pre = vv & (cols >= (jd - 5)[:, None]) & (cols <= jd[:, None])
sl = np.full(len(cov), np.nan)
for i in np.where(okd & (pre.sum(1) >= 4))[0]:
    idx = np.where(pre[i])[0]; sl[i] = -np.polyfit(idx / 10.0, sd[i, idx], 1)[0]
cov["g_sink_rate"] = sl
around = vv & (np.abs(cols - jb[:, None]) <= 6)
cov["g_min_speed"] = np.where(around.sum(1) >= 5, np.where(around, sd, np.inf).min(1), np.nan)
hdd = np.degrees(np.unwrap(np.arctan2(vxd, vyd), axis=1))
cov["g_flip"] = np.where(around.sum(1) >= 5, np.where(around, hdd, -np.inf).max(1) - np.where(around, hdd, np.inf).min(1), np.nan)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
w5 = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1) & (sr > 1)
pa = np.where(w5.sum(1) >= 3, np.where(w5, cosv, 0).sum(1) / np.maximum(w5.sum(1), 1), np.nan)
cov["cell"] = np.where(pa >= 0.7, np.where(cov.team_coverage_man_zone == "ZONE_COVERAGE", "zone-matching", "man-matching"), "other")
for f in list(P.columns):
    vals = P[f].reindex(cov.nfl_id_d.unique()); cov["z_" + f] = cov.nfl_id_d.map((P[f] - vals.mean()) / vals.std())
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 240)
E.to_parquet(f"{ROOT}/data/features/cb_sink_bend_events.parquet", index=False); P.to_parquet(f"{ROOT}/data/features/cb_sink_bend_player.parquet")
cov.drop(columns=[c for c in cov.columns if c.startswith("z_") and c != "z_drive"]).to_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet", index=False)
if "--save-only" in sys.argv:
    sys.exit(0)

print("\n=== (2) does each combine trait show up in the CB's own NFL movement? (player level, CBs >= 10 reps) ===")
GM = cov.groupby("nfl_id_d")[["g_sink_rate", "g_min_speed", "g_bend_lat", "g_bend_speed", "g_flip"]].mean()[cov.groupby("nfl_id_d").size() >= 10]
J = GM.join(P, how="inner")
pairs = [("sink_rate", "g_sink_rate"), ("sink_dist", "g_sink_rate"), ("plant_depth", "g_min_speed"), ("bend_speed", "g_bend_speed"),
         ("bend_lat", "g_bend_lat"), ("bend_radius", "g_bend_lat"), ("flip_speed", "g_flip"), ("flip_rate", "g_flip"), ("drive", "g_bend_speed")]
print("  " + " | ".join(f"{a_} -> {b_}: r {J[[a_, b_]].dropna().corr().iloc[0, 1]:+.2f}" for a_, b_ in pairs) + f"  (n {len(J)})")


def test(d, term, y, extra=""):
    d = d.dropna(subset=[y, term, "wr_skill"] + ([extra.strip(' +')] if extra else []))
    if d.nfl_id_d.nunique() < 12:
        return None
    f = f"{y} ~ {term}{extra} + wr_skill + C(route_ran)"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; null = []
    for _ in range(300):
        dd = d.copy(); dd[term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return m.params[term], (1 + (np.abs(null) >= abs(t0)).sum()) / 301, len(d), d.nfl_id_d.nunique()


print("\n=== (3) combine trait -> coverage outcome, by technique cell (b per SD; player-permutation p) ===")
rows = []
for f in ["sink_rate", "sink_dist", "plant_depth", "exit_rate", "bend_speed", "bend_lat", "bend_radius", "flip_speed", "flip_time", "flip_rate", "drive"]:
    for cell in ["zone-matching", "man-matching"]:
        for y in ["sep_gain_break", "sep_at_throw", "db_closing"]:
            for extra, lab in [("", "alone"), (" + z_drive", "+ drive")] if f != "drive" else [("", "alone")]:
                r = test(cov[cov.cell == cell], "z_" + f, y, extra)
                if r:
                    rows.append({"trait": f, "cell": cell, "outcome": y, "model": lab, "b": r[0], "p_perm": r[1], "reps": r[2], "CBs": r[3]})
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/71_cb_sink_bend.csv", index=False)
show = R[R.model == "alone"].pivot_table(index="trait", columns=["cell", "outcome"], values="b").round(3)
showp = R[R.model == "alone"].pivot_table(index="trait", columns=["cell", "outcome"], values="p_perm").round(3)
print("b per SD:\n" + show.to_string()); print("\nperm p:\n" + showp.to_string())
print("\nwith drive speed in the model (trait b, perm p) -- does the trait add beyond drive?")
print(R[(R.model == "+ drive") & (R.p_perm < .10)].round(3).to_string(index=False) or "  none at p < .10")
print("\nhits at p < .05 (alone):", len(R[(R.model == "alone") & (R.p_perm < .05)]), "of", len(R[R.model == "alone"]), f"(expected by chance {0.05 * len(R[R.model == 'alone']):.1f})")
