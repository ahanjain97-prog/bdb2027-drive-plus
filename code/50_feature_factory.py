"""Phase 2: combine feature factory -- position- and drill-specific mechanics per attempt, then stability gating.

v2 (after 51_stability_experiment): kinematics from a Savitzky-Golay fit (window 7, order 2) of x/y -- velocity, speed,
along-track accel (a_long = v.a/|v|) and lateral accel (a_lat = (vx*ay - vy*ax)/|v|) analytically; onset = first frame
>= 1 yd/s. (v0 used vendor speed + frame differences; reliability of re-acceleration .18 -> .45 with this estimator.)
Per attempt:
  global   top_speed, t_to_90 (onset -> 90% of top), speed_at_0.5s / 1.0s / 1.5s after onset, accel_0_1s (sustained),
           mean_speed, duration, path_len, straightness, jerk_rms, sustained peak accel / brake (max of 0.3-s means)
  turns    (|a_lat| > 1.5 at > 1.5 yd/s, same sign, >= 0.3 s): n_turns, turn_speed (duration-weighted), turn_lean
           (mean |a_lat| in turns), turn_lean_max, turn_radius (median apex speed^2 / |a_lat|), turn_retention
  breaks   (a speed minimum after a peak >= 3 yd/s where speed falls by >= 40%): n_breaks; first-break and mean-over-
           breaks entry_speed, min_speed, brake_time, brake_decel (sustained), brake_dist, exit_burst (+0.5 s),
           reaccel_time (to 80% of entry), cut_angle, cut_retention (min / entry)
  catch    (WR/TE drills with the ball tracked: closest player-ball approach within 1.5 yd after the ball is moving):
           speed_at_catch, decel_into_catch (speed 0.5 s before minus at), catch_carry (speed 0.5 s after / before)
Stability:
  test-retest  attempt 1 vs 2 (time order) per drill x feature, where >= 15 players repeated the drill
  traits       per position group, a feature's mean z across all of that group's drills where it exists ("trait
               composite"); reliability = split-half over drills (odd / even drills, Spearman-Brown)
Output data/features/{attempt_features, player_drill_features, trait_composites}.parquet, results/50_*.csv
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
os.makedirs(f"{ROOT}/data/features", exist_ok=True)
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct["time"] = pd.to_datetime(ct.time)
ct = ct.sort_values(["event_id", "entity_type", "time"])
GRP = {"SKILL_DRILLS_WR": "WR", "SKILL_DRILLS_TE": "TE", "SKILL_DRILLS_DB": "DB", "SKILL_DRILLS_DL": "DL", "SKILL_DRILLS_OL": "OL",
       "SKILL_DRILLS_LB": "LB", "FORTY_YARD_DASH": "ALL", "SHORT_SHUTTLE": "ALL", "THREE_CONE_DRILL": "ALL"}
ct["drill"] = (ct.drill_type.str.replace("SKILL_DRILLS_", "") + ":" + ct.drill_name)
ct.loc[ct.drill_type.isin(["FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL"]), "drill"] = ct.drill_name
# year renames of the same drill
ct["drill"] = ct.drill.replace({"OL:PASS_PRO-MIRROR_DRILL": "OL:PASS_PRO_MIRROR_DRILL", "WR:SPEED_OUT_LEFT": "WR:SPEED_OUT_ROUTE_LEFT",
                                "TE:OVER_THE_SHOULDER_ADJUST": "TE:OVER_SHOULDER_ADJUST", "DB:LINE": "DB:LINE_DRILL"})


def roll(a, w=3):
    return pd.Series(a).rolling(w, center=True, min_periods=1).mean().values


def features(g, ball):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 9 or t[-1] <= 0.5:
        return None
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    dtm = np.median(np.diff(t))
    vx = savgol_filter(x, 7, 2, deriv=1, delta=dtm); vy = savgol_filter(y, 7, 2, deriv=1, delta=dtm)
    ax = savgol_filter(x, 7, 2, deriv=2, delta=dtm); ay = savgol_filter(y, 7, 2, deriv=2, delta=dtm)
    s = np.hypot(vx, vy)
    if s.max() < 1.0:
        return None
    th = np.arctan2(vx, vy)                                  # compass heading (0 = +y), for cut angles
    al = (vx * ax + vy * ay) / np.maximum(s, 1e-6)
    lat = np.where(s > 0.5, (vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    on = int(np.argmax(s >= 1.0)); t0 = t[on]
    f = {}
    vmax = s[on:].max(); f["top_speed"] = vmax
    f["t_to_90"] = t[on + int(np.argmax(s[on:] >= 0.9 * vmax))] - t0
    for dt_ in (0.5, 1.0, 1.5):
        j = np.searchsorted(t, t0 + dt_)
        f[f"speed_at_{dt_}s"] = s[j] if j < len(s) else np.nan
    f["accel_0_1s"] = (f["speed_at_1.0s"] - s[on]) / 1.0 if np.isfinite(f["speed_at_1.0s"]) else np.nan
    f["mean_speed"] = s[on:].mean(); f["duration"] = t[-1] - t0
    steps = np.hypot(np.diff(x[on:]), np.diff(y[on:]))
    f["path_len"] = steps.sum()
    f["straightness"] = np.hypot(x[-1] - x[on], y[-1] - y[on]) / max(f["path_len"], 1e-6)
    jerk = np.gradient(al, t)[on:]
    f["jerk_rms"] = np.sqrt(np.mean(jerk ** 2)) / max(vmax, 1e-6)
    a3 = roll(al, 3)
    f["accel_sustained_peak"] = a3[on:].max(); f["brake_sustained_peak"] = -a3[on:].min()
    # turns
    onm = (np.abs(lat) > 1.5) & (s > 1.5); sign = np.sign(lat)
    tsp, tw, tlean, tmax, trad, tret = [], [], [], [], [], []
    i = 0
    while i < len(s):
        if not onm[i]:
            i += 1; continue
        j = i
        while j + 1 < len(s) and onm[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        if j - i + 1 >= 3:
            seg = slice(i, j + 1); kk = i + int(np.argmax(np.abs(lat[seg])))
            tsp.append(s[seg].mean()); tw.append(t[j] - t[i]); tlean.append(np.abs(lat[seg]).mean()); tmax.append(np.abs(lat[kk]))
            EVENTS.append({"kind": "turn", "k": len(tsp) - 1, "turn_speed": s[seg].mean(), "turn_lean": np.abs(lat[seg]).mean()})
            trad.append(s[kk] ** 2 / max(np.abs(lat[kk]), 1e-6)); tret.append(s[kk] / max(s[max(0, i - 3)], 0.5))
        i = j + 1
    f["n_turns"] = len(tsp)
    if tsp:
        f["turn_speed"] = np.average(tsp, weights=np.maximum(tw, 1e-3)); f["turn_lean"] = np.average(tlean, weights=np.maximum(tw, 1e-3))
        f["turn_lean_max"] = max(tmax); f["turn_radius"] = np.median(trad); f["turn_retention"] = np.mean(tret)
    # breaks: speed minima after a peak >= 3 yd/s with a >= 40% drop
    brk = []
    pk_i = on
    for j in range(on + 1, len(s) - 1):
        if s[j] > s[pk_i]:
            pk_i = j
        if s[j] <= s[j - 1] and s[j] < s[j + 1] and s[pk_i] >= 3.0 and s[j] <= 0.6 * s[pk_i]:
            entry = s[pk_i]
            b0 = pk_i + int(np.argmax(s[pk_i:j + 1] <= 0.9 * entry)) if (s[pk_i:j + 1] <= 0.9 * entry).any() else pk_i
            btime = max(t[j] - t[b0], 0.1)
            je = np.searchsorted(t, t[j] + 0.5); je = min(je, len(s) - 1)
            ra = np.where(s[j:] >= 0.8 * entry)[0]
            hb = np.unwrap(th)
            jb, ja = np.searchsorted(t, t[j] - 0.3), min(np.searchsorted(t, t[j] + 0.3), len(s) - 1)
            brk.append({"entry_speed": entry, "min_speed": s[j], "brake_time": btime, "brake_decel": (entry - s[j]) / btime,
                        "brake_dist": np.hypot(np.diff(x[b0:j + 1]), np.diff(y[b0:j + 1])).sum(), "exit_burst": s[je] - s[j],
                        "reaccel_time": t[j + ra[0]] - t[j] if len(ra) else np.nan,
                        "cut_angle": np.degrees(abs(np.arctan2(np.sin(hb[ja] - hb[jb]), np.cos(hb[ja] - hb[jb])))),
                        "cut_retention": s[j] / entry})
            EVENTS.append({"kind": "break", "k": len(brk) - 1, **brk[-1]})
            pk_i = j
    f["n_breaks"] = len(brk)
    if brk:
        for kname in brk[0]:
            f[f"brk1_{kname}"] = brk[0][kname]
            f[f"brk_mean_{kname}"] = np.nanmean([b[kname] for b in brk])
    # catch point
    if ball is not None and len(ball) >= 5:
        tb = (ball.time - g.time.iat[0]).dt.total_seconds().values
        bxi, byi = np.interp(t, tb, ball.x.values, left=np.nan, right=np.nan), np.interp(t, tb, ball.y.values, left=np.nan, right=np.nan)
        bd = np.hypot(x - bxi, y - byi)
        bs = np.r_[0, np.hypot(np.diff(bxi), np.diff(byi)) / np.maximum(np.diff(t), 1e-3)]
        cand = np.where((bs > 5) & np.isfinite(bd))[0]
        if len(cand):
            jc = cand[np.argmin(bd[cand])]
            if bd[jc] <= 1.5:
                jm, jp = np.searchsorted(t, t[jc] - 0.5), min(np.searchsorted(t, t[jc] + 0.5), len(s) - 1)
                f["speed_at_catch"] = s[jc]; f["decel_into_catch"] = s[jm] - s[jc]; f["catch_carry"] = s[jp] / max(s[jm], 0.5)
    return f


rows, EV = [], []
EVENTS = []
players = ct[ct.entity_type == "PLAYER"]
balls = ct[ct.entity_type == "BALL"]
ball_by_event = {e: b for e, b in balls.groupby("event_id")}
for eid, g in players.groupby("event_id"):
    EVENTS.clear()
    f = features(g, ball_by_event.get(eid))
    for e in EVENTS:
        EV.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], **e})
    if f:
        rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "group": GRP.get(g.drill_type.iat[0], "?"),
                     "t0": g.time.iat[0], **f})
A = pd.DataFrame(rows)
A.to_parquet(f"{ROOT}/data/features/attempt_features.parquet", index=False)
E = pd.DataFrame(EV); E.to_parquet(f"{ROOT}/data/features/attempt_events.parquet", index=False)
# within-attempt reliability: odd vs even events of the same kind, pooled over a player's attempts in a drill
wi = []
for (d_, kind), g in E.groupby(["drill", "kind"]):
    g = g.assign(par=g.k % 2)
    for f in [c for c in g.columns if c not in ("event_id", "nfl_id", "drill", "kind", "k", "par") and g[c].notna().any()]:
        w = g.groupby(["nfl_id", "par"])[f].mean().unstack().dropna()
        if len(w) >= 15 and w.shape[1] == 2 and w.std().min() > 0:
            r = np.corrcoef(w[0], w[1])[0, 1]
            wi.append({"drill": d_, "kind": kind, "feature": f, "n": len(w), "within_attempt_rel": 2 * r / (1 + r)})
WI = pd.DataFrame(wi); WI.to_csv(f"{ROOT}/results/50_within_attempt_reliability.csv", index=False)
print("within-attempt (odd/even events) reliability, best drill x feature:")
print(WI.sort_values("within_attempt_rel", ascending=False).head(15).round(2).to_string(index=False))
FEATS = [c for c in A.columns if c not in ("event_id", "nfl_id", "drill", "group", "t0")]
print(f"attempts {len(A)}, players {A.nfl_id.nunique()}, drills {A.drill.nunique()}, features per attempt {len(FEATS)}", flush=True)

# ---- test-retest per drill x feature (attempt 1 vs 2 in time order)
A = A.sort_values(["nfl_id", "drill", "t0"])
A["att"] = A.groupby(["nfl_id", "drill"]).cumcount()
tr = []
for d_, g in A.groupby("drill"):
    w = g[g.att <= 1].pivot_table(index="nfl_id", columns="att", values=FEATS)
    for f in FEATS:
        if (f, 0) in w and (f, 1) in w:
            m = w[[(f, 0), (f, 1)]].dropna()
            if len(m) >= 15 and m.std().min() > 0:
                tr.append({"drill": d_, "feature": f, "n_repeat": len(m), "test_retest": np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1]})
TR = pd.DataFrame(tr)
TR.to_csv(f"{ROOT}/results/50_test_retest.csv", index=False)
print("\ntest-retest available for", TR.drill.nunique(), "drills; median by feature (top/bottom):")
med = TR.groupby("feature").test_retest.median().sort_values()
print(pd.concat([med.head(8), med.tail(8)]).round(2).to_string())

# ---- player x drill means (wide), and trait composites per position group
PD = A.groupby(["nfl_id", "drill"])[FEATS].mean()
PD.to_parquet(f"{ROOT}/data/features/player_drill_features.parquet")
grp_of = A.drop_duplicates("drill").set_index("drill").group
FAMILIES = {
    "WR": {"routes": ["SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT", "GO_ROUTE_RIGHT",
                       "RED_ZONE_FADE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT", "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "CIRCUS_ROUTE_LEFT"],
           "breaking_routes": ["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT", "SLOT_STRIKE_ROUTE_LEFT", "DAGGER_ROUTE_LEFT"],
           "ball": ["OVER_SHOULDER_ADJUST", "CIRCUS_ROUTE_LEFT", "GAUNTLET_DRILL", "RED_ZONE_FADE_RIGHT"]},
    "TE": {"routes": ["FLAT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT", "IN_ROUTE_LEFT", "WHEEL_ROUTE_LEFT", "CORNER_ROUTE", "END_ZONE_FADE_RIGHT"],
           "all_te": None},
    "DB": {"backpedal": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2", "LINE_DRILL"],
           "cod": ["W_DRILL", "BOX_DRILL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2"], "all_db": None},
    "DL": {"rush": ["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL", "RUN_AND_CLUB_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION"],
           "agility": ["FOUR_BAG_AGILITY_DRILL", "BODY_CONTROL_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION"], "all_dl": None},
    "OL": {"lateral": ["FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", "PASS_PRO_MIRROR_DRILL", "PASS_RUSH_DROP"],
           "pulls": ["OL_PULL_DRILL_FOLD_BLOCK_RIGHT", "OL_PULL_DRILL_LONG_PULL_RIGHT", "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT", "SCREEN_DRILL"], "all_ol": None},
}
AGIL = ["SHORT_SHUTTLE", "THREE_CONE_DRILL"]
comp_rows, rel_rows = [], []
for G, fams in FAMILIES.items():
  fams = dict(fams); fams["agility_stopwatch_drills"] = AGIL
  for fam, names in fams.items():
    drills = [d_ for d_ in grp_of.index if grp_of[d_] == G] if names is None else \
             ([f"{G}:{n}" for n in names] if fam != "agility_stopwatch_drills" else names)
    drills = [d_ for d_ in drills if d_ in grp_of.index]
    sub = PD[PD.index.get_level_values("drill").isin(drills)]
    for f in FEATS:
        w = sub[f].unstack("drill")
        w = w.loc[:, w.notna().sum() >= 20]
        w = w.loc[:, w.std() > 0]
        if w.shape[1] < 2:
            continue
        z = (w - w.mean()) / w.std()
        comp = z.mean(axis=1, skipna=True)
        cols = list(z.columns)
        odd, even = z[cols[0::2]].mean(axis=1), z[cols[1::2]].mean(axis=1)
        both = pd.concat([odd, even], axis=1).dropna()
        r = np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1] if len(both) >= 15 else np.nan
        rel_rows.append({"group": G, "family": fam, "feature": f, "n_drills": w.shape[1], "n_players": int(comp.notna().sum()),
                         "split_half_over_drills": 2 * r / (1 + r) if np.isfinite(r) else np.nan})
        comp_rows.append(comp.rename(f"{G}|{fam}|{f}"))
TC = pd.concat(comp_rows, axis=1)
TC.to_parquet(f"{ROOT}/data/features/trait_composites.parquet")
REL = pd.DataFrame(rel_rows)
REL.to_csv(f"{ROOT}/results/50_trait_reliability.csv", index=False)
pd.set_option("display.width", 220)
print("\ntrait composites: count with split-half-over-drills >= .5, by group:")
print(REL.assign(ok5=REL.split_half_over_drills >= .5, ok4=REL.split_half_over_drills >= .4).groupby(["group", "family"]).agg(
    traits=("feature", "size"), rel_ge_40=("ok4", "sum"), rel_ge_50=("ok5", "sum")).to_string())
for G in ["WR", "DB", "DL", "OL", "TE"]:
    t = REL[REL.group == G].sort_values("split_half_over_drills", ascending=False)
    print(f"\n{G} most reliable family traits:", ", ".join(f"{fa}|{a} {b:.2f}" for fa, a, b in zip(t.family.head(14), t.feature.head(14), t.split_half_over_drills.head(14))))
