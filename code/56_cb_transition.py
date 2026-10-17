"""Deep dive: CB backpedal-and-transition (45-degree reaction) drill -> separation at the receiver's break.

Drill phases per attempt (SG-smoothed). Tracking shows no speed dip at the transition: good players hold speed through a
~40-degree heading change ("speed turn"), then drive. Phases are therefore found by HEADING:
  backpedal line = median heading of the first frames at >= 2 yd/s; turn starts when heading deviates > 10 deg (at >= 3 yd/s)
  and ends at 90% of the maximum deviation within 1.5 s (25-90 deg, which excludes the return jog some clips include).
  bp_speed  speed when the turn starts      trans_min / trans_retention  slowest speed in the turn / as a share of bp_speed
  trans_time  turn duration    turn_deg  heading change    drive_accel_1s  speed gained per s in the 1 s after the turn
  drive_top  top speed of the drive (until speed stops rising, <= 2 s)    drill_top  max speed in the clip (catalog feature)
Rep-level outcomes in CB coverage reps (REG+POST): separation gained at the break, at the break, at the throw, mean;
completion when targeted. Model: outcome ~ z(feature) + receiver's separation skill in his other reps + route +
man/zone, two-way clustered; player-permutation p. Controls: tracked 40 top speed, weight, height.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
rng = np.random.default_rng(2)
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")].copy()
ct["time"] = pd.to_datetime(ct.time); ct = ct.sort_values(["event_id", "time"])
rows = []
for eid, g in ct.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 12:
        continue
    dt = np.median(np.diff(t))
    vx = savgol_filter(g.x.values, 7, 2, deriv=1, delta=dt); vy = savgol_filter(g.y.values, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    on = int(np.argmax(s >= 1.0))
    run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
    if len(run) < 4:
        continue
    bp_head = np.median(hd[run[:4]])                              # initial (backpedal) line
    dev = np.abs(hd - bp_head)
    st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
    if not len(st):
        continue
    j0 = st[0]
    win = slice(j0, min(j0 + 15, len(s)))                          # 1.5 s after the turn starts
    dmax = dev[win].max()
    if dmax < 25 or dmax > 90:                                     # a ~45-degree break, not the return jog
        continue
    j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax))
    jmin = j0 + int(np.argmin(s[j0:j1 + 1]))
    # drive: from the end of the turn until speed stops rising (or 2 s)
    k = j1; jend = min(j1 + 20, len(s) - 1); top = s[j1]; jtop = j1
    while k < jend:
        k += 1
        if s[k] > top:
            top, jtop = s[k], k
        elif s[k] < top - 0.5:
            break
    j_1s = min(np.searchsorted(t, t[j1] + 1.0), len(s) - 1)
    rows.append({"nfl_id": g.nfl_id.iat[0], "t0": g.time.iat[0], "bp_speed": s[j0], "trans_min": s[jmin],
                 "trans_retention": s[jmin] / max(s[j0], 0.5), "trans_time": t[j1] - t[j0], "turn_deg": dmax,
                 "drive_accel_1s": (s[j_1s] - s[j1]) / max(t[j_1s] - t[j1], 0.1), "drive_top": top, "drill_top": s.max()})
A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"])
A["att"] = A.groupby("nfl_id").cumcount()
FE = ["bp_speed", "trans_min", "trans_retention", "trans_time", "turn_deg", "drive_accel_1s", "drive_top", "drill_top"]
w = A[A.att <= 1].pivot_table(index="nfl_id", columns="att", values=FE)
print("attempts", len(A), "players", A.nfl_id.nunique(), "| test-retest:",
      {f: round(w[[(f, 0), (f, 1)]].dropna().corr().iloc[0, 1], 2) for f in FE if (f, 1) in w})
P = A.groupby("nfl_id")[FE].mean()
cbs = base.index[base.nfl_position == "CB"]
print("correlations among phase metrics (CBs):\n", P.reindex(cbs).corr().round(2).to_string())

cov = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
cov = cov[cov.season_type.isin(["REG", "POST"]) & (cov.nullified != "Y") & cov.nfl_id_d.isin(cbs)].copy()
tot = cov.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
cov["wr_skill"] = (cov.nfl_id_r.map(tot["sum"]) - cov.sep_gain_break.fillna(0)) / (cov.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)
cat = pd.read_parquet(f"{ROOT}/data/features/catalog_CB.parquet")
def z(s_):
    v = s_.reindex(pd.unique(cov.nfl_id_d)); return cov.nfl_id_d.map((s_ - v.mean()) / v.std())
for f in FE:
    cov["z_" + f] = z(P[f])
cov["z_forty_vmax"] = z(cat["drill:FORTY_YARD_DASH:top_speed"]); cov["z_w"] = z(base.combine_weight); cov["z_h"] = z(base.combine_height)
cov["cls"] = cov.nfl_id_d.map(base.draft_year)
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def fit(d, y, rhs, term, perm=True):
    d = d.dropna(subset=[y, term, "wr_skill"] + [c.strip() for c in rhs.split("+")])
    f = f"{y} ~ {rhs} + wr_skill + C(team_coverage_man_zone) + C(route_ran)"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    pp_ = np.nan
    if perm:
        vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; t0 = smf.ols(f, data=d).fit().tvalues[term]; null = []
        for _ in range(400):
            dd = d.copy(); dd[term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[term])
        pp_ = (1 + (np.abs(null) >= abs(t0)).sum()) / 401
    return m.params[term], m.pvalues[term], pp_, len(d), d.nfl_id_d.nunique()


print("\n=== which drill phase carries it? (outcome: separation the receiver gains at his break) ===")
for f in FE:
    b, p, pp_, n, k = fit(cov, "sep_gain_break", "z_" + f, "z_" + f)
    print(f"  {f:15s} b {b:+.4f} yd/SD  clustered p {p:.3f}  perm p {pp_:.3f}  (reps {n}, CBs {k})")
print("\n=== drive top speed: robustness and reach ===")
for y in ["sep_gain_break", "sep_at_break", "sep_at_throw", "sep_mean"]:
    b, p, pp_, n, k = fit(cov, y, "z_drive_top", "z_drive_top", perm=(y != "sep_gain_break"))
    print(f"  outcome {y:15s} b {b:+.4f}  p {p:.3f}  perm p {pp_:.3f}  reps {n}")
for lab, rhs in [("+ tracked 40 top speed", "z_drive_top + z_forty_vmax"), ("+ weight + height", "z_drive_top + z_w + z_h"),
                 ("+ 40 top speed + weight + height", "z_drive_top + z_forty_vmax + z_w + z_h")]:
    b, p, _, n, k = fit(cov, "sep_gain_break", rhs, "z_drive_top", perm=False)
    print(f"  sep_gain_break {lab:35s} b {b:+.4f} p {p:.3f} reps {n} CBs {k}")
for cz in ["MAN_COVERAGE", "ZONE_COVERAGE"]:
    b, p, _, n, k = fit(cov[cov.team_coverage_man_zone == cz], "sep_gain_break", "z_drive_top", "z_drive_top", perm=False)
    print(f"  {cz:14s} b {b:+.4f} p {p:.3f} reps {n} CBs {k}")
print("  by draft class:", {int(c): f"{fit(g, 'sep_gain_break', 'z_drive_top', 'z_drive_top', perm=False)[0]:+.3f}" for c, g in cov.groupby("cls")})
ids = cov.dropna(subset=["z_drive_top"]).nfl_id_d.unique()
loo = [fit(cov[cov.nfl_id_d != i], "sep_gain_break", "z_drive_top", "z_drive_top", perm=False)[0] for i in ids]
print(f"  leave-one-CB-out b range [{min(loo):+.4f}, {max(loo):+.4f}] over {len(ids)} CBs")
# completion when targeted
tg = cov[cov.targeted == 1].dropna(subset=["z_drive_top"])
m = smf.ols("completion ~ z_drive_top + C(team_coverage_man_zone)", data=tg).fit(cov_type="cluster", cov_kwds={"groups": G2(tg)})
print(f"  completion when targeted: b {m.params['z_drive_top']:+.3f} per SD (p {m.pvalues['z_drive_top']:.3f}), targets {len(tg)}")
# effect size in yards and as a tercile table
cov["drive_top_tercile"] = pd.qcut(cov.z_drive_top, 3, labels=["slow drive", "mid", "fast drive"])
print("\nseparation gained at the break (yd) by CB drive-top-speed tercile:")
print(cov.groupby(["drive_top_tercile", "team_coverage_man_zone"]).sep_gain_break.agg(["mean", "size"]).round(3).unstack().to_string())
P.to_csv(f"{ROOT}/results/56_cb_transition_player.csv")
