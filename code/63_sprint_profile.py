"""Step 3: sprint physics of the drive phase vs the same player's 40.

Exponential sprint model (Furusawa / Clark):  v(t) = vmax - (vmax - v0) * exp(-t / tau)
  drill drive : fit from the end of the turn (frozen-metric definition of the turn) over the rising phase (<= 2 s)
  40-yard dash: fit from onset (v0 = 0 at the onset frame) to the clip end
Derived per attempt:
  drive_vmax, drive_tau            asymptotic drive speed and time constant (extrapolates past the drill's short run-out)
  forty_vmax, forty_tau
  transition_tax = a_drive(v1) / a_40(v1), acceleration available at the speed he comes out of the turn (v1), after the
                   turn vs in a straight line (1 = no penalty, < 1 = he loses acceleration because he turned)
  drive_share   = drive_vmax / forty_vmax
Compared with the frozen drive speed: test-retest, correlation, and rep-level prediction of separation gained at the
break and completion over expected allowed (same models as before), alone and together.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import DRILL
rng = np.random.default_rng(17)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"); ct = ct[ct.entity_type == "PLAYER"].copy(); ct["time"] = pd.to_datetime(ct.time)
dbs = base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]
ct = ct[ct.nfl_id.isin(dbs) & ct.drill_name.isin([DRILL, "FORTY_YARD_DASH"])].sort_values(["event_id", "time"])
model = lambda t, vmax, tau, v0: vmax - (vmax - v0) * np.exp(-t / tau)


def kin(g):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; dt = np.median(np.diff(t))
    vx = savgol_filter(g.x.values, 7, 2, deriv=1, delta=dt); vy = savgol_filter(g.y.values, 7, 2, deriv=1, delta=dt)
    return t, np.hypot(vx, vy), np.degrees(np.unwrap(np.arctan2(vx, vy)))


rows = []
for eid, g in ct.groupby("event_id"):
    if len(g) < 12:
        continue
    t, s, hd = kin(g)
    rec = {"nfl_id": g.nfl_id.iat[0], "t0": g.time.iat[0], "drill": g.drill_name.iat[0]}
    if rec["drill"] == "FORTY_YARD_DASH":
        on = int(np.argmax(s >= 1.0)); tt, vv = t[on:] - t[on], s[on:]
        try:
            p, _ = curve_fit(lambda x, vmax, tau: model(x, vmax, tau, vv[0]), tt, vv, p0=[10, 1.0], bounds=([5, 0.2], [13, 4]))
            rec.update(forty_vmax=p[0], forty_tau=p[1])
        except Exception:
            continue
    else:
        on = int(np.argmax(s >= 1.0)); run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
        if len(run) < 4:
            continue
        dev = np.abs(hd - np.median(hd[run[:4]]))
        st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
        if not len(st):
            continue
        j0 = st[0]; win = slice(j0, min(j0 + 15, len(s))); dmax = dev[win].max()
        if dmax < 25 or dmax > 90:
            continue
        j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax))
        k, top = j1, s[j1]
        while k + 1 < min(j1 + 20, len(s)):
            k += 1
            if s[k] > top: top = s[k]
            elif s[k] < top - 0.5: break
        seg = slice(j1, k + 1)
        tt, vv = t[seg] - t[j1], s[seg]
        if len(tt) < 5:
            continue
        try:
            p, _ = curve_fit(lambda x, vmax, tau: model(x, vmax, tau, vv[0]), tt, vv, p0=[top + 0.5, 0.8], bounds=([vv[0], 0.1], [14, 5]))
            rec.update(drive_vmax=p[0], drive_tau=p[1], v1=vv[0], drive_top=top)
        except Exception:
            continue
    rows.append(rec)
A = pd.DataFrame(rows).sort_values(["nfl_id", "drill", "t0"]); A["att"] = A.groupby(["nfl_id", "drill"]).cumcount()
print(f"fits: drill {A.drive_vmax.notna().sum()}, forty {A.forty_vmax.notna().sum()}")
tr = {}
for f, dr in [("drive_vmax", DRILL), ("drive_tau", DRILL), ("drive_top", DRILL), ("forty_vmax", "FORTY_YARD_DASH"), ("forty_tau", "FORTY_YARD_DASH")]:
    w = A[(A.drill == dr) & (A.att <= 1)].pivot(index="nfl_id", columns="att", values=f).dropna()
    tr[f] = (np.corrcoef(w[0], w[1])[0, 1], len(w))
print("test-retest:", {k: f"{v[0]:.2f} (n {v[1]})" for k, v in tr.items()})
P = A[A.drill == DRILL].groupby("nfl_id")[["drive_vmax", "drive_tau", "v1", "drive_top"]].mean().join(
    A[A.drill == "FORTY_YARD_DASH"].groupby("nfl_id")[["forty_vmax", "forty_tau"]].mean(), how="left")
a_drive = (P.drive_vmax - P.v1) / P.drive_tau
a_40 = (P.forty_vmax - P.v1).clip(lower=0.1) / P.forty_tau
P["transition_tax"] = a_drive / a_40
P["drive_share"] = P.drive_vmax / P.forty_vmax
print("\ncorrelations among CB sprint metrics:")
print(P[P.index.isin(base.index[base.nfl_position == "CB"])][["drive_top", "drive_vmax", "drive_tau", "forty_vmax", "forty_tau", "transition_tax", "drive_share"]].corr().round(2).to_string())
P.to_csv(f"{ROOT}/results/63_sprint_profile_player.csv")

cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
for f in ["drive_top", "drive_vmax", "drive_tau", "transition_tax", "drive_share", "forty_vmax", "forty_tau"]:
    v = P[f].reindex(cov.nfl_id_d.unique()); cov["z_" + f] = cov.nfl_id_d.map((P[f] - v.mean()) / v.std())
print("\nrep-level prediction (per SD; player-permutation p):")
for y, rhs in [("sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"), ("cpoe", "C(team_coverage_man_zone)")]:
    for f in ["drive_top", "drive_vmax", "drive_tau", "transition_tax", "drive_share", "forty_vmax", "forty_tau"]:
        d = cov.dropna(subset=[y, "z_" + f] + (["wr_skill"] if "wr_skill" in rhs else []))
        fml = f"{y} ~ z_{f} + {rhs}"
        m = smf.ols(fml, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
        t0 = smf.ols(fml, data=d).fit().tvalues["z_" + f]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")["z_" + f]; null = []
        for _ in range(300):
            dd = d.copy(); dd["z_" + f] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(fml, data=dd).fit().tvalues["z_" + f])
        print(f"  {y:15s} {f:15s} {m.params['z_' + f]:+.4f} (clustered p {m.pvalues['z_' + f]:.3f}, perm p {(1 + (np.abs(null) >= abs(t0)).sum()) / 301:.3f}) n {len(d)} CBs {d.nfl_id_d.nunique()}")
d = cov.dropna(subset=["sep_gain_break", "wr_skill", "z_drive_top", "z_transition_tax"])
m = smf.ols("sep_gain_break ~ z_drive_top + z_transition_tax + wr_skill + C(team_coverage_man_zone) + C(route_ran)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
print(f"\njoint: drive_top {m.params['z_drive_top']:+.4f} (p {m.pvalues['z_drive_top']:.3f}), transition_tax {m.params['z_transition_tax']:+.4f} (p {m.pvalues['z_transition_tax']:.3f}), CBs {d.nfl_id_d.nunique()}")
