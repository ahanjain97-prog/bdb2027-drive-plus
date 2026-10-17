"""Steps 4 + 6: two-dimensional coverage profile (zone closing / man mirroring) and a crossed random-effects model.

Profile metrics from the transition drill (same turn detection as the frozen metric):
  drive speed      frozen metric (top speed of the drive after the turn)
  turn ratio       speed at the end of the turn / speed at its start (carry speed THROUGH the turn; > 1 = accelerates
                   through it)
Validation per coverage type: leave-one-draft-class-out MSE gain for separation gained at the break and for completion
over expected allowed, models: none / drive / ratio / both; player-permutation p for the 'both' gain.
Hierarchical: sep_gain_break ~ fixed effects + crossed random intercepts for the CB and the receiver (statsmodels
MixedLM, variance components). CB variance with vs without drive speed -> share of true between-CB differences
explained by the drill.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import DRILL
rng = np.random.default_rng(19)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"); ct = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == DRILL)].copy()
ct["time"] = pd.to_datetime(ct.time); ct = ct.sort_values(["event_id", "time"])
rows = []
for eid, g in ct.groupby("event_id"):
    if len(g) < 12:
        continue
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; dt = np.median(np.diff(t))
    vx = savgol_filter(g.x.values, 7, 2, deriv=1, delta=dt); vy = savgol_filter(g.y.values, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
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
    rows.append({"nfl_id": g.nfl_id.iat[0], "t0": g.time.iat[0], "turn_ratio": s[j1] / max(s[j0], 0.5), "turn_rate": dmax / max(t[j1] - t[j0], 0.1)})
A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"]); A["att"] = A.groupby("nfl_id").cumcount()
for f in ["turn_ratio", "turn_rate"]:
    w = A[A.att <= 1].pivot(index="nfl_id", columns="att", values=f).dropna()
    print(f"test-retest {f}: {np.corrcoef(w[0], w[1])[0, 1]:.2f} (n {len(w)})")
PR = A.groupby("nfl_id")[["turn_ratio", "turn_rate"]].mean()
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
for f in ["turn_ratio", "turn_rate"]:
    v = PR[f].reindex(cov.nfl_id_d.unique()); cov["z_" + f] = cov.nfl_id_d.map((PR[f] - v.mean()) / v.std())
print("r(drive speed, turn ratio) over CBs:", round(pd.concat([cov.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[["z_drive", "z_turn_ratio"]]]).corr().iloc[0, 1], 2))
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])

print("\n=== step 4: profile -- which dimension predicts in which coverage? (per SD, clustered p) ===")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    for y, rhs in [("sep_gain_break", "wr_skill + C(route_ran)"), ("cpoe", "rec_target_prop")]:
        d = cov[cov.team_coverage_man_zone == cz].dropna(subset=[y, "z_drive", "z_turn_ratio"] + (["wr_skill"] if "wr_skill" in rhs else []))
        m = smf.ols(f"{y} ~ z_drive + z_turn_ratio + {rhs}", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
        # leave-one-class-out MSE gains vs the context-only model
        gains = {}
        for name, extra in [("drive", "z_drive"), ("ratio", "z_turn_ratio"), ("both", "z_drive + z_turn_ratio")]:
            gsum = 0.0
            for c in sorted(d.cls.dropna().unique()):
                tr, te = d[d.cls != c], d[d.cls == c]
                if "route_ran" in rhs:
                    te = te[te.route_ran.isin(tr.route_ran.unique())]
                b = smf.ols(f"{y} ~ {rhs}", data=tr).fit(); f_ = smf.ols(f"{y} ~ {extra} + {rhs}", data=tr).fit()
                gsum += (((te[y] - b.predict(te)) ** 2).sum() - ((te[y] - f_.predict(te)) ** 2).sum())
            gains[name] = gsum / ((d[y] - d[y].mean()) ** 2).sum() * 100
        print(f"  {cz:14s} {y:15s} drive {m.params['z_drive']:+.4f} (p {m.pvalues['z_drive']:.3f}) | turn ratio {m.params['z_turn_ratio']:+.4f} (p {m.pvalues['z_turn_ratio']:.3f}) | "
              f"held-out-class gain %: drive {gains['drive']:+.2f}, ratio {gains['ratio']:+.2f}, both {gains['both']:+.2f}  (reps {len(d)}, CBs {d.nfl_id_d.nunique()})")

print("\n=== step 6: crossed random effects (CB, receiver), separation gained at the break ===")
d = cov.dropna(subset=["sep_gain_break", "z_drive"]).copy()
d["grp"] = 1
vc = {"cb": "0 + C(nfl_id_d)", "wr": "0 + C(nfl_id_r)"}
res = {}
for lab, fe in [("context only", "C(team_coverage_man_zone) + C(route_ran)"), ("+ drive speed", "z_drive + C(team_coverage_man_zone) + C(route_ran)"),
                ("+ drive x coverage", "z_drive * C(team_coverage_man_zone) + C(route_ran)")]:
    m = smf.mixedlm(f"sep_gain_break ~ {fe}", data=d, groups="grp", vc_formula=vc, re_formula="0").fit(reml=True, method="lbfgs")
    vcomp = dict(zip(m.model.exog_vc.names, m.vcomp))
    res[lab] = (m, vcomp)
    fx = {k: f"{m.params[k]:+.4f} (p {m.pvalues[k]:.3f})" for k in m.params.index if "z_drive" in k}
    print(f"  {lab:20s} CB variance {vcomp['cb']:.4f}, receiver variance {vcomp['wr']:.4f}, residual {m.scale:.4f} | {fx}")
v0, v1 = res["context only"][1]["cb"], res["+ drive speed"][1]["cb"]
print(f"  share of true between-CB variance explained by drive speed: {1 - v1 / v0:.0%}  (CB SD {np.sqrt(v0):.3f} yd -> {np.sqrt(v1):.3f})")
cov.to_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet", index=False)
PR.to_csv(f"{ROOT}/results/64_turn_ratio_player.csv")
