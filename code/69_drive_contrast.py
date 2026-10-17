"""The shape the full model found, as one number: how much faster a CB comes OUT of the transition than he went INTO it.

From each transition-drill attempt's time-normalized speed curve (20 points, onset -> end; 66_cb_full_features):
  early   = mean speed over normalized time 0.05-0.55 (backpedal + turn)
  late    = mean speed over 0.60-1.00 (the drive)
  drive_contrast = late - early        drive_ratio = late / early
Caveat: the windows were suggested by the all-CB ridge fit (68), so the leave-one-draft-class-out tests below are not
fully independent of that choice; the fold-internal 'curves-only ridge' (68: CPOE +2.2%, p .03) is the clean version.
Tests (rep level): CPOE, separation at the throw, separation gained at the break; per SD; two-way clustered + player
permutation p; leave-one-draft-class-out MSE gain; and jointly with the frozen drive speed.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(37)
Vw = pd.read_parquet(f"{ROOT}/data/features/cb_full_views.parquet")
T = "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION|curve_speed_"
pts = np.linspace(0, 1, 20)
S = Vw[[f"{T}{i:02d}" for i in range(20)]].values
early = np.nanmean(S[:, (pts >= .05) & (pts <= .55)], 1); late = np.nanmean(S[:, pts >= .60], 1)
Vw["drive_contrast"] = late - early; Vw["drive_ratio"] = late / early
M = Vw.groupby("nfl_id")[["drive_contrast", "drive_ratio"]].mean()
w2 = Vw.pivot(index="nfl_id", columns="view", values="drive_contrast").dropna()
print(f"drive_contrast test-retest (view 0 vs 1, where the two attempts differ): {np.corrcoef(w2[0], w2[1])[0, 1]:.2f}")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
for f in ["drive_contrast", "drive_ratio"]:
    v = M[f].reindex(cov.nfl_id_d.unique()); cov["z_" + f] = cov.nfl_id_d.map((M[f] - v.mean()) / v.std())
print("r with frozen drive speed (CBs):", round(cov.drop_duplicates("nfl_id_d")[["z_drive", "z_drive_contrast"]].corr().iloc[0, 1], 2))
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def evaluate(y, ctx, term, extra=""):
    d = cov.dropna(subset=[y, term] + (["wr_skill"] if "wr_skill" in ctx else []) + ([extra.strip(" +")] if extra else [])).copy()
    f = f"{y} ~ {term}{extra} + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; null = []
    for _ in range(400):
        dd = d.copy(); dd[term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[term])
    gain, tot = 0.0, ((d[y] - d[y].mean()) ** 2).sum()
    for c in sorted(d.cls.unique()):
        tr, te = d[d.cls != c], d[d.cls == c]
        te = te[te.route_ran.isin(tr.route_ran.unique())] if "route_ran" in ctx else te
        b = smf.ols(f"{y} ~ {ctx}{extra}", data=tr).fit(); fm = smf.ols(f, data=tr).fit()
        gain += ((te[y] - b.predict(te)) ** 2).sum() - ((te[y] - fm.predict(te)) ** 2).sum()
    return m.params[term], m.pvalues[term], (1 + (np.abs(null) >= abs(t0)).sum()) / 401, gain / tot * 100, len(d), d.nfl_id_d.nunique()


for y, ctx in [("cpoe", "rec_target_prop + C(team_coverage_man_zone)"), ("sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
               ("sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)")]:
    print(f"\n=== {y} ===")
    for term, extra, lab in [("z_drive", "", "drive speed (frozen)"), ("z_drive_contrast", "", "drive contrast (late - early)"),
                             ("z_drive_ratio", "", "drive ratio (late / early)"), ("z_drive_contrast", " + z_drive", "contrast, given drive speed")]:
        b, p, pp, g, n, k = evaluate(y, ctx, term, extra)
        print(f"  {lab:32s} {b:+.4f} per SD (clustered p {p:.3f}, perm p {pp:.3f}) | held-out-class gain {g:+.2f}% | n {n}, CBs {k}")
tg = cov[cov.targeted == 1].dropna(subset=["z_drive_contrast", "completion"]).copy()
m = smf.logit("completion ~ z_drive_contrast + C(team_coverage_man_zone)", data=tg).fit(disp=0)
dv = M.drive_contrast.reindex(cov.nfl_id_d.unique()).dropna()
for q in [10, 50, 90]:
    x = tg.copy(); x["z_drive_contrast"] = (np.percentile(dv, q) - dv.mean()) / dv.std()
    print(f"  expected completion % allowed at drive-contrast p{q} ({np.percentile(dv, q):+.2f} yd/s): {m.predict(x).mean() * 100:.1f}%")
M.to_csv(f"{ROOT}/results/69_drive_contrast_player.csv")
