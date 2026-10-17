"""Step 2: the full value of CB Transition Drive Speed -- target avoidance, completion over expected allowed, EPA per rep.

Expected completion model: logistic regression (sklearn, L2) on ALL targets of pool receivers (REG+POST), pre-throw
context only (no separation, nothing the CB caused): air yards (spline-ish bins), route, receiver alignment, man/zone,
down, distance bucket, field zone, receiver's catch rate over expected on his OTHER targets (leave-one-out).
5-fold cross-fitted predictions -> xComp for every target. CPOE allowed on a coverage rep = completion - xComp.
Target model: same context on all coverage reps -> xTarget; targeted - xTarget = target rate over expected.
Tests (rep level, two-way clustered + player-permutation p, and leave-one-draft-class-out OOS):
  targeted ~ drive speed + receiver target propensity + context;  CPOE ~ drive speed;  EPA per rep ~ drive speed.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(13)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
g = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season_type"]]
t = pp.merge(g, on="game_id")
t = t[(t.season_type != "PRE") & (t.play_nullified_by_penalty != "Y") & t.target.fillna(False).astype(bool) &
      t.nfl_id.isin(base.index[base.nfl_position.isin(["WR", "TE"])]) & t.pass_result.isin(["C", "I", "IN"])].copy()
t["comp"] = (t.pass_result == "C").astype(float)
t["ay_bin"] = pd.cut(t.pass_length.fillna(t.pass_length.median()), [-99, 0, 5, 10, 15, 20, 30, 99], labels=False).astype(str)
t["ytg_b"] = pd.cut(t.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False).astype(str)
t["fz"] = np.where(t.yardline_side == t.possession_team, "own", "opp").astype(str) + pd.cut(t.yardline_number.fillna(30), [0, 20, 40, 51], labels=False).astype(str)
# receiver catch skill: leave-one-out completion rate minus the air-yard-bin mean
t["ay_mean"] = t.groupby("ay_bin").comp.transform("mean")
s_, n_ = t.groupby("nfl_id").comp.transform("sum"), t.groupby("nfl_id").comp.transform("size")
e_ = t.groupby("nfl_id").ay_mean.transform("sum")
t["rec_skill"] = ((s_ - t.comp) - (e_ - t.ay_mean)) / (n_ - 1).clip(lower=1)
CATS = ["ay_bin", "route_ran", "lined_up_position", "team_coverage_man_zone", "down", "ytg_b", "fz"]
def design(df, cats, cols):
    D = pd.get_dummies(df[cats].astype(str), drop_first=False).reindex(columns=cols, fill_value=0) if cols is not None else pd.get_dummies(df[cats].astype(str))
    return D
D = pd.get_dummies(t[CATS].astype(str)); cols = D.columns
Xc = np.column_stack([D.values.astype(float), t.rec_skill.fillna(0).values, t.pass_length.fillna(0).values / 10])
t["xcomp"] = np.nan
for tr, te in KFold(5, shuffle=True, random_state=0).split(Xc):
    m = LogisticRegression(C=0.5, max_iter=2000).fit(Xc[tr], t.comp.values[tr])
    t.iloc[te, t.columns.get_loc("xcomp")] = m.predict_proba(Xc[te])[:, 1]
brier = ((t.comp - t.xcomp) ** 2).mean(); brier0 = ((t.comp - t.comp.mean()) ** 2).mean()
print(f"expected-completion model: {len(t)} pool-receiver targets, cross-fitted Brier {brier:.4f} vs base rate {brier0:.4f} (skill {1 - brier / brier0:.1%})")

cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_enriched.parquet")
xc = t.set_index(["game_id", "play_id", "nfl_id"]).xcomp
cov["xcomp"] = xc.reindex(pd.MultiIndex.from_arrays([cov.game_id, cov.play_id, cov.nfl_id_r])).values
cov["cpoe"] = np.where(cov.targeted == 1, cov.completion - cov.xcomp, np.nan)
# receiver target propensity: targets per route over expected on routes OUTSIDE these coverage reps
r_all = pp[pp.route_ran.notna() & pp.nfl_id.isin(base.index[base.nfl_position.isin(["WR", "TE"])])].copy()
r_all["tg"] = r_all.target.fillna(False).astype(float)
cov_keys = set(zip(cov.game_id, cov.play_id, cov.nfl_id_r))
r_all = r_all[[k not in cov_keys for k in zip(r_all.game_id, r_all.play_id, r_all.nfl_id)]]
prop = (r_all.groupby("nfl_id").tg.mean() - r_all.tg.mean())
cov["rec_target_prop"] = cov.nfl_id_r.map(prop).fillna(0)
cov["cls"] = cov.nfl_id_d.map(base.draft_year)
cov["epa_rep"] = np.where(cov.targeted == 1, cov.epa, 0.0)          # value of the rep to the offense if he was not targeted: 0
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def test(d, y, rhs, label, perm=True):
    d = d.dropna(subset=[y, "z_drive"] + [c.strip() for c in rhs.split("+") if c.strip() and not c.strip().startswith("C(")])
    f = f"{y} ~ z_drive + {rhs}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    pp_ = np.nan
    if perm:
        t0 = smf.ols(f, data=d).fit().tvalues["z_drive"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").z_drive; null = []
        for _ in range(400):
            dd = d.copy(); dd["z_drive"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["z_drive"])
        pp_ = (1 + (np.abs(null) >= abs(t0)).sum()) / 401
    # leave-one-class-out: does the drive coefficient fitted on two classes predict the third (sign + OOS improvement)?
    imp = []
    for c in sorted(d.cls.dropna().unique()):
        tr, te = d[d.cls != c], d[d.cls == c]
        mb = smf.ols(f"{y} ~ {rhs}", data=tr).fit(); mf = smf.ols(f, data=tr).fit()
        te = te[te.route_ran.isin(tr.route_ran.unique())] if "route_ran" in rhs else te
        imp.append(((te[y] - mb.predict(te)) ** 2).mean() - ((te[y] - mf.predict(te)) ** 2).mean())
    print(f"  {label:44s} {m.params['z_drive']:+.4f} per SD (clustered p {m.pvalues['z_drive']:.3f}, perm p {pp_:.3f}) n {len(d)} CBs {d.nfl_id_d.nunique()} | "
          f"held-out-class MSE gain > 0 in {sum(i > 0 for i in imp)}/{len(imp)}")


print("\n=== target avoidance (all coverage reps) ===")
test(cov, "targeted", "rec_target_prop + C(team_coverage_man_zone) + C(route_ran)", "targeted ~ drive (+ receiver propensity, route, cov)")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    test(cov[cov.team_coverage_man_zone == cz], "targeted", "rec_target_prop + C(route_ran)", f"   {cz}", perm=False)
print(f"  base target rate per coverage rep {cov.targeted.mean():.3f}")
print("\n=== completion over expected allowed (targeted reps) ===")
test(cov, "cpoe", "C(team_coverage_man_zone)", "CPOE ~ drive")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    test(cov[cov.team_coverage_man_zone == cz], "cpoe", "rec_target_prop", f"   {cz}", perm=False)
print("\n=== EPA per coverage rep (0 when not targeted) ===")
test(cov, "epa_rep", "rec_target_prop + C(team_coverage_man_zone) + C(route_ran)", "EPA per rep ~ drive")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    test(cov[cov.team_coverage_man_zone == cz], "epa_rep", "rec_target_prop + C(route_ran)", f"   {cz}", perm=False)
# per-season scale: a typical starting CB is in ~350 coverage reps of this kind per season? report per 100 reps
d = cov.dropna(subset=["z_drive"])
m_t = smf.ols("targeted ~ z_drive + rec_target_prop + C(team_coverage_man_zone) + C(route_ran)", data=d).fit()
m_e = smf.ols("epa_rep ~ z_drive + rec_target_prop + C(team_coverage_man_zone) + C(route_ran)", data=d.dropna(subset=["epa_rep"])).fit()
print(f"\nper 100 tight-coverage reps, +1 SD drive speed: {m_t.params['z_drive'] * 100:+.1f} targets, {m_e.params['z_drive'] * 100:+.1f} offensive EPA")
cov.to_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet", index=False)
