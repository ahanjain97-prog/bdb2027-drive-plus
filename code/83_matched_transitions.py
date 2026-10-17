"""Can other DB drills measure the SAME thing as Drive+ (speed 0.8-1.0 s out of a ~45-degree break)?

From 75a's transitions in every DB drill, keep only PHASE-MATCHED ones (look like the 45-degree drill's break):
  turn 25-90 deg, entered at >= 3 yd/s, and an uninterrupted straight run >= 1.0 s after the turn (so steps 3-5 exist).
Per transition: speed 0.8 s and 1.0 s after the turn, post-turn top; z within drill x combine year; player mean per drill.
Questions:
  (1) how many matched transitions each drill has (and per player)
  (2) convergent validity: does each drill's matched drive agree with the frozen 45-degree Drive+ (CBs)?
  (3) reliability: do two different non-45 drills agree with each other (split by drill)?
  (4) NFL validity of an "other drills only" matched-drive score (no 45-degree data at all): zone-match closing,
      completion when targeted, open at the throw -- 3-yd reps, all graded DBs, player-permutation p
  (5) does adding it to Drive+ help out of sample (leave-one-draft-class-out, season-forward)?
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(191)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
T = pd.read_parquet(f"{ROOT}/data/features/db_transitions.parquet")
cbs = base.index[base.nfl_position == "CB"]; dbs = base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]
FORTY5 = "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"
M = T[(T.angle.between(25, 90)) & (T.v_entry >= 3) & (T.run_time >= 1.0) & T.v_10.notna() & T.nfl_id.isin(dbs)].copy()
pd.set_option("display.width", 220)
print("(1) phase-matched transitions by drill (all DBs):")
print(M.groupby("drill").agg(transitions=("k", "size"), players=("nfl_id", "nunique"), CBs=("nfl_id", lambda s: s.isin(cbs).sum() and s[s.isin(cbs)].nunique()),
                             per_player=("nfl_id", lambda s: s.value_counts().median()), angle=("angle", "median")).round(1).to_string())
for f in ["v_08", "v_10", "post_top"]:
    grp = M.groupby(["drill", "yr"])[f]; M[f + "_mz"] = (M[f] - grp.transform("mean")) / grp.transform("std")
M["late"] = M[["v_08_mz", "v_10_mz", "post_top_mz"]].mean(1)
PD = M.groupby(["nfl_id", "drill"]).late.mean().unstack()
others = [c for c in PD.columns if c != FORTY5]
frozen = (GR.drive_plus - 100) / 10
print("\n(2) agreement with frozen Drive+ (CBs; r, n):")
for c in PD.columns:
    j = pd.concat([PD[c], frozen], axis=1, join="inner").reindex([i for i in cbs if i in PD.index]).dropna()
    if len(j) >= 12:
        print(f"   {c:46s} r {j.corr().iloc[0, 1]:+.2f}  (n {len(j)})")
O = PD[others]
cnt = O.notna().sum(1)
other_score = O.mean(1)
j = pd.concat([other_score.rename("other"), frozen.rename("frozen")], axis=1).reindex(cbs).dropna()
print(f"   OTHER-DRILLS composite (mean of matched drive over non-45 drills)  r {j.corr().iloc[0, 1]:+.2f} (n {len(j)})")
print("   DBs with matched transitions in >= 1 other drill:", int((cnt >= 1).sum()), "| >= 2 drills:", int((cnt >= 2).sum()),
      "| DBs with other-drill data but NO 45-degree grade:", int(((cnt >= 1) & ~O.index.isin(GR.index)).sum()))
# (3) reliability: odd vs even other drills within player (drills in a fixed order)
order = {d: i for i, d in enumerate(sorted(others))}
long = O.stack().rename("v").reset_index(); long["par"] = long.drill.map(order) % 2
h = long.groupby(["nfl_id", "par"]).v.mean().unstack().dropna()
r = np.corrcoef(h[0], h[1])[0, 1]
print(f"\n(3) do different non-45 drills agree with each other? split-half r {r:+.2f} (n {len(h)}) -> composite reliability ~{2 * r / (1 + r):.2f}")

# (4) NFL validity
R = pd.read_parquet(f"{ROOT}/data/outcomes/79_scored_coverage_reps.parquet")
R["zone"] = (R.team_coverage_man_zone == "ZONE_COVERAGE").astype(int)
def zs(s):
    v = s.reindex(R.nfl_id_d.unique()); return R.nfl_id_d.map((s - v.mean()) / v.std())
R["T_other"] = zs(other_score); R["T_frozen"] = zs(frozen)
R["T_combo"] = zs(pd.concat([frozen, (other_score - other_score.mean()) / other_score.std()], axis=1).mean(1))


def slope(x, y, term, ctx, n_perm=400):
    x = x.dropna(subset=[y, term]).copy(); f = f"{y} ~ {term} + {ctx}"
    t0 = smf.ols(f, data=x).fit().tvalues[term]; b = smf.ols(f, data=x).fit().params[term]
    vals = x.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; null = []
    for _ in range(n_perm):
        xx = x.copy(); xx[term] = xx.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=xx).fit().tvalues[term])
    return b, (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), x.nfl_id_d.nunique()


TASKS = [("zone-match closing", R[R.cell == "zone-matching"], "db_closing", "wr_skill + C(route_ran) + is_s", 1),
         ("completion when targeted", R[R.targeted == 1], "completion", "C(team_coverage_man_zone) + C(route_ran) + is_s", 100),
         ("open at throw", R, "open", "C(team_coverage_man_zone) + C(route_ran) + is_s", 100)]
print("\n(4) NFL validity per SD (player-permutation p):")
for term, lab in [("T_frozen", "Drive+ (45-degree drill)"), ("T_other", "other drills, phase-matched (no 45 data)"), ("T_combo", "Drive+ + other drills (equal)")]:
    out = []
    for nm, x, y, ctx, sc in TASKS:
        b, p, k = slope(x, y, term, ctx); out.append(f"{nm} {b * sc:+.3f} (p {p:.3f}, {k} DBs)")
    print(f"   {lab:42s} " + " | ".join(out), flush=True)

# (5) out of sample: does the combo beat Drive+ alone?
def oos(d, y, ctx, term, splits):
    d = d.dropna(subset=[y, term]).copy(); gain = tot = 0.0
    for tr, te in splits(d):
        a, b = d[tr].copy(), d[te].copy(); b = b[b.route_ran.isin(a.route_ran.unique())] if "route_ran" in ctx else b
        m0 = smf.ols(f"{y} ~ {ctx}", data=a).fit(); m1 = smf.ols(f"{y} ~ {term} + {ctx}", data=a).fit()
        gain += ((b[y] - m0.predict(b)) ** 2).sum() - ((b[y] - m1.predict(b)) ** 2).sum(); tot += ((b[y] - m0.predict(b)) ** 2).sum()
    return 100 * gain / tot
R["cls"] = R.nfl_id_d.map(base.draft_year)
LOCO = lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())]
SFWD = lambda d: [((d.season <= 2024).values, (d.season == 2025).values)]
print("\n(5) out-of-sample MSE reduction % (same rows for all three; rows need all three scores):")
for nm, x, y, ctx, sc in TASKS:
    xx = x.dropna(subset=["T_frozen", "T_other", "T_combo"]).assign(cls=lambda q: q.nfl_id_d.map(base.draft_year))
    print(f"   {nm:26s} " + " | ".join(f"{lab}: class-out {oos(xx, y, ctx, t, LOCO):+.2f}, season-fwd {oos(xx, y, ctx, t, SFWD):+.2f}"
                                       for t, lab in [("T_frozen", "Drive+"), ("T_other", "other drills"), ("T_combo", "combo")]) + f"  (DBs {xx.nfl_id_d.nunique()})")
PD.to_parquet(f"{ROOT}/data/features/matched_drive_by_drill.parquet")
