"""Drive+ : a Stuff+-style headline grade (100 = average CB, 10 = 1 SD), single-drill vs multi-drill vs learned.

Per-drill drive measures (player level, within drill x year (x angle) z): 45-degree drill = FROZEN metric; 90-degree drill
= its drive top (73b); box, W, Teryl Austin 1 and 2 = post-turn drive (post_top, 75a).
Versions:
  A  Drive+ (45 only)          the frozen metric
  B  Drive+ multi, equal        mean of the available per-drill drive z (every drill counts the same)
  C  Drive+ multi, convergent   weighted mean, weight = the drill's correlation with the 45-degree drill (outcome-free)
  D  Drive+ learned (Stuff+)    ridge from all per-drill drive features (post_top, speed 0.4 / 0.8 s after the turn) to
                                the NFL target, fitted ONLY inside training folds (Stuff+ = model of expected outcome)
  E  Drive+ learned, drive tops ridge on the 6 per-drill drive tops only (fewer weights)
Test: rep-level MSE reduction vs context-only, leave-one-draft-class-out and season-forward (2023-24 -> 2025), with a
CB-identity permutation null for class-out; outcomes = zone-matching closing, separation gained (all), sep at throw, CPOE.
Then the grade card: tiers of the chosen Drive+ vs NFL outcomes, and a leaderboard.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(113)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cbs = base.index[base.nfl_position == "CB"]
Z45 = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
Z90 = pd.read_parquet(f"{ROOT}/data/features/cb_bp90_parts.parquet")
W = pd.read_parquet(f"{ROOT}/data/features/db_drive_by_drill.parquet")
D = pd.DataFrame(index=cbs)
D["d45"] = Z45.top; D["d90"] = Z90.top
SHORT = {"BOX_DRILL": "box", "W_DRILL": "w", "TERYL_AUSTIN_DRILL_1": "ta1", "TERYL_AUSTIN_DRILL_2": "ta2"}
for dr, sh in SHORT.items():
    D[f"d{sh}"] = W.get(f"{dr}|post_top")
TOPS = ["d45", "d90", "dbox", "dw", "dta1", "dta2"]
FEATS = TOPS + ["v10_45", "v04_45", "v08_90", "v04_90"] + [f"{sh}_{f}" for sh in SHORT.values() for f in ("v_04", "v_08")]
D["v10_45"], D["v04_45"] = Z45.v_10, Z45.v_04
D["v08_90"], D["v04_90"] = Z90.v_08, Z90.v_04
for dr, sh in SHORT.items():
    for f in ("v_04", "v_08"):
        D[f"{sh}_{f}"] = W.get(f"{dr}|{f}")
D = D.reindex(cbs)
D = (D - D.mean()) / D.std()
convw = {c: max(D[[c, "d45"]].dropna().corr().iloc[0, 1], 0) if c != "d45" else 1.0 for c in TOPS}
print("convergent weights (r with the 45-degree drill, floored at 0):", {k: round(v, 2) for k, v in convw.items()})
print("CBs with each drill:", D[TOPS].notna().sum().to_dict())
D["A"] = D.d45
D["B"] = D[TOPS].mean(1)
wv = pd.Series(convw); D["C"] = (D[TOPS] * wv).sum(1, min_count=1) / D[TOPS].notna().mul(wv).sum(1).replace(0, np.nan)
print("r(A, B) %.2f | r(A, C) %.2f | CBs gradable: A %d, B %d, C %d" % (D[["A", "B"]].corr().iloc[0, 1], D[["A", "C"]].corr().iloc[0, 1],
      D.A.notna().sum(), D.B.notna().sum(), D.C.notna().sum()))
D.to_parquet(f"{ROOT}/data/features/drive_plus_inputs.parquet")

cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
for c in ["A", "B", "C"] + FEATS:
    v = D[c].reindex(cov.nfl_id_d.unique()); cov["c_" + c] = cov.nfl_id_d.map((D[c] - v.mean()) / v.std())


def block(name, a, b):
    if name in ("A", "B", "C"):
        return a["c_" + name].values, b["c_" + name].values
    cols = ["c_" + c for c in (FEATS if name == "D" else TOPS)]
    Pa = a.groupby("nfl_id_d").agg(res=("res", "mean"), n=("res", "size")).join(a.groupby("nfl_id_d")[cols].first())
    Pa = Pa[Pa[cols].notna().any(axis=1)]
    r = RidgeCV(alphas=np.logspace(-1, 3.5, 25)).fit(Pa[cols].fillna(0).values, Pa.res.values, sample_weight=Pa.n.values)
    f = lambda d: np.where(d[cols].notna().any(axis=1), r.predict(d[cols].fillna(0).values), np.nan)
    return f(a), f(b)


def oos(d, y, ctx, name, splits):
    d = d.dropna(subset=[y] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    gain = tot = 0.0
    for tr, te in splits(d):
        a, b = d[tr].copy(), d[te].copy()
        if "route_ran" in ctx:
            b = b[b.route_ran.isin(a.route_ran.unique())]
        m0 = smf.ols(f"{y} ~ {ctx}", data=a).fit(); a["res"] = m0.resid; b["res"] = b[y] - m0.predict(b)
        a["xb"], b["xb"] = block(name, a, b)
        a2, b2 = a.dropna(subset=["xb"]), b.dropna(subset=["xb"])
        s = smf.ols("res ~ xb", data=a2).fit()
        gain += (b2.res ** 2).sum() - ((b2.res - s.predict(b2)) ** 2).sum(); tot += (b2.res ** 2).sum()
    return 100 * gain / tot


LOCO = lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())]
SFWD = lambda d: [((d.season <= 2024).values, (d.season == 2025).values)]
TASKS = [("zone-matching closing", cov[cov.cell == "zone-matching"], "db_closing", "wr_skill + C(route_ran)"),
         ("sep gained (all)", cov, "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
         ("sep at throw (all)", cov, "sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
         ("CPOE", cov[cov.targeted == 1], "cpoe", "rec_target_prop + C(team_coverage_man_zone)")]
res = []
for lab, d, y, ctx in TASKS:
    for sample in ["CBs with the 45-degree drill", "all CBs with any drill"]:
        dd = d[d.c_A.notna()] if sample.startswith("CBs with the 45") else d[d.c_B.notna()]
        for name in ["A", "B", "C", "D", "E"]:
            if name == "A" and sample.startswith("all"):
                continue
            r0 = {"outcome": lab, "sample": sample, "version": name, "CBs": dd.nfl_id_d.nunique()}
            r0["class-out %"] = oos(dd, y, ctx, name, LOCO); r0["season-fwd %"] = oos(dd, y, ctx, name, SFWD)
            ids = dd.nfl_id_d.unique(); cc = [c for c in dd.columns if c.startswith("c_")]
            per = dd.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[cc]; null = []
            for _ in range(40):
                mp = dict(zip(ids, rng.permutation(ids)))
                null.append(oos(dd.drop(columns=cc).join(per.rename(index=mp), on="nfl_id_d"), y, ctx, name, LOCO))
            r0["class-out perm p"] = (1 + (np.array(null) >= r0["class-out %"]).sum()) / 41
            res.append(r0); print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r0.items()}, flush=True)
R = pd.DataFrame(res); R.to_csv(f"{ROOT}/results/75b_drive_plus_oos.csv", index=False)
pd.set_option("display.width", 240)
print("\n=== out-of-sample MSE reduction (%) by Drive+ version ===")
print(R.pivot_table(index=["sample", "version"], columns="outcome", values="class-out %").round(2).to_string())
print(R.pivot_table(index=["sample", "version"], columns="outcome", values="season-fwd %").round(2).to_string())
print(R.pivot_table(index=["sample", "version"], columns="outcome", values="class-out perm p").round(3).to_string())
