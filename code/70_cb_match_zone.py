"""Is CB Transition Drive Speed a match-zone trait?

(1) By coverage shell (team_coverage_type): man family (Cover 0/1/2-man), Cover 2 zone, Cover 3 zone, Cover 4 (quarters),
    Cover 6, and quarters-type match pooled (4 + 6). Drive effect per SD (within-year z, frozen metric) on separation
    gained at the break, separation at the throw, closing after the break; two-way clustered + player permutation p.
(2) By technique, measured from tracking BEFORE the receiver's break (so it is not caused by the break itself):
    pre-break alignment = mean cosine between the CB's and the receiver's velocity over the 0.5 s before the break,
    with the CB moving (> 1 yd/s). 'matching' = cos >= 0.7 (moving with the receiver: pattern-match / trail technique),
    'not matching' = cos < 0.7 (sitting, spot-dropping, or reading). Done for zone reps and man reps separately.
(3) Interaction tests (player permutation): zone-matching vs zone-not-matching; zone-matching vs man.
(4) Scheme fit: share of each defense's CB coverage snaps by shell (all CB dropback snaps in the data), 2023-25.
"""
import json, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
rng = np.random.default_rng(43)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
d_ = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))
yr = base.draft_year.reindex(d_.index)
dz = (d_ - d_.groupby(yr).transform("mean")) / d_.groupby(yr).transform("std")
v = dz.reindex(cov.nfl_id_d.unique()); cov["zd"] = cov.nfl_id_d.map((dz - v.mean()) / v.std())
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet").set_index(["game_id", "play_id", "nfl_id"])
cov["shell"] = meta.team_coverage_type.reindex(pd.MultiIndex.from_arrays([cov.game_id, cov.play_id, cov.nfl_id_d])).values
FAM = {"COVER_0_MAN": "man (0/1/2-man)", "COVER_1_MAN": "man (0/1/2-man)", "COVER_2_MAN": "man (0/1/2-man)", "COVER_2_ZONE": "Cover 2 zone",
       "COVER_3_ZONE": "Cover 3 zone", "COVER_4_ZONE": "Cover 4 quarters", "COVER_6_ZONE": "Cover 6"}
cov["fam"] = cov.shell.map(FAM).fillna("other")

# ---------------- pre-break alignment (technique), from tracking
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = cov.idx_r.values, cov.idx_d.values
vxr, vyr, sr, lr = g(A, "vx"), g(A, "vy"), g(A, "s"), g(A, "a_lat")
vxd, vyd, sd = g(B, "vx"), g(B, "vy"), g(B, "s")
pr = g(A, "post_pass"); vv = V[A] & V[B]
T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1)
w = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
cov["pre_align"] = np.where(w.sum(1) >= 3, np.where(w, cosv, 0).sum(1) / np.maximum(w.sum(1), 1), np.nan)
cov["tech"] = np.where(cov.pre_align >= 0.7, "matching", np.where(cov.pre_align.notna(), "not matching", None))
cov["cell"] = np.where(cov.team_coverage_man_zone == "ZONE_COVERAGE", "zone, " + cov.tech.astype(str), "man, " + cov.tech.astype(str))
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 230)
print("CB coverage reps by shell:", cov.fam.value_counts().to_dict())
print("pre-break technique by coverage:", pd.crosstab(cov.team_coverage_man_zone, cov.tech).to_dict())
print("share 'matching' by shell:", cov.groupby("fam").apply(lambda d: (d.tech == "matching").mean(), include_groups=False).round(2).to_dict())


def eff(d, y, ctx, n_perm=300):
    d = d.dropna(subset=[y, "zd"] + (["wr_skill"] if "wr_skill" in ctx else []))
    if d.nfl_id_d.nunique() < 10 or len(d) < 60:
        return None
    f = f"{y} ~ zd + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    t0 = smf.ols(f, data=d).fit().tvalues["zd"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").zd; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["zd"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["zd"])
    return m.params["zd"], m.pvalues["zd"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d), d.nfl_id_d.nunique()


OUT = [("sep_gain_break", "wr_skill + C(route_ran)"), ("sep_at_throw", "wr_skill + C(route_ran)"), ("db_closing", "wr_skill + C(route_ran)")]
rows = []
for unit_col, units in [("fam", ["man (0/1/2-man)", "Cover 2 zone", "Cover 3 zone", "Cover 4 quarters", "Cover 6"]),
                        ("cell", ["zone, matching", "zone, not matching", "man, matching", "man, not matching"])]:
    for u in units:
        sub = cov[cov[unit_col] == u]
        for y, ctx in OUT:
            r = eff(sub, y, ctx)
            if r:
                rows.append({"group": u, "outcome": y, "b_per_SD": r[0], "p_clustered": r[1], "p_perm": r[2], "reps": r[3], "CBs": r[4]})
q46 = cov[cov.fam.isin(["Cover 4 quarters", "Cover 6"])]
for y, ctx in OUT:
    r = eff(q46, y, ctx)
    if r:
        rows.append({"group": "quarters-type (4 + 6)", "outcome": y, "b_per_SD": r[0], "p_clustered": r[1], "p_perm": r[2], "reps": r[3], "CBs": r[4]})
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/70_cb_match_zone.csv", index=False)
print("\n=== drive-speed effect (per SD) by coverage shell and by pre-break technique ===")
print(R.pivot_table(index="group", columns="outcome", values="b_per_SD").round(3).to_string())
print("\nplayer-permutation p:")
print(R.pivot_table(index="group", columns="outcome", values="p_perm").round(3).to_string())
print("\nreps / CBs:", R.drop_duplicates("group").set_index("group")[["reps", "CBs"]].to_dict("index"))


def interaction(d, a_lab, b_lab, col, y, ctx, n_perm=400):
    d = d[d[col].isin([a_lab, b_lab])].dropna(subset=[y, "zd", "wr_skill"]).copy()
    d["is_a"] = (d[col] == a_lab).astype(float)
    f = f"{y} ~ zd * is_a + {ctx}"
    t0 = smf.ols(f, data=d).fit().tvalues["zd:is_a"]; b0 = smf.ols(f, data=d).fit().params["zd:is_a"]
    vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").zd; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["zd"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["zd:is_a"])
    return b0, (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d)


print("\n=== interactions (difference in the drive slope; player-permutation p) ===")
for y, ctx in OUT:
    for a_lab, b_lab, col in [("zone, matching", "zone, not matching", "cell"), ("zone, matching", "man, matching", "cell"),
                              ("Cover 4 quarters", "Cover 3 zone", "fam")]:
        b0, p, n = interaction(cov, a_lab, b_lab, col, y, ctx)
        print(f"  {y:15s} {a_lab} minus {b_lab}: {b0:+.3f} yd/SD (perm p {p:.3f}, reps {n})")

# ---------------- (4) scheme fit: which defenses play these shells
mm = meta.reset_index()
mm = mm[(mm.play_type == "DROPBACK") & mm.nfl_id.isin(base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]) & mm.season_type.isin(["REG", "POST"])]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")[["game_id", "play_id", "nfl_id", "team_abbr"]]
mm = mm.merge(pp, on=["game_id", "play_id", "nfl_id"]).drop_duplicates(["game_id", "play_id", "team_abbr"])
mm["fam"] = mm.team_coverage_type.map(FAM).fillna("other")
mix = pd.crosstab(mm.team_abbr, mm.fam, normalize="index")
mix["plays"] = mm.groupby("team_abbr").size()
mix = mix[mix.plays >= 150]
print("\nleague dropback coverage mix (plays with a pool DB on the field):", mm.fam.value_counts(normalize=True).round(3).to_dict())
for col in ["Cover 4 quarters", "Cover 6", "Cover 3 zone", "man (0/1/2-man)"]:
    print(f"  most {col}: " + ", ".join(f"{t} {v:.0%}" for t, v in mix[col].sort_values(ascending=False).head(6).items()))
mix.to_csv(f"{ROOT}/results/70_team_coverage_mix.csv")
