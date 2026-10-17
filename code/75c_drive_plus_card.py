"""Drive+ grade card: grade every CB (100 = average, 10 = 1 SD) and check NFL outcomes by grade tier.

Version: argv[1] in {A, B, C} (outcome-free versions only; A = frozen 45-degree drill). Tiers: 110+ "plus-plus",
100-110 "plus", 90-100 "minus", < 90 "minus-minus". Per tier (context-adjusted where noted):
  zone-matching closing (yd/s, adj)   separation gained at break (yd, all reps, adj)   completion % allowed when targeted
  CPOE (pp)   EPA per target (offense)
Top-vs-bottom tier difference with a CB bootstrap 95% CI. Leaderboard of graded CBs with their NFL numbers.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
VER = sys.argv[1] if len(sys.argv) > 1 else "A"
rng = np.random.default_rng(127)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
D = pd.read_parquet(f"{ROOT}/data/features/drive_plus_inputs.parquet")
g = D[VER].dropna(); grade = 100 + 10 * (g - g.mean()) / g.std()
TIERS = [(110, 999, "110+"), (100, 110, "100-110"), (90, 100, "90-100"), (-999, 90, "<90")]
tier = grade.map(lambda v: next(l for lo, hi, l in TIERS if lo <= v < hi))
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
cov = cov[cov.nfl_id_d.isin(grade.index)].copy()
cov["tier"] = cov.nfl_id_d.map(tier)
z = cov[cov.cell == "zone-matching"].dropna(subset=["db_closing", "wr_skill"]).copy()
z["adj"] = smf.ols("db_closing ~ wr_skill + C(route_ran)", data=z).fit().resid + z.db_closing.mean()
a = cov.dropna(subset=["sep_gain_break", "wr_skill"]).copy()
a["adj"] = smf.ols("sep_gain_break ~ wr_skill + C(team_coverage_man_zone) + C(route_ran)", data=a).fit().resid + a.sep_gain_break.mean()
t = cov[cov.targeted == 1].copy()


def by_tier(fn):
    return {l: fn(l) for _, _, l in TIERS}


rows = []
for _, _, l in TIERS:
    ids = tier.index[tier == l]
    zz, aa, tt = z[z.nfl_id_d.isin(ids)], a[a.nfl_id_d.isin(ids)], t[t.nfl_id_d.isin(ids)]
    rows.append({"tier": l, "CBs graded": len(ids), "CBs with NFL reps": cov[cov.nfl_id_d.isin(ids)].nfl_id_d.nunique(),
                 "zone-match closing (yd/s, adj)": zz.adj.mean(), "zone-match reps": len(zz),
                 "sep gained at break (yd, adj)": aa.adj.mean(), "completion % allowed": 100 * tt.completion.mean(),
                 "CPOE (pp)": 100 * tt.cpoe.mean(), "EPA per target": tt.epa.mean(), "targets": len(tt)})
C = pd.DataFrame(rows).set_index("tier")
pd.set_option("display.width", 240)
print(f"Drive+ version {VER}: graded CBs {len(grade)}, tier sizes {tier.value_counts().reindex([l for _, _, l in TIERS]).to_dict()}")
print(C.round(3).to_string())


def boot(df, col, hi="110+", lo="<90", n=2000):
    ih, il = tier.index[tier == hi], tier.index[tier == lo]
    ih = [i for i in ih if i in set(df.nfl_id_d)]; il = [i for i in il if i in set(df.nfl_id_d)]
    gm = df.groupby("nfl_id_d")[col].agg(["sum", "size"])
    out = []
    for _ in range(n):
        sh, sl = gm.loc[rng.choice(ih, len(ih))].sum(), gm.loc[rng.choice(il, len(il))].sum()
        out.append(sh["sum"] / sh["size"] - sl["sum"] / sl["size"])
    est = gm.loc[ih, "sum"].sum() / gm.loc[ih, "size"].sum() - gm.loc[il, "sum"].sum() / gm.loc[il, "size"].sum()
    return est, np.percentile(out, [2.5, 97.5])


print("\n110+ minus <90 (CB bootstrap 95% CI):")
for lab, df, col, sc in [("zone-match closing (yd/s)", z, "adj", 1), ("sep gained at break (yd)", a, "adj", 1),
                         ("completion % allowed", t.dropna(subset=["completion"]), "completion", 100), ("CPOE (pp)", t.dropna(subset=["cpoe"]), "cpoe", 100),
                         ("EPA per target", t.dropna(subset=["epa"]), "epa", 1)]:
    e, ci = boot(df, col)
    print(f"  {lab:28s} {e * sc:+.3f} [{ci[0] * sc:+.3f}, {ci[1] * sc:+.3f}]")
pl = pd.DataFrame({"Drive+": grade.round(0), "tier": tier, "class": base.draft_year.reindex(grade.index), "name": base.display_name.reindex(grade.index),
                   "forty": base.forty.reindex(grade.index)})
pl["NFL reps"] = cov.groupby("nfl_id_d").size(); pl["zone-match closing (adj)"] = z.groupby("nfl_id_d").adj.mean()
pl["targets"] = t.groupby("nfl_id_d").size(); pl["completion % allowed"] = 100 * t.groupby("nfl_id_d").completion.mean()
pl = pl.sort_values("Drive+", ascending=False)
pl.to_csv(f"{ROOT}/results/75c_drive_plus_grades_{VER}.csv")
print("\nr(Drive+, official 40 time) over graded CBs: %+.2f" % pl[["Drive+", "forty"]].corr().iloc[0, 1])
print("\ntop 10:\n" + pl.head(10).round(2).to_string() + "\n\nbottom 10:\n" + pl.tail(10).round(2).to_string())
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
order = [l for _, _, l in TIERS][::-1]; col = ["#c0392b", "#e59866", "#7fb3d5", "#2471a3"]
for a_, m, lab in [(ax[0], "zone-match closing (yd/s, adj)", "zone-match closing after the break (yd/s)"),
                   (ax[1], "completion % allowed", "completion % allowed when targeted"), (ax[2], "EPA per target", "offense EPA per target")]:
    a_.bar(order, C.loc[order, m], color=col); a_.set_title(lab, fontsize=10); a_.set_xlabel("Drive+ tier")
    for i, l in enumerate(order):
        a_.annotate(f"{int(C.loc[l, 'CBs with NFL reps'])} CBs", (i, C.loc[l, m]), ha="center", va="bottom", fontsize=8)
fig.suptitle(f"Drive+ (100 = average CB, 10 = 1 SD; version {VER}) vs NFL coverage", fontsize=12); fig.tight_layout()
fig.savefig(f"{ROOT}/results/75c_drive_plus_card_{VER}.png", dpi=140)
