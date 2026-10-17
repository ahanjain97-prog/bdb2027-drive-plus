"""LT vs RT in pass protection.

Side per dropback = which side of the field's center line the tackle lines up on at the snap. After standardization
the offense faces +x, so +y is the offense's left. The ball is snapped between the hashes (<= 3.1 yd from center) and
a tackle sets up ~2.5-3.5 yd from the ball, so y_field > +0.5 -> LT, < -0.5 -> RT (the middle band is dropped).
(Kick-slide direction was tried first and rejected: per-tackle consistency only ~0.71.)
Each tackle's primary side = his majority side per season (consistency reported).
a) All tackle dropbacks (REG+POST): peak pressure probability allowed and pressure allowed by side.
b) Head-to-head tackle-vs-edge reps: who LTs vs RTs face (rusher hoop turn speed, weight), and whether rusher bend
   hurts more on one side (outcome ~ z_hoop x LT, two-way clustered).
c) What counters bend, by side: outcome ~ z_hoop x z_trait within LT reps and within RT reps (traits: five-yard
   wave speed, mirror speed, arm length, 10-yd split, weight).
d) Player level, by primary side: tackle combine traits -> his pressure probability over expected on all dropbacks.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
tk = meta[(meta.play_type == "DROPBACK") & (meta.lined_up_position == "T")].merge(base[["nfl_position", "display_name"]], left_on="nfl_id", right_index=True)
tk = tk[tk.nfl_position.isin(["T", "G", "C"])]
yi = F.index("y_field")
yf = np.asarray(X[tk.idx.values, PRE, yi], dtype=np.float32)
tk["side"] = np.where(np.abs(yf) < 0.5, None, np.where(yf > 0, "LT", "RT"))
tk = tk[tk.side.notna()]
maj = tk.groupby("nfl_id").side.agg(lambda s: s.value_counts().index[0]); share = tk.groupby("nfl_id").side.agg(lambda s: s.value_counts(normalize=True).iloc[0])
ps = tk.groupby(["nfl_id", "season"]).side.agg(lambda s: s.value_counts(normalize=True).iloc[0]); pn = tk.groupby(["nfl_id", "season"]).size()
print(f"side consistency per tackle-SEASON (>= 100 dropbacks): median {ps[pn >= 100].median():.3f}, 10th pct {ps[pn >= 100].quantile(.1):.3f}")
seas_side = tk.groupby(["nfl_id", "season"]).side.agg(lambda s: s.value_counts().index[0]).unstack()
print(seas_side.reindex(cnt_ := tk.groupby("nfl_id").size().sort_values(ascending=False).index[:16]).join(base.display_name).to_string())
cnt = tk.groupby("nfl_id").size()
print(f"tackle dropbacks with a clear side (|y_field| >= 0.5): {len(tk)}; tackles {tk.nfl_id.nunique()}")
print(f"side consistency per tackle (>= 100 dropbacks): median {share[cnt >= 100].median():.3f}, 10th pct {share[cnt >= 100].quantile(.1):.3f}")
show = pd.DataFrame({"side": maj, "consistency": share.round(2), "dropbacks": cnt}).join(base.display_name)
print(show[show.dropbacks >= 300].sort_values("dropbacks", ascending=False).head(16).to_string())
o = pp.set_index(["game_id", "play_id", "nfl_id"])
key = pd.MultiIndex.from_arrays([tk.game_id, tk.play_id, tk.nfl_id])
tk["ppa"] = o.peak_pressure_probability_allowed.reindex(key).values
tk["pa"] = o.pressure_allowed.reindex(key).astype(float).values
tk["sa"] = pd.to_numeric(o.sack_allowed.reindex(key), errors="coerce").values
tko = tk[tk.season_type.isin(["REG", "POST"]) & (tk.nullified != "Y")].copy()
print("\na) all tackle dropbacks by side (REG+POST):")
print(tko.groupby("side")[["ppa", "pa", "sa"]].agg(["mean", "count"]).round(4).to_string())
g = pd.factorize(tko.nfl_id)[0]
_d = tko.dropna(subset=["ppa", "down", "offense_formation"])
mm = smf.ols("ppa ~ C(side) + C(down) + C(offense_formation)", data=_d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(_d.nfl_id)[0]})
print(f"   LT vs RT ppa difference (RT - LT, context-adjusted, clustered by tackle): {mm.params['C(side)[T.RT]']:+.4f} (p {mm.pvalues['C(side)[T.RT]']:.3f})")
# within-tackle: tackles who played both sides
both = tko.groupby("nfl_id").side.nunique(); bt = tko[tko.nfl_id.isin(both[both == 2].index)]
w = bt.groupby(["nfl_id", "side"]).ppa.agg(["mean", "size"]).unstack()
w = w[(w["size"]["LT"] >= 30) & (w["size"]["RT"] >= 30)]
if len(w) >= 5:
    dd = w["mean"]["RT"] - w["mean"]["LT"]
    print(f"   within-tackle (played both sides, >= 30 each, n {len(w)}): RT - LT ppa mean {dd.mean():+.4f}, t-test p {stats.ttest_1samp(dd, 0).pvalue:.3f}")

# ---------------- b, c) head-to-head tackle vs edge reps
exec(open(f"{ROOT}/code/31_matchup_model.py").read().split("res = []")[0])       # builds `rep` with z-scored traits
te = rep[(rep.ol_align == "T") & (rep.dl_align == "EDGE")].copy()
side_map = tk.set_index(["game_id", "play_id", "nfl_id"]).side
te["side"] = side_map.reindex(pd.MultiIndex.from_arrays([te.game_id, te.play_id, te.ol_id])).values
te = te[te.side.notna()].copy(); te["LT"] = (te.side == "LT").astype(int)
print(f"\nb) tackle-vs-edge head-to-head reps with side: {len(te)} (LT {te.LT.sum()}, RT {(1 - te.LT).sum()}); rushers {te.dl_id.nunique()}, tackles {te.ol_id.nunique()}")
print("   rushers faced by side (rep-weighted means): " + "; ".join(
    f"{s_}: hoop turn speed z {d_.z_dl_hoop.mean():+.2f}, rusher weight {d_.dl_w.mean():.0f} lb, corner turned {d_.corner_turned.mean():.3f}, ppa {d_.ppa.mean():.3f}"
    for s_, d_ in te.groupby("side")))
CTXT = "C(down) + C(ytg_b) + C(blitzing)"
def clus(d):
    return {"groups": np.column_stack([pd.factorize(d.dl_id)[0], pd.factorize(d.ol_id)[0]])}
for y in ["ppa", "corner_turned", "ol_pressure_allowed", "dl_pressure"]:
    d = te.dropna(subset=[y, "z_dl_hoop", "down", "ytg_b", "blitzing"])
    m_ = smf.ols(f"{y} ~ z_dl_hoop * LT + {CTXT}", data=d).fit(cov_type="cluster", cov_kwds=clus(d))
    b0, bi = m_.params["z_dl_hoop"], m_.params["z_dl_hoop:LT"]
    print(f"   {y:20s} rusher-bend slope vs RT {b0:+.4f} (p {m_.pvalues['z_dl_hoop']:.3f}) | vs LT {b0 + bi:+.4f} | interaction p {m_.pvalues['z_dl_hoop:LT']:.3f}  (reps {len(d)})")
print("\nc) what counters rusher bend, by side (interaction rusher hoop z x tackle trait z; negative = trait blunts bend):")
for tr in ["z_ol_wave", "z_ol_mirror", "z_ol_arm", "z_ol_ten", "z_ol_w"]:
    line = []
    for s_ in ["LT", "RT"]:
        for y in ["ppa", "corner_turned"]:
            d = te[te.side == s_].dropna(subset=[y, "z_dl_hoop", tr, "down", "ytg_b", "blitzing"])
            if d.ol_id.nunique() < 8:
                line.append(f"{s_} {y}: n/a"); continue
            m_ = smf.ols(f"{y} ~ z_dl_hoop * {tr} + {CTXT}", data=d).fit(cov_type="cluster", cov_kwds=clus(d))
            k_ = f"z_dl_hoop:{tr}"
            line.append(f"{s_} {y} {m_.params[k_]:+.4f} (p {m_.pvalues[k_]:.3f}, T {d.ol_id.nunique()})")
    print(f"   {tr:12s} | " + " | ".join(line))

# ---------------- d) player level: tackle combine traits -> pressure probability over expected, by primary side
tko["ytg_b"] = pd.cut(tko.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
key2 = tko[["down", "offense_formation", "side"]].astype(str).agg("|".join, axis=1)
tko["ppa_oe"] = tko.ppa - tko.groupby(key2).ppa.transform("mean")
pl_ = tko.groupby("nfl_id").agg(ppa_oe=("ppa_oe", "mean"), n=("ppa_oe", "size"))
pl_ = pl_[pl_.n >= 100].join(maj.rename("side"))
traits = pd.DataFrame({"wave_speed": rep.drop_duplicates("ol_id").set_index("ol_id").ol_wave,
                       "mirror_speed": rep.drop_duplicates("ol_id").set_index("ol_id").ol_mirror})
traits = traits.reindex(base.index)
for c in ["arm_length", "wing_span", "ten_yd_split", "three_cone", "short_shuttle", "combine_weight", "combine_height"]:
    traits[c] = base[c]
print("\nd) tackle traits -> his peak pressure probability over expected (all dropbacks; negative r = trait helps):")
rows = []
for s_ in ["LT", "RT"]:
    P = pl_[pl_.side == s_]
    for c in traits.columns:
        d = pd.concat([traits[c], P.ppa_oe], axis=1).reindex(P.index).dropna()
        if len(d) >= 10:
            r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1]); rows.append({"side": s_, "trait": c, "n": len(d), "r": r, "p": p})
D = pd.DataFrame(rows)
print(D.pivot_table(index="trait", columns="side", values="r").round(2).join(D.pivot_table(index="trait", columns="side", values="n"), rsuffix="_n").to_string())
print("p-values:"); print(D.pivot_table(index="trait", columns="side", values="p").round(3).to_string())
te.to_parquet(f"{ROOT}/data/bend/tackle_edge_reps_with_side.parquet", index=False)
