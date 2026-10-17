"""How much signal is recoverable? Attenuation, a composite criterion, and one cheap augmentation.

(1) Reliability of the criterion: split-half (odd/even reps) reliability of each player's mean on each outcome and on a
    composite coverage score (per-rep z of sep gained (-), closing (+), sep at throw (-), open (-); mean of available).
(2) Player-level correlation of Drive+ with each criterion, observed and disattenuated for both reliabilities
    (Drive+ two-attempt reliability .66; criterion reliability from (1)).
(3) Does the composite give a stronger / more stable Drive+ slope than any single outcome (wide reps, perm p)?
(4) One augmentation that was never tested on its own: tracked 40-yard-dash top speed next to Drive+, leave-one-class-out.
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; rng = np.random.default_rng(229)
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
W["cls"] = W.nfl_id_d.map(base.draft_year)
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other")
# context-adjust each outcome, then z within rep sample
OUT = {"sep_gain_break": -1, "db_closing": +1, "sep_at_throw": -1, "open": -1}
for y, sgn in OUT.items():
    d = W.dropna(subset=[y, "wr_skill"]); W.loc[d.index, y + "_r"] = smf.ols(f"{y} ~ wr_skill + C(zone) + C(route_ran) + is_s", data=d).fit().resid
    W[y + "_z"] = sgn * (W[y + "_r"] - W[y + "_r"].mean()) / W[y + "_r"].std()
W["composite"] = W[[y + "_z" for y in OUT]].mean(axis=1, skipna=True)
W = W.sort_values(["nfl_id_d", "game_id", "play_id"]); W["half"] = W.groupby("nfl_id_d").cumcount() % 2
print("=== (1) criterion reliability: split-half of player means (players with >= 8 reps per half) ===")
rel = {}
for y in [k + "_z" for k in OUT] + ["composite"]:
    h = W.dropna(subset=[y]).groupby(["nfl_id_d", "half"])[y].agg(["mean", "size"]).unstack(); h = h[h["size"].min(1) >= 8]
    r = np.corrcoef(h["mean"][0], h["mean"][1])[0, 1]; rel[y] = 2 * r / (1 + r); print(f"  {y:18s} split-half r {r:+.2f} -> reliability of the full mean {rel[y]:.2f} (n {len(h)})")
print("\n=== (2) Drive+ vs player means: observed r and disattenuated r (Drive+ reliability .66) ===")
pm = W.groupby("nfl_id_d").agg(**{y: (y, "mean") for y in [k + "_z" for k in OUT] + ["composite"]}, n=("dp", "size"), dp=("dp", "first"))
pm = pm[pm.n >= 10]
for y in [k + "_z" for k in OUT] + ["composite"]:
    r = pm[[y, "dp"]].corr().iloc[0, 1]; print(f"  {y:18s} observed r {r:+.2f} | disattenuated {r / np.sqrt(.66 * max(rel[y], .05)):+.2f}  (players {len(pm)})")
print("\n=== (3) Drive+ slope per 10 on the composite (SD units) vs single outcomes, wide reps, one-sided perm p ===")
def slope(d, y, ctx, n_perm=300):
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"; mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]}); t0 = mod.fit().tvalues["dp"]
    vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"])
    return m.params["dp"], m.bse["dp"], (1 + (np.array(null) >= t0).sum()) / (n_perm + 1), used.nfl_id_d.nunique()
for lab, d in [("all reps", W), ("zone-matching", W[W.cell == "zone-matching"]), ("man-matching", W[W.cell == "man-matching"])]:
    for y in ["composite"] + [k + "_z" for k in OUT]:
        b, se, p, k = slope(d, y, "is_s"); print(f"  {lab:14s} {y:18s} {b:+.3f} SD per 10 (SE {se:.3f}, t {b / se:+.1f}, p1 {p:.3f}, DBs {k})")
print("\n=== (4) augmentation: tracked 40 top speed next to Drive+, leave-one-class-out MSE reduction on the composite ===")
cat = pd.read_parquet(f"{ROOT}/data/features/catalog_CB.parquet")
f40 = cat["drill:FORTY_YARD_DASH:top_speed"]; v = f40.reindex(W.nfl_id_d.unique()); W["f40"] = W.nfl_id_d.map((f40 - v.mean()) / v.std())
def oos(d, y, terms):
    d = d.dropna(subset=[y] + terms).copy(); gain = tot = 0.0
    for c in sorted(d.cls.unique()):
        a, b = d[d.cls != c], d[d.cls == c]; m0 = smf.ols(f"{y} ~ is_s", data=a).fit(); m1 = smf.ols(f"{y} ~ is_s + {' + '.join(terms)}", data=a).fit()
        gain += ((b[y] - m0.predict(b)) ** 2).sum() - ((b[y] - m1.predict(b)) ** 2).sum(); tot += ((b[y] - m0.predict(b)) ** 2).sum()
    return 100 * gain / tot
d40 = W.dropna(subset=["f40"])
for terms, lab in [(["dp"], "Drive+ alone"), (["f40"], "40 top speed alone"), (["dp", "f40"], "Drive+ + 40 top speed")]:
    print(f"  {lab:24s} composite {oos(d40, 'composite', terms):+.2f}% | completion {oos(d40[d40.targeted == 1], 'completion', terms):+.2f}%  (DBs {d40.nfl_id_d.nunique()})")
