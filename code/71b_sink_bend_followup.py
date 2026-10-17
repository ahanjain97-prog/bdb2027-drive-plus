"""Follow-ups to 71 (CB sink / bend / hip flip).

(A) Combine reliability done right: attempt-level trait means, odd vs even ATTEMPT of the same drill (71's odd/even
    events inside one clip alternate left/right turns, which made those halves anti-correlated).
(B) The NFL version of each trait, measured in the CB's own movement at the receiver's break (71): split-half
    reliability over his reps, and whether it predicts coverage at all -- cross-fit at player level: trait mean over
    his odd reps vs the context-adjusted outcome mean over his even reps (and the reverse), by technique cell.
    If sink does not matter even measured in games, chasing it at the combine is pointless.
(C) Combine-vs-combine matchups (WR combine trait x CB combine trait), the coverage analogue of rusher-bend x
    blocker-feet. WR traits limited to those that repeat (52 catalog, reliability >= .40), within-year z among WRs:
      wr_sink   breaking-route sustained braking peak (.40)   wr_burst  route accel over the first 1 s (.70)
      wr_speed  route top speed (.59)                       wr_ss_brake short-shuttle mean braking decel (.55)
    CB traits: drive (frozen), sink (composite of sink_rate, -sink_dist), flip_speed.
    Model: y ~ cb * wr + wr_skill + C(route_ran), zone-matching and man-matching reps, WR targets only;
    interaction p from permuting the CB trait across CBs (and the WR trait across WRs); BH over the family.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(53)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
E = pd.read_parquet(f"{ROOT}/data/features/cb_sink_bend_events.parquet")
P = pd.read_parquet(f"{ROOT}/data/features/cb_sink_bend_player.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
cbs = base.index[base.nfl_position == "CB"]
TR = ["sink_rate", "sink_dist", "plant_depth", "exit_rate", "bend_speed", "bend_lat", "bend_radius", "flip_speed", "flip_time"]
pd.set_option("display.width", 240)

# ---------------- (A)
print("=== (A) combine reliability: odd vs even attempt of the same drill, CBs + safeties (Spearman-Brown) ===")
A = E[E.kind != "attempt"].groupby(["event_id", "nfl_id", "drill"])[[f + "_z" for f in TR]].mean().reset_index()
t0 = E.drop_duplicates("event_id").set_index("event_id").index
A["ord"] = A.event_id.map({e: i for i, e in enumerate(sorted(t0))})
A = A.sort_values(["nfl_id", "drill", "ord"]); A["par"] = A.groupby(["nfl_id", "drill"]).cumcount() % 2
two = A.groupby(["nfl_id", "drill"]).par.transform("max") == 1
rel = {}
for f in TR:
    h = A[two].dropna(subset=[f + "_z"]).groupby(["nfl_id", "par"])[f + "_z"].mean().unstack().dropna()
    r = np.corrcoef(h[0], h[1])[0, 1]; rel[f] = (round(2 * r / (1 + r), 2), len(h))
print("  " + " | ".join(f"{k} {v[0]:+.2f} (n {v[1]})" for k, v in rel.items()))
by_drill = {}
for f in ["sink_rate", "sink_dist", "bend_speed", "flip_speed"]:
    for dr, g in A[two].dropna(subset=[f + "_z"]).groupby("drill"):
        h = g.groupby(["nfl_id", "par"])[f + "_z"].mean().unstack().dropna()
        if len(h) >= 20:
            by_drill[(f, dr)] = round(np.corrcoef(h[0], h[1])[0, 1], 2)
print("  single-drill attempt-to-attempt r:", {f"{a}|{b[:18]}": v for (a, b), v in by_drill.items()})

# ---------------- (B)
print("\n=== (B) NFL sink / bend / flip in the CB's own movement at the receiver's break ===")
G = ["g_sink_rate", "g_min_speed", "g_bend_lat", "g_bend_speed", "g_flip"]
cov = cov.sort_values(["nfl_id_d", "game_id", "play_id"]); cov["half"] = cov.groupby("nfl_id_d").cumcount() % 2
OUT = ["sep_gain_break", "sep_at_throw", "db_closing"]
for y in OUT:
    d = cov.dropna(subset=[y, "wr_skill"])
    cov.loc[d.index, y + "_res"] = smf.ols(f"{y} ~ wr_skill + C(route_ran) + C(cell)", data=d).fit().resid
for cell in ["all", "zone-matching", "man-matching"]:
    d = cov if cell == "all" else cov[cov.cell == cell]
    Hm = d.groupby(["nfl_id_d", "half"])[G + [y + "_res" for y in OUT]].mean()
    n = d.groupby(["nfl_id_d", "half"]).size().unstack()
    ok = n.index[(n.min(1) >= 5)]
    H0, H1 = Hm.xs(0, level=1).reindex(ok), Hm.xs(1, level=1).reindex(ok)
    relg = {g: round(np.corrcoef(H0[g], H1[g])[0, 1], 2) for g in G}
    res = []
    for g in G:
        for y in OUT:
            r = 0.5 * (pd.concat([H0[g], H1[y + "_res"]], axis=1).corr().iloc[0, 1] + pd.concat([H1[g], H0[y + "_res"]], axis=1).corr().iloc[0, 1])
            res.append((g, y, r))
    print(f"  [{cell}] CBs {len(ok)} | split-half reliability:", relg)
    print("     trait(odd reps) -> outcome(even reps), r:", " | ".join(f"{g}->{y.split('_')[0]}{'_' + y.split('_')[-1] if y != 'db_closing' else ''} {r:+.2f}" for g, y, r in res))

# ---------------- (C)
print("\n=== (C) combine x combine matchups: CB trait x WR trait (WR targets only) ===")
TC = pd.read_parquet(f"{ROOT}/data/features/trait_composites.parquet")
wrs = base.index[base.nfl_position == "WR"]
W = pd.DataFrame({"wr_sink": TC["WR|breaking_routes|brake_sustained_peak"], "wr_burst": TC["WR|routes|accel_0_1s"],
                  "wr_speed": TC["WR|routes|top_speed"]}).reindex(wrs)
af = pd.read_parquet(f"{ROOT}/data/features/attempt_features.parquet")
ss = af[(af.drill == "SHORT_SHUTTLE") & af.nfl_id.isin(wrs)].groupby("nfl_id").brk_mean_brake_decel.mean() if "drill" in af.columns else None
if ss is not None:
    W["wr_ss_brake"] = ss
wy = base.draft_year.reindex(W.index)
W = (W - W.groupby(wy).transform("mean")) / W.groupby(wy).transform("std")
cb = pd.DataFrame({"cb_drive": P["drive"], "cb_sink": (P["sink_rate"] - P["sink_dist"]) / 2, "cb_flip": P["flip_speed"]})
cv = cov[cov.nfl_id_r.isin(wrs) & cov.cell.isin(["zone-matching", "man-matching"])].copy()
for c in W.columns:
    v = W[c].reindex(cv.nfl_id_r.unique()); cv[c] = cv.nfl_id_r.map((W[c] - v.mean()) / v.std())
for c in cb.columns:
    v = cb[c].reindex(cv.nfl_id_d.unique()); cv[c] = cv.nfl_id_d.map((cb[c] - v.mean()) / v.std())
print("  WR targets in matching reps:", cv.cell.value_counts().to_dict(), "| WRs", cv.nfl_id_r.nunique(), "CBs", cv.nfl_id_d.nunique())
print("  WR combine trait corr:", W.reindex(cv.nfl_id_r.unique()).corr().round(2).to_dict())
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def inter(d, a, b, y, n_perm=200):
    d = d.dropna(subset=[a, b, y, "wr_skill"]).copy()
    f = f"{y} ~ {a} * {b} + wr_skill + C(route_ran)"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    t0 = smf.ols(f, data=d).fit().tvalues[f"{a}:{b}"]
    out = []
    for who, col in (("nfl_id_d", a), ("nfl_id_r", b)):
        vals = d.drop_duplicates(who).set_index(who)[col]; null = []
        for _ in range(n_perm):
            dd = d.copy(); dd[col] = dd[who].map(pd.Series(rng.permutation(vals.values), index=vals.index))
            null.append(smf.ols(f, data=dd).fit().tvalues[f"{a}:{b}"])
        out.append((1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1))
    return m.params[f"{a}:{b}"], m.pvalues[f"{a}:{b}"], out[0], out[1], m.params[a], len(d)


rows = []
for cell in ["zone-matching", "man-matching"]:
    for a in cb.columns:
        for b in W.columns:
            for y in OUT:
                r = inter(cv[cv.cell == cell], a, b, y)
                rows.append({"cell": cell, "cb": a, "wr": b, "outcome": y, "b_inter": r[0], "p_clustered": r[1], "p_perm_cb": r[2],
                             "p_perm_wr": r[3], "cb_main": r[4], "reps": r[5]})
    print(f"  {cell} done", flush=True)
R = pd.DataFrame(rows); R["p_perm"] = R[["p_perm_cb", "p_perm_wr"]].max(1)
R["q_bh"] = multipletests(R.p_perm, method="fdr_bh")[1]
R.to_csv(f"{ROOT}/results/71b_matchups.csv", index=False)
print(R.sort_values("p_perm").head(15).round(3).to_string(index=False))
print(f"\n  interactions at p_perm < .05: {(R.p_perm < .05).sum()} of {len(R)} (chance {0.05 * len(R):.1f}); min BH q {R.q_bh.min():.3f}")
