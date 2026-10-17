"""Why do the catch outcomes (completion, EPA per target) show only in 2025 when the movement outcomes show in both seasons?
A movement vs catches by season (same reps)    B is 2025 really different? dp x 2025 interaction, player-permutation
C combine class (2024 clips truncated)          D same DBs in both seasons      E NFL experience
F target mix by season                          G is 2025 driven by a few DBs? (leave-one-DB-out)
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; rng = np.random.default_rng(503)
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
W["cls"] = W.nfl_id_d.map(GR.draft_year); W["exp"] = W.season - W.cls + 1
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other")
T = W[W.targeted == 1].merge(pd.read_parquet(f"{ROOT}/data/outcomes/98_wide_targets_cpoe.parquet")[["game_id", "play_id", "nfl_id_r", "xc_sep", "xc_ctx", "cpoe_sep", "pass_length"]],
                              on=["game_id", "play_id", "nfl_id_r"], how="left")
T = T.merge(pp[["game_id", "play_id", "nfl_id", "expected_points_added"]].rename(columns={"nfl_id": "nfl_id_r", "expected_points_added": "epa"}), on=["game_id", "play_id", "nfl_id_r"], how="left")
DIR = {"open": -1, "sep_at_throw": -1, "sep_gain_break": -1, "completion": -1, "epa": -1, "xc_sep": -1, "cpoe_sep": -1, "db_closing": +1}
SC = {"open": 100, "completion": 100, "xc_sep": 100, "cpoe_sep": 100}


def est(d, y, ctx="C(zone) + C(route_ran) + is_s", n_perm=300, short=False):
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"
    try: mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]; fit = mod.fit()
    except Exception: return "n/a"
    if used.nfl_id_d.nunique() < 6: return f"too few (n {len(used)}, DBs {used.nfl_id_d.nunique()})"
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]}); sg = DIR.get(y, -1); sc = SC.get(y, 1)
    t0 = sg * fit.tvalues["dp"]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(sg * smf.ols(f, data=dd).fit().tvalues["dp"])
    p = (1 + (np.array(null) >= t0).sum()) / (n_perm + 1); b, se = m_.params["dp"] * sc, m_.bse["dp"] * sc
    return f"{b:+7.2f}" if short else f"{b:+7.2f} [{b - 1.2816 * se:+7.2f}, {b + 1.2816 * se:+7.2f}] p1 {p:.3f}  n {len(used)}, DBs {used.nfl_id_d.nunique()}"


pd.set_option("display.width", 250)
print("=== A. movement vs catches by season (per 10 Drive+; open/completion/xComp in pts) ===")
rows = [("sep gained at key move (yd), all reps", "sep_gain_break", W, "wr_skill + C(zone) + C(route_ran) + is_s"), ("open at throw, all reps", "open", W, "C(zone) + C(route_ran) + is_s"),
        ("sep at throw (yd), all reps", "sep_at_throw", W, "wr_skill + C(zone) + C(route_ran) + is_s"), ("sep at throw (yd), TARGETS only", "sep_at_throw", T, "wr_skill + C(zone) + C(route_ran) + is_s"),
        ("expected completion from sep+context (targets)", "xc_sep", T, "C(zone) + C(route_ran) + is_s"), ("ACTUAL completion (targets)", "completion", T, "C(zone) + C(route_ran) + is_s"),
        ("after the throw: completion - xc_sep (targets)", "cpoe_sep", T, "C(zone) + C(route_ran) + is_s"), ("EPA per target", "epa", T, "C(zone) + C(route_ran) + is_s")]
for s in [2023, 2024, 2025]:
    print(f"  -- season {s}: reps {int((W.season == s).sum())}, targets {int((T.season == s).sum())}, DBs w/ targets {T[T.season == s].nfl_id_d.nunique()}")
    for nm, y, d, ctx in rows:
        print(f"     {nm:50s} {est(d[d.season == s], y, ctx)}", flush=True)

print("\n=== B. is 2025 different from 2024? dp x 2025 interaction (2024+2025 targets; grades shuffled across players) ===")
for y, sc in [("completion", 100), ("epa", 1), ("cpoe_sep", 100), ("xc_sep", 100)]:
    d = T[T.season.isin([2024, 2025])].dropna(subset=[y]).copy(); d["y25"] = (d.season == 2025).astype(int)
    f = f"{y} ~ dp * y25 + C(zone) + C(route_ran) + is_s"; mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]}); t0 = mod.fit().tvalues["dp:y25"]
    vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(500):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp:y25"])
    p2 = (1 + (np.abs(null) >= abs(t0)).sum()) / 501; b, se = m_.params["dp:y25"] * sc, m_.bse["dp:y25"] * sc
    print(f"  {y:10s} 2025 minus 2024 slope {b:+7.2f} [{b - 1.2816 * se:+7.2f}, {b + 1.2816 * se:+7.2f}]  clustered p {m_.pvalues['dp:y25']:.3f}  permutation p (two-sided) {p2:.3f}", flush=True)

print("\n=== C. by combine class (targets, all seasons pooled; within-class grade spread) ===")
for c in [2023, 2024, 2025]:
    d = T[T.cls == c]; sd = d.drop_duplicates("nfl_id_d").dp.std() * 10
    print(f"  class {c}: targets {len(d)}, DBs {d.nfl_id_d.nunique()}, Drive+ SD {sd:.1f} | completion {est(d, 'completion')}\n{'':16s}EPA {est(d, 'epa')}", flush=True)
print("  targets by class x season:\n" + pd.crosstab(T.cls, T.season).to_string())
print("  CBs: completion slope by class within each season:")
for s in [2024, 2025]:
    for c in [2023, 2024, 2025]:
        d = T[(T.season == s) & (T.cls == c)]
        if len(d) >= 30: print(f"    season {s}, class {c}: {est(d, 'completion')}", flush=True)

print("\n=== D. same DBs in both seasons (>= 3 targets in each) ===")
cnt = T.groupby(["nfl_id_d", "season"]).size().unstack(fill_value=0); both = cnt.index[(cnt.get(2024, 0) >= 3) & (cnt.get(2025, 0) >= 3)]
for s in [2024, 2025]:
    d = T[(T.season == s) & T.nfl_id_d.isin(both)]
    print(f"  season {s}, {len(both)} DBs: completion {est(d, 'completion')}\n{'':20s}EPA {est(d, 'epa')}", flush=True)
d = T[(T.season == 2025) & ~T.nfl_id_d.isin(both)]; print(f"  season 2025, DBs NOT in the panel: completion {est(d, 'completion')}", flush=True)

print("\n=== E. NFL experience (season - class + 1), targets ===")
for e in [1, 2, 3]:
    d = T[T.exp == e]; print(f"  year {e}: completion {est(d, 'completion')}\n{'':10s}EPA {est(d, 'epa')}", flush=True)

print("\n=== F. target mix by season ===")
mix = T.groupby("season").agg(targets=("completion", "size"), DBs=("nfl_id_d", "nunique"), comp=("completion", "mean"), epa=("epa", "mean"), air=("pass_length", "mean"),
                              zone=("zone", "mean"), zone_match=("cell", lambda c: (c == "zone-matching").mean()), go=("route_ran", lambda r: (r == "GO").mean()),
                              safeties=("is_s", "mean"), dp_sd=("dp", lambda v: v.std() * 10), wr_skill=("wr_skill", "mean"))
print(mix.round(3).to_string())

print("\n=== G. 2025 completion slope, leave one DB out ===")
d25 = T[T.season == 2025].dropna(subset=["completion"]); f = "completion ~ dp + C(zone) + C(route_ran) + is_s"; base = smf.ols(f, data=d25).fit().params["dp"] * 100
loo = {i: smf.ols(f, data=d25[d25.nfl_id_d != i]).fit().params["dp"] * 100 for i in d25.nfl_id_d.unique()}
s_ = pd.Series(loo); infl = (s_ - base).sort_values(ascending=False)
print(f"  full {base:+.2f}; leave-one-out range {s_.min():+.2f} to {s_.max():+.2f}")
for i in infl.index[:5]:
    dd = d25[d25.nfl_id_d == i]; print(f"    without {GR.display_name.get(i, i):24s} (Drive+ {GR.drive_plus.get(i, np.nan):5.1f}, {len(dd)} targets, comp {dd.completion.mean():.2f}): {s_[i]:+.2f}")
drop5 = d25[~d25.nfl_id_d.isin(infl.index[:5])]; print(f"  without the 5 most influential DBs: {smf.ols(f, data=drop5).fit().params['dp'] * 100:+.2f}")
