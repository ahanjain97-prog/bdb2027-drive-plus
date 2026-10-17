"""CPOE instead of CP for "completion when his man is targeted" on the registered wide reps (code/89).

Expected completion must not contain the coverage itself, or CPOE strips out what Drive+ is supposed to produce.
Three xComp versions, each predicted out of fold (5 folds by game) on all pool-receiver targets:
  ctx  throw context only (competition data): air yards (pass_length), receiver depth / speed / sideline distance at the
       throw, time to throw, pressure on the play, down, distance, field position, route, man/zone, and the passer's
       leave-one-out season completion %
  sep  ctx + NGS separation at the throw -> its CPOE is the after-throw part only (as in code/79)
  nv   nflverse cp (outside data; a check, not the registered measure)
Outcomes on targets of his man: CP (registered), CPOE for each xComp, and xComp_ctx itself (does a high-Drive+ DB draw
harder throws?). Same estimator and evidence standard as code/89: controls zone + route + position, one-sided
player-permutation p, 80% interval from DB-clustered SEs; per 10 Drive+ points, in percentage points.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; NV = "/Users/ahanjain/nfl_data/raw"
rng = np.random.default_rng(311)
import json
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")[["game_id", "play_id", "nfl_id", "idx", "season"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")


def at_throw(idx):
    pr = np.asarray(X[idx][:, :, F.index("post_pass")]); has = (pr >= .5).any(1); j = np.where(has, (pr >= .5).argmax(1), 0)
    r = np.arange(len(idx)); g = lambda n: np.asarray(X[idx][:, :, F.index(n)])[r, j]
    out = pd.DataFrame({"depth": g("x_rel"), "rspeed": g("s"), "sideline": 26.65 - np.abs(g("y_field")), "t_throw": (j - PRE) / 10})
    out.loc[~(has & V[idx, j]), :] = np.nan
    return out


# ---------------- all pool-receiver targets with throw context
tg = pp[(pp.target == 1) & (pp.team_abbr == pp.possession_team) & pp.pass_result.isin(["C", "I", "IN"])].merge(meta, on=["game_id", "play_id", "nfl_id"])
tg = pd.concat([tg.reset_index(drop=True), at_throw(tg.idx.values)], axis=1)
tg["comp"] = (tg.pass_result == "C").astype(int)
tg["ytg_goal"] = np.where(tg.yardline_side == tg.possession_team, 100 - tg.yardline_number, tg.yardline_number)
press = pp.groupby(["game_id", "play_id"]).pressure_allowed.max().fillna(0).astype(float).rename("press")
tg = tg.merge(press, on=["game_id", "play_id"], how="left")
qb = pp[pp.lined_up_position == "QB"].drop_duplicates(["game_id", "play_id"]).set_index(["game_id", "play_id"]).nfl_id.rename("qb")
tg = tg.merge(qb, on=["game_id", "play_id"], how="left")
q = tg.groupby(["qb", "season"]).comp.agg(["sum", "count"]); tg = tg.join(q, on=["qb", "season"])
tg["qb_loo"] = np.where(tg["count"] > 1, (tg["sum"] - tg.comp) / (tg["count"] - 1), np.nan)
tg["sep"] = tg.separation_at_pass_forward
CAT = ["route_ran", "team_coverage_man_zone"]
NUM_CTX = ["pass_length", "depth", "rspeed", "sideline", "t_throw", "press", "down", "yards_to_go", "ytg_goal", "qb_loo"]
games = tg.game_id.unique(); fold = dict(zip(games, rng.integers(0, 5, len(games)))); tg["fold"] = tg.game_id.map(fold)
ll = lambda y, p: -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))


def xcomp(num, name):
    def design(d):
        Xd = d[num].astype(float).copy()
        for c in CAT:
            Xd[c] = pd.Categorical(d[c].fillna("NA"), categories=sorted(tg[c].fillna("NA").unique())).codes
        return Xd
    tg[name] = np.nan
    for k in range(5):
        tr, te = tg[tg.fold != k], tg.fold == k
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=40,
                                           categorical_features=[len(num), len(num) + 1]).fit(design(tr), tr.comp)
        tg.loc[te, name] = m.predict_proba(design(tg[te]))[:, 1]
    ok = tg[name].notna()
    print(f"  xComp {name:8s}: out-of-fold log-loss {ll(tg.comp[ok], tg[name][ok]):.4f} vs base rate {ll(tg.comp[ok], np.full(ok.sum(), tg.comp[ok].mean())):.4f} (n {ok.sum()})")


print(f"pool-receiver targets: {len(tg)}")
xcomp(NUM_CTX, "xc_ctx"); xcomp(NUM_CTX + ["sep"], "xc_sep")

# nflverse cp (outside data, check only)
pbp = pd.concat([pd.read_parquet(f"{NV}/pbp/play_by_play_{s}.parquet", columns=["old_game_id", "play_id", "cp", "complete_pass"]) for s in [2023, 2024, 2025]], ignore_index=True)
pbp["game_id"] = pbp.old_game_id.astype(int); pbp = pbp.drop_duplicates(["game_id", "play_id"])
tg = tg.merge(pbp[["game_id", "play_id", "cp"]], on=["game_id", "play_id"], how="left")

# ---------------- the wide reps' targets
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
T = W[W.targeted == 1].merge(tg[["game_id", "play_id", "nfl_id", "xc_ctx", "xc_sep", "cp", "pass_length"]].rename(columns={"nfl_id": "nfl_id_r"}),
                              on=["game_id", "play_id", "nfl_id_r"], how="left")
T["cpoe_ctx"] = T.completion - T.xc_ctx; T["cpoe_sep"] = T.completion - T.xc_sep; T["cpoe_nv"] = T.completion - T.cp
print(f"\nwide targets {len(T)}: with xc_ctx {T.xc_ctx.notna().sum()}, xc_sep {T.xc_sep.notna().sum()}, nflverse cp {T.cp.notna().sum()} | "
      f"mean completion {T.completion.mean():.3f}, xc_ctx {T.xc_ctx.mean():.3f}, xc_sep {T.xc_sep.mean():.3f}, cp {T.cp.mean():.3f}")


def est(d, y, n_perm=500, ctx="C(zone) + C(route_ran) + is_s", sc=100):
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"
    mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]})
    t0 = -mod.fit().tvalues["dp"]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(-smf.ols(f, data=dd).fit().tvalues["dp"])
    p1 = (1 + (np.array(null) >= t0).sum()) / (n_perm + 1); b, se = m_.params["dp"] * sc, m_.bse["dp"] * sc
    lo, hi = b - 1.2816 * se, b + 1.2816 * se
    v = "supported" if (b < 0 and p1 < .10 and hi < 0) else ("right sign" if b < 0 else "wrong sign")
    return f"{b:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  110 vs 90 {2 * b:+6.1f}  p1 {p1:.3f}  {v:10s} n {len(used)}, DBs {used.nfl_id_d.nunique()}"


OUT = [("CP: completion (registered)", "completion"), ("xComp_ctx: how hard the throws were", "xc_ctx"),
       ("CPOE_ctx: vs throw context (competition data)", "cpoe_ctx"), ("CPOE_sep: vs context + separation (after-throw only)", "cpoe_sep"),
       ("CPOE_nv: vs nflverse cp (outside data)", "cpoe_nv")]
same = T.dropna(subset=["xc_ctx", "xc_sep", "cp"])
for lab, d in [("all wide targets (each outcome on its own available rows)", T), ("identical rows (all three xComps present)", same)] + \
              [(f"season {s} (identical rows)", same[same.season == s]) for s in sorted(same.season.unique())]:
    print(f"\n=== {lab} === per 10 Drive+, percentage points; one-sided p for the negative direction")
    for nm, y in OUT:
        print(f"  {nm:52s} {est(d, y)}", flush=True)
T.to_parquet(f"{ROOT}/data/outcomes/98_wide_targets_cpoe.parquet", index=False)
