"""(103) Control for receiver skill: separation skill (wr_skill) and catch skill (leave-one-out catches over expected).
Setup copied from 98.
CPOE instead of CP for "completion when his man is targeted" on the registered wide reps (code/89).

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



xcomp(NUM_CTX + ["sep"], "xc_sep")
tg["r"] = tg.comp - tg.xc_sep; agg = tg.groupby("nfl_id").r.agg(["sum", "count"])
tg["rcv_n"] = tg.nfl_id.map(agg["count"]) - 1; tg["catch_skill"] = np.where(tg.rcv_n > 0, (tg.nfl_id.map(agg["sum"]) - tg.r) / tg.rcv_n.clip(lower=1), np.nan)
K = 30; tg["catch_skill_s"] = np.where(tg.rcv_n > 0, (tg.nfl_id.map(agg["sum"]) - tg.r) / (tg.rcv_n + K), 0.0)   # shrunk toward 0 (K = 30 targets)
b_cs = smf.ols("r ~ catch_skill_s", data=tg).fit().params["catch_skill_s"]
print(f"pool targets {len(tg)} | receivers {tg.nfl_id.nunique()} | catch skill (shrunk) predicts this target's catch-over-expected: slope {b_cs:.2f}")
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
T = W[W.targeted == 1].merge(tg[["game_id", "play_id", "nfl_id", "xc_sep", "catch_skill_s", "rcv_n"]].rename(columns={"nfl_id": "nfl_id_r"}), on=["game_id", "play_id", "nfl_id_r"], how="left")
T["after"] = T.completion - T.xc_sep
T["ws_c"] = T.wr_skill - T.wr_skill.mean(); T["cs_c"] = T.catch_skill_s - T.catch_skill_s.mean()
q = pd.qcut(T.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp, 3, labels=["bottom", "middle", "top"]); T["third"] = T.nfl_id_d.map(q)
print(f"wide targets {len(T)}, receivers {T.nfl_id_r.nunique()} (with 2+ targets here: {(T.nfl_id_r.value_counts() >= 2).sum()})")


def est(d, y, extra="", n_perm=500):
    d = d.dropna(subset=[y, "dp", "ws_c", "cs_c"]).copy(); f = f"{y} ~ dp + C(zone) + C(route_ran) + is_s" + (f" + {extra}" if extra else ""); mod = smf.ols(f, data=d)
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]}); t0 = -mod.fit().tvalues["dp"]
    vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(-smf.ols(f, data=dd).fit().tvalues["dp"])
    p = (1 + (np.array(null) >= t0).sum()) / (n_perm + 1); b, se = 100 * m_.params["dp"], 100 * m_.bse["dp"]
    return f"{b:+6.2f} [{b - 1.2816 * se:+6.2f}, {b + 1.2816 * se:+6.2f}] p1 {p:.3f}"


print("\n=== balance: receiver skill faced vs Drive+ (per 10 Drive+) ===")
for sname, d in [("all", T), ("2024", T[T.season == 2024]), ("2025", T[T.season == 2025])]:
    out = []
    for y, sc, lab in [("ws_c", 1, "separation skill (yd)"), ("cs_c", 100, "catch skill (pts)")]:
        m = smf.ols(f"{y} ~ dp + C(zone) + C(route_ran) + is_s", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.dropna(subset=[y]).nfl_id_d)[0]})
        out.append(f"{lab} {sc * m.params['dp']:+.3f} (SE {sc * m.bse['dp']:.3f})")
    print(f"  {sname:5s} " + " | ".join(out))

SPECS = [("no receiver controls", ""), ("+ separation skill", "ws_c"), ("+ catch skill", "cs_c"), ("+ both", "ws_c + cs_c"), ("same receiver only (receiver fixed effects)", "C(nfl_id_r)")]
for y, lab in [("completion", "COMPLETION (CP)"), ("after", "AFTER THE THROW (completion - league xComp with separation)")]:
    print(f"\n=== {lab}, per 10 Drive+ (pts) ===")
    print(f"  {'':46s} {'all seasons':32s} {'2024':32s} {'2025':32s}")
    for nm, ex in SPECS:
        cells = [est(d, y, ex) for d in [T, T[T.season == 2024], T[T.season == 2025]]]
        print(f"  {nm:46s} " + " ".join(f"{c:32s}" for c in cells), flush=True)

print("\n=== catches above expected per 100 targets, expected now ALSO includes the receiver's catch skill ===")
T["exp_adj"] = T.xc_sep + b_cs * T.catch_skill_s
for sname in [2024, 2025]:
    d = T[T.season == sname]; g = d.groupby("third").agg(targets=("completion", "size"), caught=("completion", "sum"), exp=("xc_sep", "sum"), exp_adj=("exp_adj", "sum"))
    g["above_sep"] = 100 * (g.caught - g.exp) / g.targets; g["above_sep_and_receiver"] = 100 * (g.caught - g.exp_adj) / g.targets
    print(f"  -- {sname}\n" + g.round(1).to_string())
