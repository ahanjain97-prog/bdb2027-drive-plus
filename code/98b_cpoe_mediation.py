"""(98b) Is the harder-throw shift caused by his coverage? Copied setup from 98.
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



print(f"pool-receiver targets: {len(tg)}")
NUM_SIT = ["down", "yards_to_go", "ytg_goal", "qb_loo", "quarter"]
xcomp(NUM_SIT, "xc_sit"); xcomp(NUM_CTX, "xc_ctx")
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other")
T = W[W.targeted == 1].merge(tg[["game_id", "play_id", "nfl_id", "xc_sit", "xc_ctx"] + NUM_CTX].rename(columns={"nfl_id": "nfl_id_r"}),
                              on=["game_id", "play_id", "nfl_id_r"], how="left")
T["cpoe_sit"] = T.completion - T.xc_sit; T["cpoe_ctx"] = T.completion - T.xc_ctx
T["throw_shift"] = T.xc_ctx - T.xc_sit      # how much harder the throw was than the situation alone predicts


def est(d, y, sc=100, extra="", n_perm=500, two=False, ctx="C(zone) + C(route_ran) + is_s"):
    d = d.dropna(subset=[y, "dp"] + [c.strip() for c in extra.split("+") if c.strip()]).copy(); f = f"{y} ~ dp + {ctx}" + (f" + {extra}" if extra else "")
    mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    if used.nfl_id_d.nunique() < 8: return "too few"
    m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]})
    t0 = mod.fit().tvalues["dp"]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"])
    null = np.array(null); p = (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1) if two else (1 + (-null >= -t0).sum()) / (n_perm + 1)
    b, se = m_.params["dp"] * sc, m_.bse["dp"] * sc
    return f"{b:+7.2f} [{b - 1.2816 * se:+7.2f}, {b + 1.2816 * se:+7.2f}]  {'two' if two else 'one'}-sided p {p:.3f}  n {len(used)}, DBs {used.nfl_id_d.nunique()}"


print(f"\nwide targets {len(T)} | mean completion {T.completion.mean():.3f}, xc_sit {T.xc_sit.mean():.3f}, xc_ctx {T.xc_ctx.mean():.3f}")
print("\n=== 1. outcome measures, per 10 Drive+ (pts) ===")
for nm, y in [("CP (registered)", "completion"), ("CPOE vs SITUATION only (pre-snap: down, dist, field, quarter, QB, route, coverage)", "cpoe_sit"),
              ("CPOE vs throw context (situation + throw choice)", "cpoe_ctx"), ("throw made harder than the situation predicts", "throw_shift")]:
    print(f"  {nm:88s} {est(T, y)}", flush=True)
print("\n=== 2. which parts of the throw move with Drive+ (per 10, raw units; two-sided) ===")
for nm, y, sc in [("air yards (pass_length)", "pass_length", 1), ("receiver depth at the throw (yd)", "depth", 1), ("time to throw (s)", "t_throw", 1),
                  ("receiver distance to sideline at throw (yd)", "sideline", 1), ("receiver speed at throw (yd/s)", "rspeed", 1), ("pressure on the play (pts)", "press", 100)]:
    print(f"  {nm:46s} {est(T, y, sc=sc, two=True)}", flush=True)
print("\n=== 3. mediation: does holding his coverage fixed remove the harder-throw shift? (throw_shift, pts per 10) ===")
for nm, ex in [("no coverage controls", ""), ("+ separation gained at the key move", "sep_gain_break"), ("+ separation at the throw", "sep_at_throw"),
               ("+ both", "sep_gain_break + sep_at_throw")]:
    print(f"  {nm:40s} {est(T, 'throw_shift', extra=ex)}", flush=True)
print("\n=== 4. where does the harder-throw shift live? (throw_shift, pts per 10) — Drive+ only changes coverage in zone-match ===")
for c in ["zone-matching", "man-matching", "other"]:
    print(f"  {c:16s} {est(T[T.cell == c], 'throw_shift', ctx='C(route_ran) + is_s')}", flush=True)
print("\n=== 5. CPOE vs situation by season (pts per 10) ===")
for sname in sorted(T.season.unique()):
    print(f"  {sname}  CP {est(T[T.season == sname], 'completion')}\n        CPOE_sit {est(T[T.season == sname], 'cpoe_sit')}", flush=True)
T.to_parquet(f"{ROOT}/data/outcomes/98b_wide_targets_cpoe_sit.parquet", index=False)
