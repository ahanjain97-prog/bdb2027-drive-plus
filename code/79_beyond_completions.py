"""Beyond completions: score EVERY coverage rep, check which metrics are stable, and backtest small-sample shrinkage.

(1) Expected completion / expected EPA "if targeted" model, trained on ALL pool-receiver targets (player_play target = 1,
    ~9.8K throws) with features at the throw frame: NGS separation at pass forward (nearest of all 22), receiver depth
    beyond the LOS, receiver speed, distance to the nearest sideline, time from snap to throw, route, man/zone, down,
    distance, field position. Gradient boosting, cross-fitted by game (5 folds): every prediction comes from a model that
    never saw that game.
(2) Applied to every coverage rep (4-yd and 3-yd, all graded DBs) with the DB's OWN separation at the throw -> per rep:
    xcomp_rep = completion probability if the ball had gone to his man; xepa_rep = expected EPA if targeted.
    Also open rate = separation at throw >= 2 yd.
(3) Split-half reliability (odd vs even reps by date within player, context-adjusted): completion allowed and CPOE
    (targets only) vs the every-rep metrics.
(4) Drive+ slope (per 10 points) on each metric, player-permutation p.
(5) Small-sample backtest: predict each DB's 2025 completion % allowed (log-loss on his 2025 targets) from 2023-24 with:
    league mean | his raw 2023-24 rate | beta-binomial shrinkage to the league mean | shrinkage to a Drive+ prior |
    Drive+ prior alone | his 2023-24 mean xcomp_rep (every rep) | rookies: Drive+ prior vs league mean.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
src = open(f"{ROOT}/code/78_grade_card_all_db.py").read()
ns = {}; exec(src.split('pd.set_option("display.width", 250)')[0], ns)
build, GR, X, V, F, PRE = ns["build"], ns["GR"], ns["X"], ns["V"], ns["F"], ns["PRE"]
rng = np.random.default_rng(157)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")[["game_id", "play_id", "nfl_id", "idx"]]
T = X.shape[1]


def at_throw(idx):
    pr = np.asarray(X[idx][:, :, F.index("post_pass")]); has = (pr >= .5).any(1); j = np.where(has, (pr >= .5).argmax(1), 0)
    r = np.arange(len(idx)); g = lambda n: np.asarray(X[idx][:, :, F.index(n)])[r, j]
    out = pd.DataFrame({"depth": g("x_rel"), "rspeed": g("s"), "sideline": 26.65 - np.abs(g("y_field")), "t_throw": (j - PRE) / 10})
    out.loc[~(has & V[idx, j]), :] = np.nan
    return out


# ---------------- (1) train on all pool-receiver targets
tg = pp[(pp.target == 1) & (pp.team_abbr == pp.possession_team) & pp.pass_result.isin(["C", "I", "IN"])].merge(meta, on=["game_id", "play_id", "nfl_id"])
tg = pd.concat([tg.reset_index(drop=True), at_throw(tg.idx.values)], axis=1)
tg["ytg_goal"] = np.where(tg.yardline_side == tg.possession_team, 100 - tg.yardline_number, tg.yardline_number)
tg["sep"] = tg.separation_at_pass_forward; tg["comp"] = (tg.pass_result == "C").astype(int); tg["epa"] = tg.expected_points_added
tg = tg.dropna(subset=["sep", "depth", "rspeed"]).reset_index(drop=True)
CAT = ["route_ran", "team_coverage_man_zone"]
NUM = ["sep", "depth", "rspeed", "sideline", "t_throw", "down", "yards_to_go", "ytg_goal"]


def design(d):
    Xd = d[NUM].astype(float).copy()
    for c in CAT:
        Xd[c] = pd.Categorical(d[c].fillna("NA"), categories=sorted(tg[c].fillna("NA").unique())).codes
    return Xd


games = tg.game_id.unique(); fold_of = dict(zip(games, rng.integers(0, 5, len(games))))
tg["fold"] = tg.game_id.map(fold_of)
MC, MR = {}, {}
for k in range(5):
    tr = tg[tg.fold != k]
    MC[k] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, categorical_features=[NUM.__len__(), NUM.__len__() + 1]).fit(design(tr), tr.comp)
    MR[k] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, categorical_features=[NUM.__len__(), NUM.__len__() + 1]).fit(design(tr), tr.epa)
tg["xcomp"] = np.nan; tg["xepa"] = np.nan
for k in range(5):
    m = tg.fold == k; tg.loc[m, "xcomp"] = MC[k].predict_proba(design(tg[m]))[:, 1]; tg.loc[m, "xepa"] = MR[k].predict(design(tg[m]))
ll = lambda y, p: -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
print(f"xComp model: {len(tg)} targets | out-of-fold log-loss {ll(tg.comp, tg.xcomp):.4f} vs base rate {ll(tg.comp, np.full(len(tg), tg.comp.mean())):.4f} | "
      f"xEPA out-of-fold R2 {1 - ((tg.epa - tg.xepa) ** 2).mean() / tg.epa.var():.3f}")

# ---------------- (2) apply to every coverage rep (DB's own separation)
OUT = {}
for nm in ["coverage_reps", "coverage_reps_4yd"]:
    R = build(nm)
    R = R.merge(pp[["game_id", "play_id", "nfl_id", "down", "yards_to_go", "yardline_side", "yardline_number", "possession_team"]]
                .rename(columns={"nfl_id": "nfl_id_r"}), on=["game_id", "play_id", "nfl_id_r"], how="left", suffixes=("", "_pp"))
    R = pd.concat([R.reset_index(drop=True), at_throw(R.idx_r.values).drop(columns=["t_throw"]).reset_index(drop=True)], axis=1)
    R["t_throw"] = R.t_throw if "t_throw" in R.columns else np.nan
    R["ytg_goal"] = np.where(R.yardline_side == R.possession_team, 100 - R.yardline_number, R.yardline_number)
    R["sep"] = R.sep_at_throw
    R["fold"] = R.game_id.map(fold_of).fillna(0).astype(int)
    R["down"] = R["down"].fillna(R.get("down_pp")); R["yards_to_go"] = R["yards_to_go"].fillna(R.get("yards_to_go_pp"))
    ok = R[["sep", "depth", "rspeed"]].notna().all(1)
    R["xcomp_rep"] = np.nan; R["xepa_rep"] = np.nan
    for k in range(5):
        m = ok & (R.fold == k)
        if m.any():
            R.loc[m, "xcomp_rep"] = MC[k].predict_proba(design(R[m]))[:, 1]; R.loc[m, "xepa_rep"] = MR[k].predict(design(R[m]))
    R["open"] = np.where(R.sep.notna(), (R.sep >= 2).astype(float), np.nan)
    R["cpoe_t"] = np.where(R.targeted == 1, R.completion - R.xcomp_rep, np.nan)
    OUT[nm] = R
    R.to_parquet(f"{ROOT}/data/outcomes/79_scored_{nm}.parquet", index=False)
    t = R[R.targeted == 1].dropna(subset=["xcomp_rep", "completion"])
    print(f"\n[{nm}] reps {len(R)}, reps scored (a throw happened) {int(ok.sum())}, targets {len(t)}, DBs {R.nfl_id_d.nunique()} | "
          f"calibration on these targets: mean xcomp {t.xcomp_rep.mean():.3f} vs actual {t.completion.mean():.3f}; log-loss {ll(t.completion, t.xcomp_rep):.4f} vs {ll(t.completion, np.full(len(t), t.completion.mean())):.4f}")

# ---------------- (3) reliability and (4) Drive+ slopes
CL = lambda d: pd.factorize(d.nfl_id_d)[0]
METRICS = [("completion allowed (targets)", "completion", "targets"), ("CPOE (targets)", "cpoe_t", "targets"),
           ("xComp allowed per rep", "xcomp_rep", "reps"), ("xEPA allowed per rep", "xepa_rep", "reps"),
           ("separation at throw", "sep", "reps"), ("open rate (sep >= 2 yd)", "open", "reps"), ("zone-match closing", "db_closing", "zm")]
pd.set_option("display.width", 250)
for nm, R in OUT.items():
    R = R.sort_values(["nfl_id_d", "game_id", "play_id"]).reset_index(drop=True)
    rows = []
    for lab, col, unit in METRICS:
        d = R[R.targeted == 1] if unit == "targets" else (R[R.cell == "zone-matching"] if unit == "zm" else R)
        d = d.dropna(subset=[col]).copy()
        d["res"] = smf.ols(f"{col} ~ C(team_coverage_man_zone) + C(route_ran) + is_s", data=d).fit().resid
        d["half"] = d.groupby("nfl_id_d").cumcount() % 2
        h = d.groupby(["nfl_id_d", "half"]).res.agg(["mean", "size"]).unstack()
        h = h[(h["size"].min(1) >= 5)]
        r = np.corrcoef(h["mean"][0], h["mean"][1])[0, 1] if len(h) >= 8 else np.nan
        f = f"{col} ~ dp + C(team_coverage_man_zone) + C(route_ran) + is_s"
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
        t0 = smf.ols(f, data=d).fit().tvalues["dp"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
        for _ in range(400):
            dd = d.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"])
        rows.append({"metric": lab, "observations": len(d), "DBs": d.nfl_id_d.nunique(), "obs per DB (median)": d.groupby("nfl_id_d").size().median(),
                     "split-half r (DBs with >=5 per half)": r, "DBs in split-half": len(h),
                     "Drive+ slope per 10 pts": m.params["dp"], "perm p": (1 + (np.abs(null) >= abs(t0)).sum()) / 401,
                     "slope / metric SD": m.params["dp"] / d[col].std()})
    Tb = pd.DataFrame(rows).set_index("metric")
    print(f"\n=== [{nm}] which metric is a stable DB trait, and what Drive+ does to it ===")
    print(Tb.round(3).to_string())
    Tb.to_csv(f"{ROOT}/results/79_metrics_{nm}.csv")

# ---------------- (5) small-sample backtest: predict 2025 completion % allowed
R = OUT["coverage_reps_4yd"]
t = R[R.targeted == 1].dropna(subset=["completion"])
tr_, te_ = t[t.season <= 2024], t[t.season == 2025]
league = tr_.completion.mean()
agg = tr_.groupby("nfl_id_d").completion.agg(["sum", "count"])
xrep = R[R.season <= 2024].groupby("nfl_id_d").xcomp_rep.mean()


def eb_fit(k, n, prior_mean):
    def nll(log_s):
        s = np.exp(log_s); a, b = prior_mean * s, (1 - prior_mean) * s
        return -(betaln(k + a, n - k + b) - betaln(a, b)).sum()
    return np.exp(minimize_scalar(nll, bounds=(-2, 8), method="bounded").x)


lg = smf.logit("completion ~ dp", data=tr_).fit(disp=0)
prior_dp = lambda ids: lg.predict(pd.DataFrame({"dp": pd.Series(ids).map(GR.drive_plus.sub(100).div(10)).values})).values
ids_tr = agg.index.values
s0 = eb_fit(agg["sum"].values, agg["count"].values, np.full(len(agg), league))
s1 = eb_fit(agg["sum"].values, agg["count"].values, prior_dp(ids_tr))
print(f"\n=== (5) predicting 2025 completion % allowed (log-loss on 2025 targets; lower = better) ===")
print(f"  train targets 2023-24: {len(tr_)} ({len(agg)} DBs); 2025 targets {len(te_)} ({te_.nfl_id_d.nunique()} DBs); fitted prior strength: league {s0:.1f} targets, Drive+ {s1:.1f}")
ret = te_[te_.nfl_id_d.isin(ids_tr)]; new = te_[~te_.nfl_id_d.isin(ids_tr)]
def predict(df, how):
    ids = df.nfl_id_d.values; k = df.nfl_id_d.map(agg["sum"]).fillna(0).values; n = df.nfl_id_d.map(agg["count"]).fillna(0).values
    if how == "league mean": return np.full(len(df), league)
    if how == "his raw 2023-24 rate": return np.clip(np.where(n > 0, k / np.maximum(n, 1), league), .02, .98)
    if how == "shrink to league mean": return (k + s0 * league) / (n + s0)
    if how == "shrink to Drive+ prior": pm = prior_dp(ids); return (k + s1 * pm) / (n + s1)
    if how == "Drive+ prior alone": return prior_dp(ids)
    if how == "his 2023-24 xComp per rep (calibrated)":
        x = df.nfl_id_d.map(xrep).values
        return np.where(np.isnan(x), league, league + (x - np.nanmean(xrep.reindex(ids_tr))))
HOW = ["league mean", "his raw 2023-24 rate", "shrink to league mean", "shrink to Drive+ prior", "Drive+ prior alone", "his 2023-24 xComp per rep (calibrated)"]
for lab, df in [("returning DBs", ret), ("2025 rookies (no NFL history)", new), ("all 2025 targets", te_)]:
    if len(df) < 20:
        continue
    base_ll = ll(df.completion.values, predict(df, "league mean"))
    print(f"  {lab} ({len(df)} targets, {df.nfl_id_d.nunique()} DBs): " + " | ".join(
        f"{h}: {100 * (base_ll - ll(df.completion.values, predict(df, h))) / base_ll:+.1f}%" for h in HOW))
