"""Full unshrunk CB combine data -> NFL coverage. Which model uses the combine best, out of sample?

Rep-level outcomes (CB coverage reps vs pool receivers, REG+POST): separation gained at the break, separation at the
throw, closing after the break, completion over expected (targets); plus zone-only separation.
Per training fold: context OLS (receiver skill, route, man/zone) -> residuals -> each training CB's mean residual (his
'coverage effect', weight = reps). CB-effect models:
  drive      weighted LS on the frozen drive speed (1 feature)
  ridge_all  ridge on ALL 1,246 features (attempt views as rows, mean-imputed, standardized); alpha by grouped CV
             (folds = CBs) inside the training fold only
  enet_all   elastic net on all features, (alpha, l1_ratio) by the same grouped CV
  xgb_all    gradient boosting on the rep residuals with all features (depth 2, 300 trees, eta .03, colsample .3,
             min_child_weight 40 reps), NaN-native
Test prediction = context + predicted CB effect. Score: OOS MSE reduction vs context only (%), and the weighted
correlation between predicted and realized test-CB effects. Splits: leave-one-draft-class-out, season-forward
(train <= 2024 seasons, test 2025), new prospects (train 2023-24 classes, test 2025 class).
Null: permute the CB -> combine-data mapping (all features + drive together), rerun everything.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(29)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_value.parquet")
Vw = pd.read_parquet(f"{ROOT}/data/features/cb_full_views.parquet")
Pm = pd.read_parquet(f"{ROOT}/data/features/cb_full_player.parquet")
FE = [c for c in Vw.columns if c not in ("nfl_id", "view")]
drive = Pm["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION|trans_drive_top"]
CBS = sorted(set(cov.nfl_id_d) & set(Pm.index))
cov = cov[cov.nfl_id_d.isin(CBS)].copy()
ALPHAS = np.logspace(-1, 4, 16)


def wridge_fit(X, y, w, alpha):
    sw = np.sqrt(w)[:, None]; xm = np.average(X, axis=0, weights=w); ym = np.average(y, weights=w)
    Xc, yc = (X - xm) * sw, (y - ym) * sw[:, 0]
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    coef = Vt.T @ ((s / (s ** 2 + alpha)) * (U.T @ yc))
    return coef, xm, ym


def grouped_folds(groups, k=5):
    ug = np.unique(groups); rng_ = np.random.default_rng(0); lab = dict(zip(ug, rng_.permutation(len(ug)) % k))
    f = np.array([lab[g] for g in groups]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def prep_views(ids, mu=None, sd=None):
    V = Vw[Vw.nfl_id.isin(ids)].sort_values(["nfl_id", "view"])
    X = V[FE].values.astype(float)
    if mu is None:
        mu = np.nanmean(X, 0); mu = np.where(np.isnan(mu), 0, mu)
        Xf = np.where(np.isnan(X), mu, X); sd = Xf.std(0); sd[sd < 1e-9] = 1
    X = (np.where(np.isnan(X), mu, X) - mu) / sd
    return V.nfl_id.values, X, mu, sd


def cb_models(tr_eff, te_ids, which):
    """tr_eff: DataFrame index=CB, cols ybar, n. Returns dict model -> Series of predicted effect for te_ids."""
    out = {}
    ids = tr_eff.index.values
    if "drive" in which:
        d = pd.concat([drive.rename("x"), tr_eff], axis=1).loc[ids].dropna()
        X = np.column_stack([np.ones(len(d)), d.x]); W = d.n.values
        b = np.linalg.lstsq(X * np.sqrt(W)[:, None], d.ybar.values * np.sqrt(W), rcond=None)[0]
        out["drive"] = pd.Series(b[0] + b[1] * drive.reindex(te_ids).fillna(drive.reindex(ids).mean()).values, index=te_ids)
    if "ridge_all" in which or "enet_all" in which:
        vid, X, mu, sd = prep_views(ids)
        y = tr_eff.ybar.reindex(vid).values; nv = pd.Series(vid).map(pd.Series(vid).value_counts()).values
        w = tr_eff.n.reindex(vid).values / nv
        tid, Xt, _, _ = prep_views(te_ids, mu, sd)
        if "ridge_all" in which:
            folds = grouped_folds(vid); err = []
            for a in ALPHAS:
                e = 0.0
                for tri, tei in folds:
                    c, xm, ym = wridge_fit(X[tri], y[tri], w[tri], a)
                    e += np.sum(w[tei] * ((X[tei] - xm) @ c + ym - y[tei]) ** 2)
                err.append(e)
            a = ALPHAS[int(np.argmin(err))]
            c, xm, ym = wridge_fit(X, y, w, a)
            out["ridge_all"] = pd.Series((Xt - xm) @ c + ym, index=tid).groupby(level=0).mean().reindex(te_ids)
            out["_ridge_coef"] = pd.Series(c, index=FE); out["_ridge_alpha"] = a
        if "enet_all" in which:
            folds = grouped_folds(vid); best = (np.inf, None)
            for l1 in (0.2, 0.5, 0.9):
                for a in np.logspace(-3, 0, 8):
                    e = 0.0
                    for tri, tei in folds:
                        m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(X[tri], y[tri], sample_weight=w[tri])
                        e += np.sum(w[tei] * (m.predict(X[tei]) - y[tei]) ** 2)
                    if e < best[0]:
                        best = (e, (a, l1))
            m = ElasticNet(alpha=best[1][0], l1_ratio=best[1][1], max_iter=5000).fit(X, y, sample_weight=w)
            out["enet_all"] = pd.Series(m.predict(Xt), index=tid).groupby(level=0).mean().reindex(te_ids)
            out["_enet_nonzero"] = int((m.coef_ != 0).sum())
    return out


def run_split(d, y, ctx, splits, which, perm_map=None):
    """Returns {model: (mse_reduction_%, weighted r)}."""
    global drive, Vw, Pm
    sse = {m: 0.0 for m in ["ctx"] + which}; pr_rows = []
    for tr_mask, te_mask in splits:
        tr, te = d[tr_mask].copy(), d[te_mask].copy()
        if "route_ran" in ctx:
            te = te[te.route_ran.isin(tr.route_ran.unique())]
        if len(te) < 20 or tr.nfl_id_d.nunique() < 10:
            continue
        cm = smf.ols(f"{y} ~ {ctx}", data=tr).fit()
        tr["res"] = cm.resid; te["base"] = cm.predict(te); te["res"] = te[y] - te.base
        eff = tr.groupby("nfl_id_d").res.agg(["mean", "size"]).rename(columns={"mean": "ybar", "size": "n"})
        te_ids = te.nfl_id_d.unique()
        preds = cb_models(eff, te_ids, [m for m in which if m != "xgb_all"])
        if "xgb_all" in which:
            Xtr = Pm.reindex(tr.nfl_id_d).values; Xte = Pm.reindex(te.nfl_id_d).values
            xm = XGBRegressor(max_depth=2, n_estimators=300, learning_rate=0.03, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=40, reg_lambda=1.0, n_jobs=4, verbosity=0).fit(Xtr, tr.res.values)
            preds["xgb_all"] = pd.Series(xm.predict(Xte), index=te.index).groupby(te.nfl_id_d).mean()
        sse["ctx"] += np.sum(te.res ** 2)
        for m in which:
            p = te.nfl_id_d.map(preds[m]).fillna(0).values
            sse[m] += np.sum((te.res.values - p) ** 2)
        realized = te.groupby("nfl_id_d").res.agg(["mean", "size"])
        for m in which:
            pr_rows.append(pd.DataFrame({"model": m, "pred": preds[m].reindex(realized.index).values, "real": realized["mean"].values, "n": realized["size"].values}))
    P = pd.concat(pr_rows)
    res = {}
    for m in which:
        q = P[P.model == m].dropna()
        wr = np.cov(q.pred, q.real, aweights=q.n)[0, 1] / np.sqrt(np.cov(q.pred, aweights=q.n) * np.cov(q.real, aweights=q.n)) if len(q) > 3 and q.pred.std() > 0 else np.nan
        res[m] = ((1 - sse[m] / sse["ctx"]) * 100, wr)
    return res


OUTS = [("sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)", None),
        ("sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran)", None),
        ("db_closing", "wr_skill + C(team_coverage_man_zone) + C(route_ran)", None),
        ("cpoe", "rec_target_prop + C(team_coverage_man_zone)", None),
        ("sep_gain_break", "wr_skill + C(route_ran)", "ZONE_COVERAGE"),
        ("sep_at_throw", "wr_skill + C(route_ran)", "ZONE_COVERAGE")]
SPLITS = {"draft-class-out": lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())],
          "season-forward": lambda d: [((d.season <= 2024).values, (d.season == 2025).values)],
          "new-prospect": lambda d: [((d.cls <= 2024).values, (d.cls == 2025).values)]}
MODELS = ["drive", "ridge_all", "enet_all", "xgb_all"]
N_PERM = {"drive": 100, "ridge_all": 100, "enet_all": 15, "xgb_all": 25}
rows = []
orig = (drive.copy(), Vw.copy(), Pm.copy())
for y, ctx, sub in OUTS:
    d = cov if sub is None else cov[cov.team_coverage_man_zone == sub]
    d = d.dropna(subset=[y] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    for sname, sp in SPLITS.items():
        obs = run_split(d, y, ctx, sp(d), MODELS)
        # permutation: remap CB ids -> another CB's combine data (all features + drive together)
        nulls = {m: [] for m in MODELS}
        for i in range(max(N_PERM.values())):
            perm = dict(zip(CBS, rng.permutation(CBS)))
            drive = orig[0].rename(index=lambda k: perm.get(k, k))
            Vw = orig[1].assign(nfl_id=orig[1].nfl_id.map(lambda k: perm.get(k, k)))
            Pm = orig[2].rename(index=lambda k: perm.get(k, k))
            todo = [m for m in MODELS if i < N_PERM[m]]
            r_ = run_split(d, y, ctx, sp(d), todo)
            for m in todo:
                nulls[m].append(r_[m][0])
        drive, Vw, Pm = orig[0].copy(), orig[1].copy(), orig[2].copy()
        for m in MODELS:
            nl = np.array(nulls[m])
            rows.append({"outcome": y + ("" if sub is None else " (zone)"), "split": sname, "model": m, "reps": len(d), "CBs": d.nfl_id_d.nunique(),
                         "oos_mse_reduction_%": obs[m][0], "cb_effect_r": obs[m][1], "perm_p": (1 + (nl >= obs[m][0]).sum()) / (1 + len(nl)),
                         "null_95pct": np.percentile(nl, 95), "n_perm": len(nl)})
        print(f"{y}{'' if sub is None else ' (zone)'} | {sname}: " + " | ".join(f"{m} {obs[m][0]:+.2f}% (r {obs[m][1]:+.2f})" for m in MODELS), flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/67_cb_full_models.csv", index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 200)
print("\n" + R.round(3).to_string(index=False))
