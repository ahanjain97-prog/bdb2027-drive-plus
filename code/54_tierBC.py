"""Phase 3, tiers B and C.

Tier B -- selection-aware open screen, per unit x outcome (>= 20 players):
  for each held-out draft class: on the other two classes, rank every reliable tracking feature by |r| with the
  outcome (>= 12 players), keep the top K = 3; predict held-out players with the mean of sign(r) * z(feature)
  (train mean / sd; missing -> 0). Pool predictions over the three folds -> OOS r. Null: permute the outcome within
  draft class and rerun the whole selection (200 perms). Same for the official stopwatch block, for comparison.
  Also: which features were picked in >= 2 folds, with their held-out-fold correlations.
Tier C -- does tracking add to size + stopwatch? Leave-one-class-out ridge (alpha by exact LOO inside training),
  OOS R^2 for: size; size + stopwatch; size + stopwatch + tracking (tracking block reduced to 6 PCs fitted on training).
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
rng = np.random.default_rng(0)
K, NPERM = 3, 200
ALPHAS = np.logspace(-2, 4, 25)


def oos_select(Fz_raw, y, cls, K):
    """Fz_raw: DataFrame players x features (raw); returns pooled OOS predictions and selected features per fold."""
    pred = pd.Series(np.nan, index=y.index); picks = []
    for c in np.unique(cls):
        tr, te = (cls != c), (cls == c)
        Ftr, ytr = Fz_raw[tr], y[tr]
        rs = {}
        for f in Ftr.columns:
            m = Ftr[f].notna()
            if m.sum() >= 12 and Ftr.loc[m, f].std() > 0:
                rs[f] = np.corrcoef(Ftr.loc[m, f], ytr[m])[0, 1]
        if not rs:
            continue
        top = sorted(rs, key=lambda f: -abs(rs[f]))[:K]
        picks.append({f: rs[f] for f in top})
        z = []
        for f in top:
            mu, sd = Ftr[f].mean(), Ftr[f].std()
            z.append(np.sign(rs[f]) * ((Fz_raw.loc[te, f] - mu) / sd).fillna(0))
        pred[te] = np.mean(z, axis=0)
    return pred, picks


def ridge_loo(Xtr, ytr, Xte):
    xm, ym = Xtr.mean(0), ytr.mean()
    Xc, yc = Xtr - xm, ytr - ym
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uty = U.T @ yc
    Fm = sv[None] ** 2 / (sv[None] ** 2 + ALPHAS[:, None])
    fitted = U @ (Fm * Uty[None]).T
    h = (U ** 2) @ Fm.T + 1.0 / len(ytr)
    a = np.argmin((((yc[:, None] - fitted) / np.clip(1 - h, 1e-6, None)) ** 2).mean(0))
    return (Xte - xm) @ (Vt.T @ ((sv / (sv ** 2 + ALPHAS[a])) * Uty)) + ym


def prep(Mtr, Mte):
    mu = np.nanmean(Mtr, 0); mu = np.where(np.isnan(mu), 0, mu)
    Mtr = np.where(np.isnan(Mtr), mu, Mtr); Mte = np.where(np.isnan(Mte), mu, Mte)
    sd = Mtr.std(0); sd[sd < 1e-9] = 1
    return (Mtr - mu) / sd, (Mte - mu) / sd


def tierC(S, W, T_, y, cls):
    out = {}
    for name, blocks in [("size", [S]), ("size+stopwatch", [S, W]), ("size+stopwatch+tracking", [S, W, T_])]:
        pred = np.full(len(y), np.nan)
        for c in np.unique(cls):
            tr, te = cls != c, cls == c
            parts_tr, parts_te = [], []
            for B in blocks:
                Btr, Bte = prep(B[tr], B[te])
                if B is T_:
                    U, sv, Vt = np.linalg.svd(Btr - Btr.mean(0), full_matrices=False)
                    k = min(6, Btr.shape[1], tr.sum() - 2)
                    Btr, Bte = (Btr - Btr.mean(0)) @ Vt[:k].T, (Bte - Btr.mean(0)) @ Vt[:k].T
                parts_tr.append(Btr); parts_te.append(Bte)
            pred[te] = ridge_loo(np.hstack(parts_tr), y[tr], np.hstack(parts_te))
        out[name] = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return out


rowsB, rowsC, pickrows = [], [], []
for unit in ["WR", "TE", "CB", "S", "EDGE", "IDL", "T", "IOL"]:
    F = pd.read_parquet(f"{ROOT}/data/features/catalog_{unit}.parquet")
    M = pd.read_csv(f"{ROOT}/data/features/catalog_{unit}_meta.csv", index_col=0)
    O = pd.read_parquet(f"{ROOT}/data/outcomes/outcomes_{unit}.parquet")
    trk = [c for c in F.columns if M.loc[c, "kind"] in ("family_composite", "drill_feature", "frozen_metric")]
    stw = [c for c in F.columns if c.startswith("stopwatch:")]
    siz = [c for c in F.columns if c.startswith("size:")]
    for outc in O.columns:
        y = O[outc].dropna()
        y = y[y.index.isin(F.index)]
        if len(y) < 20:
            continue
        cls = base.loc[y.index, "draft_year"].values
        res = {}
        for blk_name, cols in [("tracking", trk), ("stopwatch", stw)]:
            Fb = F.loc[y.index, cols]
            pred, picks = oos_select(Fb, y, cls, K)
            ok = pred.notna()
            r_obs = np.corrcoef(pred[ok], y[ok])[0, 1] if ok.sum() >= 10 else np.nan
            null = []
            for _ in range(NPERM):
                yp = y.copy()
                for c in np.unique(cls):
                    ix = np.where(cls == c)[0]; yp.iloc[ix] = rng.permutation(yp.iloc[ix].values)
                pp_, _ = oos_select(Fb, yp, cls, K)
                okp = pp_.notna()
                null.append(np.corrcoef(pp_[okp], yp[okp])[0, 1] if okp.sum() >= 10 else 0)
            p = (1 + (np.array(null) >= r_obs).sum()) / (NPERM + 1)
            res[blk_name] = (r_obs, p)
            if blk_name == "tracking":
                cnt = pd.Series([f for pk in picks for f in pk]).value_counts()
                for f, nsel in cnt[cnt >= 2].items():
                    # held-out-fold r for this feature in each fold where it was picked
                    tr_r = []
                    for c in np.unique(cls):
                        te = cls == c
                        m = F.loc[y.index[te], f].notna()
                        if m.sum() >= 5:
                            tr_r.append(np.corrcoef(F.loc[y.index[te], f][m], y[te][m])[0, 1])
                    pickrows.append({"unit": unit, "outcome": outc, "feature": f, "folds_selected": nsel,
                                     "train_r_signs": "".join("+" if pk.get(f, 0) > 0 else "-" for pk in picks if f in pk),
                                     "heldout_class_r": ", ".join(f"{v:+.2f}" for v in tr_r)})
        rowsB.append({"unit": unit, "outcome": outc, "n": len(y), "oos_r_tracking": res["tracking"][0], "p_tracking": res["tracking"][1],
                      "oos_r_stopwatch": res["stopwatch"][0], "p_stopwatch": res["stopwatch"][1]})
        # tier C
        S_ = F.loc[y.index, siz].values.astype(float); W_ = F.loc[y.index, stw].values.astype(float); T_ = F.loc[y.index, trk].values.astype(float)
        rowsC.append({"unit": unit, "outcome": outc, "n": len(y), **{f"R2 {k}": v for k, v in tierC(S_, W_, T_, y.values, cls).items()}})
    print(f"{unit} done", flush=True)
B = pd.DataFrame(rowsB); C = pd.DataFrame(rowsC); Pk = pd.DataFrame(pickrows)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


B["q_tracking"] = bh(B.p_tracking); B["q_stopwatch"] = bh(B.p_stopwatch)
B.to_csv(f"{ROOT}/results/54_tierB.csv", index=False); C.to_csv(f"{ROOT}/results/54_tierC.csv", index=False); Pk.to_csv(f"{ROOT}/results/54_tierB_picks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 200)
print("\n=== Tier B: selection-aware held-out-class OOS r (top-3 features chosen inside training classes) ===")
print(B.round(3).to_string(index=False))
print(f"\ntracking p<.05: {(B.p_tracking < .05).sum()} of {len(B)} (expected {.05 * len(B):.1f}); q<.10: {(B.q_tracking < .10).sum()} | stopwatch p<.05: {(B.p_stopwatch < .05).sum()}")
print("\n=== features picked in >= 2 of 3 training folds (with their held-out-class r) ===")
print(Pk.sort_values(["unit", "outcome"]).to_string(index=False))
print("\n=== Tier C: OOS R^2 (leave-one-class-out ridge) ===")
C["tracking_adds"] = C["R2 size+stopwatch+tracking"] - C["R2 size+stopwatch"]
print(C.round(3).to_string(index=False))
print(f"\ntracking adds > 0 in {(C.tracking_adds > 0).sum()} of {len(C)}; > .05 in {(C.tracking_adds > .05).sum()}")
