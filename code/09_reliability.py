"""Are the signatures stable, individual, and novel? (No outcomes used.)

For each block x position group, players with >= MIN plays in both game halves (h0, h1):
  reliability     PCA fitted on h0 signatures; per-PC correlation between h0 and h1 projections across players
  fingerprint     top-1 accuracy of matching a player's h1 signature to his own h0 signature among all players
                  in the group (cosine in standardized space), vs chance 1/n; also within subgroup
  novelty (ssl)   share of each reliable SSL PC's variance NOT explained by the 16 hand-built physics features
                  + official measurables/tests (in-sample OLS R^2 -> 1 - R^2), on h0 signatures
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Accelerate BLAS emits spurious matmul warnings; outputs checked finite
import pandas as pd

ROOT = "/Users/ahanjain/bdb_ssl"
MIN, K = 40, 8
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
MEAS = ["combine_height", "combine_weight", "arm_length", "forty", "ten_yd_split", "vertical", "broad_jump"]


def halves(block, kind):
    s = pd.read_parquet(f"{ROOT}/data/sig/{block}.parquet")
    s = s[(s.kind == kind) & s.part.isin(["h0", "h1"])]
    cols = [c for c in s.columns if c.startswith("f")]
    a = s[s.part == "h0"].set_index("nfl_id"); b = s[s.part == "h1"].set_index("nfl_id")
    ids = a.index[(a.n >= MIN)].intersection(b.index[b.n >= MIN])
    return a.loc[ids, cols].values.astype(float), b.loc[ids, cols].values.astype(float), ids


def analyze(block, kind, grp, phys=None):
    A, Bm, ids = halves(block, kind)
    m = base.loc[ids, "grp"].values == grp
    A, Bm, ids = A[m], Bm[m], ids[m]
    n = len(ids)
    pooled = np.vstack([A, Bm]); mu, sd = pooled.mean(0), pooled.std(0); sd[sd < 1e-9] = 1
    Za, Zb = (A - mu) / sd, (Bm - mu) / sd
    k = min(K, Za.shape[1])
    U, sv, Vt = np.linalg.svd(Za - Za.mean(0), full_matrices=False)
    pa, pb = (Za - Za.mean(0)) @ Vt[:k].T, (Zb - Za.mean(0)) @ Vt[:k].T
    rel = np.array([np.corrcoef(pa[:, j], pb[:, j])[0, 1] for j in range(k)])
    na = Za / np.linalg.norm(Za, axis=1, keepdims=True); nb = Zb / np.linalg.norm(Zb, axis=1, keepdims=True)
    sim = nb @ na.T
    top1 = (sim.argmax(1) == np.arange(n)).mean()
    sub = base.loc[ids, "sub"].values
    within = []
    for i in range(n):
        same = np.where(sub == sub[i])[0]
        within.append(same[sim[i, same].argmax()] == i)
    rec = {"block": f"{block}:{kind}", "group": grp, "n": n, "rel_pc1": rel[0], "rel_mean_top8": rel.mean(),
           "n_pc_rel>.5": int((rel > .5).sum()), "fingerprint_top1": top1, "chance": 1 / n,
           "fingerprint_within_sub": np.mean(within), "chance_within_sub": np.mean([1 / (sub == s).sum() for s in sub])}
    if phys is not None:
        P = phys.loc[ids].values
        M = base.loc[ids, MEAS].values
        M = np.where(np.isnan(M), np.nanmean(M, 0), M)
        X = np.column_stack([np.ones(n), (P - P.mean(0)) / (P.std(0) + 1e-9), (M - M.mean(0)) / (M.std(0) + 1e-9)])
        nov = []
        for j in range(k):
            beta, *_ = np.linalg.lstsq(X, pa[:, j], rcond=None)
            r2 = 1 - ((pa[:, j] - X @ beta) ** 2).sum() / ((pa[:, j] - pa[:, j].mean()) ** 2).sum()
            nov.append(1 - r2)
        nov = np.array(nov)
        rec["novel_share_top8"] = nov.mean()
        rec["reliable_and_novel_pcs"] = int(((rel > .5) & (nov > .5)).sum())
        rec["pc_detail"] = "; ".join(f"PC{j+1} rel {rel[j]:.2f} novel {nov[j]:.2f}" for j in range(k))
    return rec


rows = []
ph = pd.read_parquet(f"{ROOT}/data/sig/phys.parquet")
ph = ph[(ph.kind == "resid") & (ph.part == "h0")].set_index("nfl_id")[[f"f{i}" for i in range(16)]]
for block, kind in [("ssl", "resid"), ("ssl", "raw"), ("sslasym", "resid"), ("phys", "resid"), ("phys", "raw"), ("surprise", "resid")]:
    for grp in ["REC", "OL", "DL", "DB"]:
        try:
            rows.append(analyze(block, kind, grp, phys=ph if block.startswith("ssl") else None))
        except FileNotFoundError:
            pass
R = pd.DataFrame(rows)
R.to_csv(f"{ROOT}/results/09_reliability.csv", index=False)
pd.set_option("display.width", 250)
print(R.drop(columns=[c for c in ["pc_detail"] if c in R]).round(3).to_string(index=False))
if "pc_detail" in R:
    for r in R[R.pc_detail.notna()].itertuples():
        print(f"\n{r.block} {r.group}: {r.pc_detail}")
