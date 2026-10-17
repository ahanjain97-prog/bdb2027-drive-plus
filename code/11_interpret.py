"""Describe the learned signature dimensions in football terms.

Per position group: PCA of context-adjusted SSL signatures (all of a player's plays, n >= MIN). For each PC:
  - split-half reliability (from 09_reliability on h0/h1)
  - correlation with each hand-built physics feature and each official measurable/test
  - variance NOT explained by physics + measurables (novelty)
  - the 4 highest and lowest players
Also: per-play view -- for each PC, mean speed / along-track accel / lateral accel / body-vs-motion angle
profiles over time for the top vs bottom decile of plays (projected per-play residual), same context mix.
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Accelerate BLAS spurious matmul warnings
import pandas as pd
from sig_utils import load_meta, loo_context_means

ROOT = "/Users/ahanjain/bdb_ssl"
MIN, K = 80, 6
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
MEAS = ["combine_height", "combine_weight", "arm_length", "forty", "ten_yd_split", "vertical", "broad_jump",
        "three_cone", "short_shuttle", "log_pick"]
phys_names = json.load(open(f"{ROOT}/data/blocks/phys_names.json"))

meta = load_meta()
E = np.load(f"{ROOT}/data/blocks/ssl_emb.npy").astype(np.float32)
mu, _ = loo_context_means(E, meta)
R = E - mu
Pp = np.load(f"{ROOT}/data/blocks/phys_play.npy")
Pp = np.where(np.isnan(Pp), np.nanmedian(Pp, 0), Pp)
pmu, _ = loo_context_means(Pp, meta)
PR = Pp - pmu
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r")
voc = json.load(open(f"{ROOT}/data/seq/vocab.json"))
F = voc["feats"]
meta["grp"] = meta.nfl_id.map(base.grp)

out_lines = []
say = lambda s="": (print(s), out_lines.append(s))
rel_tab = pd.read_csv(f"{ROOT}/results/09_reliability.csv")
for grp in ["REC", "OL", "DL", "DB"]:
    m = (meta.grp == grp).values
    df = pd.DataFrame(R[m]); df["nfl_id"] = meta.nfl_id.values[m]
    g = df.groupby("nfl_id"); S = g.mean(); n = g.size(); S = S[n >= MIN]
    ids = S.index
    Z = (S.values - S.values.mean(0)) / (S.values.std(0) + 1e-9)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :K] * sv[:K]
    evr = sv[:K] ** 2 / (sv ** 2).sum()
    dfp = pd.DataFrame(PR[m], columns=phys_names); dfp["nfl_id"] = meta.nfl_id.values[m]
    PS = dfp.groupby("nfl_id").mean().loc[ids]
    M = base.loc[ids, MEAS]
    say(f"\n################ {grp}  (players {len(ids)}, plays/player >= {MIN})")
    detail = rel_tab[(rel_tab.block == "ssl:resid") & (rel_tab.group == grp)]
    if len(detail):
        say(f"split-half (h0 vs h1) PC reliability: {detail.pc_detail.iat[0]}")
    # per-play projection onto each PC direction (same standardization)
    Rm = (R[m] - S.values.mean(0)) / (S.values.std(0) + 1e-9)
    proj = Rm @ Vt[:K].T
    idx_m = np.where(m)[0]
    for j in range(K):
        pc = pd.Series(pcs[:, j], index=ids)
        cp = PS.apply(lambda c: np.corrcoef(c, pc)[0, 1]).sort_values(key=np.abs, ascending=False)
        cm = M.apply(lambda c: pd.Series(c).corr(pc)).sort_values(key=np.abs, ascending=False)
        Xd = np.column_stack([np.ones(len(ids)), (PS - PS.mean()) / (PS.std() + 1e-9),
                              M.fillna(M.mean()).sub(M.mean()).div(M.std() + 1e-9)])
        beta, *_ = np.linalg.lstsq(Xd, pc.values, rcond=None)
        nov = ((pc.values - Xd @ beta) ** 2).sum() / ((pc.values - pc.mean()) ** 2).sum()   # 1 - R^2
        say(f"\n-- {grp} PC{j+1}: {evr[j]*100:.1f}% var; unexplained by physics+measurables {nov:.2f}")
        say("   physics r: " + ", ".join(f"{k} {v:+.2f}" for k, v in cp.head(5).items()))
        say("   measurables r: " + ", ".join(f"{k} {v:+.2f}" for k, v in cm.head(4).items()))
        top = pc.sort_values(); nm = base.display_name
        say("   high: " + ", ".join(f"{nm[i]} ({base.at[i,'sub']})" for i in top.index[-4:][::-1]))
        say("   low:  " + ", ".join(f"{nm[i]} ({base.at[i,'sub']})" for i in top.index[:4]))
        # per-play: top vs bottom decile of plays on this direction (post-snap 0-4 s profiles)
        pj = proj[:, j]
        hi, lo = idx_m[pj >= np.quantile(pj, .9)], idx_m[pj <= np.quantile(pj, .1)]
        rs = np.random.default_rng(0)
        hi, lo = np.sort(rs.choice(hi, min(3000, len(hi)), replace=False)), np.sort(rs.choice(lo, min(3000, len(lo)), replace=False))
        def prof(ix, ch):
            a = np.asarray(X[ix][:, 10:51, F.index(ch)], dtype=np.float32)
            a = np.abs(a) if ch == "a_lat" else a       # lateral sign = turn direction; magnitude is the trait
            return a[:, ::10].mean(0)                   # at 0,1,2,3,4 s after snap
        for ch in ("s", "a_long", "a_lat", "cos_od"):
            say(f"   {('|a_lat|' if ch == 'a_lat' else ch):7s} @0..4s  top10% {np.round(prof(hi, ch), 2)}  bottom10% {np.round(prof(lo, ch), 2)}")
open(f"{ROOT}/results/11_interpret.txt", "w").write("\n".join(out_lines))
