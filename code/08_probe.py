"""Probe harness: does a feature block predict NFL outcomes beyond what was known at the draft?

Designs (signature data never overlaps the games the target is measured on):
  half     signature from one random half of games (h0) -> target on the other half (h1)
  half_rev the same with halves swapped (replication, not independent of `half`)
  forward  signature from season t -> target in season t+1 (latest eligible pair: 2024->2025, else 2023->2024)
Exactly ONE row per player in every design: stacking a player's two rows lets near-identical signatures leak
the sibling row's outcome through the leave-one-out alpha selection (seen in debugging: delta R^2 down to -0.95).
  career   signature from the rookie season -> career outcomes per season
Baselines:
  draft    position subgroup + draft pick/UDFA + age + conference + measurables + official combine tests +
           NGS scores + log(#signature plays)  [volume control]
  prior    draft + the same target measured on the signature data (does movement add to track record?)
  +phys    the above plus the 16 hand-built physics features (context-adjusted): does the learned
           embedding add anything a simple physics summary does not?
Model: block -> label-free PCA(k) over the test's rows; in-fold impute/standardize; ridge on [baseline, c*PCs] with\n(alpha, c) chosen by exact LOO; c = 0 nests the baseline. Repeated 5-fold CV.
Stat: delta OOS R^2 (full - baseline), paired on identical folds. Null: permute block rows within
subgroup x stratum, refit everything; p = (1 + #null >= obs) / (1 + n_perm). Players weighted equally.

Usage: python 08_probe.py [--blocks ssl:resid phys:resid ...] [--n_perm 200]
"""
import argparse, json, time
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Accelerate BLAS emits spurious matmul warnings; outputs checked finite
import pandas as pd
from joblib import Parallel, delayed

ROOT = "/Users/ahanjain/bdb_ssl"
ap = argparse.ArgumentParser()
ap.add_argument("--blocks", nargs="+", default=["ssl:resid", "ssl:raw", "phys:resid", "phys:raw", "surprise:resid",
                                                "combine_mr:-"])
ap.add_argument("--designs", nargs="+", default=["half", "half_rev", "forward", "career"])
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--n_rep", type=int, default=10)
ap.add_argument("--n_perm", type=int, default=200)
ap.add_argument("--min_sig", type=int, default=40)
ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--out", default="probe_results")
args = ap.parse_args()

TARGETS = {"REC": [("sep", 30), ("tprr", 50), ("yprr", 50)],
           "OL": [("ppa", 50), ("pressure_allowed", 50)],
           "DL": [("pressure_rate", 40), ("get_off", 40)],
           "DB": [("tackle_rate", 100)]}
CAREER = ["log_snaps_per_season", "starts_per_season", "honors"]
BASE = ["log_pick", "udfa", "age_at_draft", "power_conf", "combine_height", "combine_weight", "hand_size",
        "arm_length", "wing_span", "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle",
        "bench_reps", "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
PRIMARY_BLOCK = "ssl:resid"

base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
tg = pd.read_parquet(f"{ROOT}/data/targets.parquet")
TG = {(r.nfl_id, r.part, r.target): (r.value, r.n) for r in tg.itertuples()}
SIG = {}


def load_block(name, kind):
    key = (name, kind)
    if key not in SIG:
        if name == "combine_mr":
            df = pd.read_parquet(f"{ROOT}/data/sig/combine_mr.parquet").set_index("nfl_id")
            SIG[key] = ("static", df)
        else:
            df = pd.read_parquet(f"{ROOT}/data/sig/{name}.parquet")
            df = df[df.kind == kind]
            cols = [c for c in df.columns if c.startswith("f")]
            SIG[key] = ("part", {(r[0], r[1]): (np.asarray(r[3:], dtype=np.float32), r[2])
                                 for r in df[["nfl_id", "part", "n"] + cols].itertuples(index=False)})
    return SIG[key]


def sig_vec(block_spec, pid, part, grp_cols=None):
    """Concatenate one or more blocks ('a:k|b:k'). Returns (vector or None, n_plays)."""
    vecs, n = [], None
    for spec in block_spec.split("|"):
        name, kind = spec.split(":")
        typ, store = load_block(name, kind)
        if typ == "static":
            if pid not in store.index:
                return None, 0
            row = store.loc[pid]
            vecs.append(row[grp_cols].values.astype(np.float32) if grp_cols is not None else row.values.astype(np.float32))
        else:
            v = store.get((pid, part))
            if v is None:
                return None, 0
            vecs.append(v[0]); n = v[1]
    return np.concatenate(vecs), n


def combine_cols(grp_players):
    """Combine drills done by >= 50% of the group's players at the combine (any subgroup)."""
    df = load_block("combine_mr", "-")[1]
    df = df.loc[df.index.intersection(grp_players)]
    drills = sorted({c.split("__")[0] for c in df.columns})
    keep = []
    for d in drills:
        cols = [c for c in df.columns if c.startswith(d + "__")]
        cov = df[cols[0]].notna().groupby(base.loc[df.index, "sub"]).mean()
        if cov.max() >= 0.5:
            keep += cols
    return keep


def assemble(grp, target, design, block, min_t):
    players = base.index[base.grp == grp]
    gcols = combine_cols(players) if block.startswith("combine_mr") or "combine_mr" in block else None
    rows = []
    pairs = {"half": [("h0", "h1")], "half_rev": [("h1", "h0")], "forward": [("2024", "2025"), ("2023", "2024")],
             "career": [("rookie", "career")]}[design]
    for pid in players:
        for a, b in pairs:
            yv = TG.get((pid, b, target))
            if yv is None or not np.isfinite(yv[0]) or yv[1] < (min_t if design != "career" else 1):
                continue
            # volume + eligibility are defined by the movement signature (SSL block), identical across blocks
            s_ref = load_block("phys", "raw")[1].get((pid, a))
            if s_ref is None or s_ref[1] < args.min_sig:
                continue
            prior = TG.get((pid, a, target)) if design != "career" else None
            if design != "career" and (prior is None or not np.isfinite(prior[0]) or prior[1] < min_t // 2):
                continue
            sv, _ = sig_vec(block, pid, a, gcols)
            if sv is None:
                continue
            pv, _ = sig_vec("phys:resid", pid, a)
            rows.append({"nfl_id": pid, "stratum": a, "sub": base.at[pid, "sub"], "y": yv[0], "phys": pv,
                         "prior": prior[0] if prior else np.nan, "log_prior_n": np.log(prior[1]) if prior else np.nan,
                         "log_sig_n": np.log(s_ref[1]), "sig": sv})
            break                                   # one row per player (first eligible pair)
    return pd.DataFrame(rows)


def design_matrices(df, baseline):
    subs = sorted(df["sub"].unique())
    B = pd.DataFrame(index=df.index)
    for s in subs[1:]:
        B[f"sub_{s}"] = (df["sub"] == s).astype(float)
    if df.stratum.nunique() > 1:
        B["stratum"] = pd.factorize(df.stratum)[0].astype(float)
    bb = base.loc[df.nfl_id, BASE].reset_index(drop=True)
    bb.index = df.index
    for c in BASE:
        B[c] = bb[c]
        if bb[c].isna().any():
            B[c + "_miss"] = bb[c].isna().astype(float)
    B["log_sig_n"] = df.log_sig_n
    if baseline.startswith("prior"):
        B["prior"] = df.prior; B["log_prior_n"] = df.log_prior_n
    if baseline.endswith("+phys"):
        P = np.stack(df.phys.values)
        for j in range(P.shape[1]):
            B[f"phys{j}"] = P[:, j]
    S = np.stack(df.sig.values)
    return B.values.astype(np.float64), S.astype(np.float64)


ALPHAS = np.logspace(-1, 4, 26)


C_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def ridge_loo(Xtr, ytr, Xte, return_err=False):
    """Ridge with intercept; alpha chosen by exact leave-one-out error (vectorized over ALPHAS via one SVD)."""
    xm, ym = Xtr.mean(0), ytr.mean()
    Xc, yc = Xtr - xm, ytr - ym
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uty = U.T @ yc
    F = sv[None] ** 2 / (sv[None] ** 2 + ALPHAS[:, None])          # [A, r]
    fitted = U @ (F * Uty[None]).T                                 # [n, A]
    h = (U ** 2) @ F.T + 1.0 / len(ytr)                            # hat diagonal incl. intercept
    loo = (yc[:, None] - fitted) / np.clip(1 - h, 1e-6, None)
    err = (loo ** 2).mean(0)
    a = np.argmin(err)
    coef = Vt.T @ ((sv / (sv ** 2 + ALPHAS[a])) * Uty)
    pred = (Xte - xm) @ coef + ym
    return (pred, err[a]) if return_err else pred


def ridge_two_penalty(Btr, Ptr, ytr, Bte, Pte):
    """Block columns get their own scale c (penalty alpha / c^2); (c, alpha) chosen jointly by LOO.
    c = 0 is the baseline-only model, so a useless block can be switched off instead of adding variance."""
    best = None
    for c in C_GRID:
        pred, err = ridge_loo(np.hstack([Btr, c * Ptr]), ytr, np.hstack([Bte, c * Pte]), return_err=True)
        if best is None or err < best[1] - 1e-12:
            best = (pred, err)
    return best[0]


def _std(tr, te, M, impute):
    Mtr, Mte = M[tr], M[te]
    fill = impute(Mtr)
    fill = np.where(np.isnan(fill), 0, fill)
    Mtr = np.where(np.isnan(Mtr), fill, Mtr); Mte = np.where(np.isnan(Mte), fill, Mte)
    mu, sd = Mtr.mean(0), Mtr.std(0); sd[sd < 1e-9] = 1.0
    return (Mtr - mu) / sd, (Mte - mu) / sd


def block_pcs(S, k):
    """Label-free PCA of the block over all rows of this test (outcome-blind, so permutation-invariant)."""
    fill = np.nanmean(S, 0); fill = np.where(np.isnan(fill), 0, fill)
    Z = np.where(np.isnan(S), fill, S)
    mu, sd = Z.mean(0), Z.std(0); sd[sd < 1e-9] = 1.0
    Z = (Z - mu) / sd
    if Z.shape[1] <= k:
        return Z
    U, sv, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :k] * sv[:k]


class CV:
    """Repeated K-fold; baseline design prepared and fitted once per fold, block columns swapped per call."""
    def __init__(self, B, y, folds):
        self.y, self.folds = y, folds
        self.sst = ((y - y.mean()) ** 2).sum()
        self.prep = [[(_std(tr, te, B, lambda m: np.nanmedian(m, 0)), tr, te) for tr, te in f] for f in folds]
        r2 = []
        for f in self.prep:
            p = np.zeros_like(y)
            for (Btr, Bte), tr, te in f:
                p[te] = ridge_loo(Btr, y[tr], Bte)
            r2.append(1 - ((y - p) ** 2).sum() / self.sst)
        self.r2_base = float(np.mean(r2))

    def r2_full(self, P):
        y, r2 = self.y, []
        for f in self.prep:
            p = np.zeros_like(y)
            for (Btr, Bte), tr, te in f:
                Ptr, Pte = _std(tr, te, P, lambda m: np.nanmean(m, 0))
                p[te] = ridge_two_penalty(Btr, Ptr, y[tr], Bte, Pte)
            r2.append(1 - ((y - p) ** 2).sum() / self.sst)
        return float(np.mean(r2))


def make_folds(groups, n_rep, seed):
    ug = np.unique(groups)
    out = []
    for r in range(n_rep):
        rs = np.random.default_rng(seed + r)
        lab = dict(zip(ug, rs.permutation(len(ug)) % 5))
        f = np.array([lab[g] for g in groups])
        out.append([(np.where(f != i)[0], np.where(f == i)[0]) for i in range(5)])
    return out


def run_test(grp, target, design, baseline, block, min_t, seed=0):
    t0 = time.time()
    df = assemble(grp, target, design, block, min_t)
    rec = {"group": grp, "target": target, "design": design, "baseline": baseline, "block": block,
           "n_rows": len(df), "n_players": df.nfl_id.nunique() if len(df) else 0}
    if len(df) < 30 or df.nfl_id.nunique() < 25:
        return {**rec, "note": "too few"}
    B, S = design_matrices(df, baseline)
    y = df.y.values.astype(np.float64)
    groups = df.nfl_id.values
    folds = make_folds(groups, args.n_rep, seed)
    P = block_pcs(S, args.k)
    cv = CV(B, y, folds)
    r2b, r2f = cv.r2_base, cv.r2_full(P)
    obs = r2f - r2b
    strata = (df["sub"] + "|" + df.stratum).values
    rs = np.random.default_rng(seed + 999)
    null = []
    for _ in range(args.n_perm):
        perm = np.arange(len(df))
        for s in np.unique(strata):
            ix = np.where(strata == s)[0]
            perm[ix] = rs.permutation(ix)
        null.append(cv.r2_full(P[perm]) - r2b)
        # adaptive stop: clearly null after 50 perms
        if len(null) == 50 and np.mean(np.array(null) >= obs) > 0.3:
            break
    null = np.array(null)
    return {**rec, "r2_base": r2b, "r2_full": r2f, "delta_r2": obs, "p_perm": (1 + (null >= obs).sum()) / (1 + len(null)),
            "null_mean": null.mean(), "null_sd": null.std(), "n_perm": len(null), "sec": time.time() - t0}


tests = []
for grp, tl in TARGETS.items():
    for design in args.designs:
        if design == "career":
            for t in CAREER:
                for block in args.blocks:
                    tests.append((grp, t, "career", "draft", block, 1))
                if PRIMARY_BLOCK in args.blocks:
                    tests.append((grp, t, "career", "draft+phys", PRIMARY_BLOCK, 1))
            continue
        tl2 = tl + ([("snaps_per_game", 4)] if design == "forward" else [])
        for t, mn in tl2:
            for baseline in ("draft", "prior"):
                for block in args.blocks:
                    tests.append((grp, t, design, baseline, block, mn))
            if PRIMARY_BLOCK in args.blocks:
                tests.append((grp, t, design, "prior+phys", PRIMARY_BLOCK, mn))
print(f"{len(tests)} tests", flush=True)
t0 = time.time()
res = Parallel(n_jobs=args.jobs, verbose=0)(delayed(run_test)(*t) for t in tests)
R = pd.DataFrame(res)
R.to_csv(f"{ROOT}/results/{args.out}.csv", index=False)
print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
