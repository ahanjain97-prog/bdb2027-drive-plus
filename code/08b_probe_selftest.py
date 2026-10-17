"""Harness self-test: a block carrying real signal must be detected; pure-noise blocks must be null (p ~ U(0,1))."""
import sys
import numpy as np
sys.argv = ["x", "--blocks", "phys:resid", "--n_rep", "5"]
exec(open("/Users/ahanjain/bdb_ssl/code/08_probe.py").read().split("tests = []")[0])


def run_custom(grp, target, design, baseline, min_t, S_fn, n_perm=100, seed=0):
    df = assemble(grp, target, design, "phys:resid", min_t)
    B, _ = design_matrices(df, baseline)
    y = df.y.values.astype(float)
    S = S_fn(df, np.random.default_rng(seed))
    P = block_pcs(S, args.k)
    folds = make_folds(df.nfl_id.values, args.n_rep, seed)
    cv = CV(B, y, folds)
    obs = cv.r2_full(P) - cv.r2_base
    strata = (df["sub"] + "|" + df.stratum).values
    rs = np.random.default_rng(seed + 999); null = []
    for _ in range(n_perm):
        perm = np.arange(len(df))
        for s_ in np.unique(strata):
            ix = np.where(strata == s_)[0]; perm[ix] = rs.permutation(ix)
        null.append(cv.r2_full(P[perm]) - cv.r2_base)
    return len(df), cv.r2_base, obs, (1 + (np.array(null) >= obs).sum()) / (1 + n_perm)


pos = lambda df, rs: np.column_stack([df.prior.values, rs.normal(size=(len(df), 7))])
noise = lambda df, rs: rs.normal(size=(len(df), 192))
for grp, t, mt in [("REC", "sep", 30), ("OL", "ppa", 50), ("DL", "get_off", 40)]:
    n, rb, d, p = run_custom(grp, t, "half", "draft", mt, pos)
    print(f"POSITIVE {grp} {t}: n {n} base R2 {rb:.3f}  delta {d:+.3f}  p {p:.3f}")
ps = []
for seed in range(12):
    n, rb, d, p = run_custom("REC", "sep", "half", "draft", 30, noise, n_perm=50, seed=seed)
    ps.append(p); print(f"NOISE seed {seed}: delta {d:+.3f} p {p:.2f}")
print("noise p-values: mean %.2f, share<.05 %.2f" % (np.mean(ps), np.mean(np.array(ps) < .05)))
