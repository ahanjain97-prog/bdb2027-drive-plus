"""Robustness of the one primary hit: OL pressure_allowed, ssl:resid beyond draft + prior + physics (half design).
Checks: both directions, leave-one-player-out influence, more CV repeats, which PCs carry it, and whether
the permutation p reflects real gain (delta > 0) or just 'better than a random block' (null mean < 0)."""
import sys
import numpy as np
sys.argv = ["x", "--blocks", "ssl:resid", "--n_rep", "20"]
exec(open("/Users/ahanjain/bdb_ssl/code/08_probe.py").read().split("tests = []")[0])

for target, mt in [("pressure_allowed", 50), ("ppa", 50)]:
    for design in ["half", "half_rev"]:
        df = assemble("OL", target, design, "ssl:resid", mt)
        y = df.y.values.astype(float)
        for bl in ["prior", "prior+phys"]:
            B, S = design_matrices(df, bl)
            P = block_pcs(S, args.k)
            folds = make_folds(df.nfl_id.values, 20, 0)
            cv = CV(B, y, folds)
            d = cv.r2_full(P) - cv.r2_base
            # leave-one-player-out influence on delta (5 reps each)
            infl = []
            for i in range(len(df)):
                keep = np.arange(len(df)) != i
                f2 = make_folds(df.nfl_id.values[keep], 5, 1)
                c2 = CV(B[keep], y[keep], f2)
                infl.append(c2.r2_full(P[keep]) - c2.r2_base)
            infl = np.array(infl)
            print(f"{target:16s} {design:8s} {bl:10s} n {len(df)}  delta(20 reps) {d:+.3f}  "
                  f"LOO-player delta min {infl.min():+.3f} median {np.median(infl):+.3f}  share>0 {np.mean(infl > 0):.2f}")
        # which PCs: partial correlation of each PC with the target residual after the prior+phys baseline (in-sample)
        B, S = design_matrices(df, "prior+phys")
        P = block_pcs(S, args.k)
        Bf = np.where(np.isnan(B), np.nanmedian(B, 0), B)
        Bf = (Bf - Bf.mean(0)) / (Bf.std(0) + 1e-9)
        res = y - ridge_loo(Bf, y, Bf)
        pc_r = [np.corrcoef(P[:, j], res)[0, 1] for j in range(P.shape[1])]
        print("    PC r with residual (prior+phys):", np.round(pc_r, 2))
