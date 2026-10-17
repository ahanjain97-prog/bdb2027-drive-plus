"""Summarize probe results: Benjamini-Hochberg over the pre-declared primary family, then secondary tables."""
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/ahanjain/bdb_ssl"
name = sys.argv[1] if len(sys.argv) > 1 else "probe_results"
R = pd.read_csv(f"{ROOT}/results/{name}.csv")
R = R[R.delta_r2.notna()].copy()


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# primary family: learned embedding (context-adjusted) on held-out-games and next-season designs, all baselines
prim = R.block.eq("ssl:resid") & R.design.isin(["half", "forward"])
R.loc[prim, "q_bh_primary"] = bh(R.loc[prim, "p_perm"])
R["q_bh_all"] = bh(R.p_perm)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
cols = ["group", "target", "design", "baseline", "block", "n_players", "r2_base", "r2_full", "delta_r2", "p_perm"]
print("=== PRIMARY (ssl:resid, half + forward) ===")
print(R[prim][cols + ["q_bh_primary"]].sort_values("p_perm").round(3).to_string(index=False))
print("\n=== half-split replication (h1 -> h0) for the primary block ===")
print(R[R.block.eq("ssl:resid") & R.design.eq("half_rev")][cols].round(3).to_string(index=False))
print("\n=== block comparison: mean delta R^2 and #p<.05 by block x baseline (half + forward) ===")
s = R[R.design.isin(["half", "forward"])].groupby(["block", "baseline"]).agg(
    tests=("p_perm", "size"), mean_delta=("delta_r2", "mean"), n_p05=("p_perm", lambda p: int((p < .05).sum())),
    expected_p05=("p_perm", lambda p: round(.05 * len(p), 1)))
print(s.round(4).to_string())
print("\n=== career design ===")
print(R[R.design.eq("career")][cols].sort_values(["group", "target", "block"]).round(3).to_string(index=False))
print("\n=== anything with BH q < .10 across ALL tests ===")
print(R[R.q_bh_all < .10][cols + ["q_bh_all"]].round(3).to_string(index=False) or "none")
R.to_csv(f"{ROOT}/results/{name}_with_q.csv", index=False)
