"""Follow-up to 77: offenses throw DEEPER to a high-Drive+ CB's side. Is the completion drop just the depth shift?

(1) Completion and EPA toward-away contrasts with air-yard bins added to the fixed design (depth held constant).
(2) Separately for short throws (air < 10 yd) and longer throws (air >= 10 yd): completion and EPA contrasts.
(3) Share of throws to his side that are short (< 10 yd air) -- are short throws being taken away?
Zone coverage (the registered setting) and all coverage. Player-permutation p, same model as 76b/77.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
src = open(f"{ROOT}/code/77_epa_decomposition.py").read()
ns = {}; exec(src.split("OUT = [")[0], ns)
d, fit = ns["d"], ns["fit"]
rng = np.random.default_rng(149)
d["air_bin"] = pd.cut(d.air, [-99, 0, 5, 10, 15, 20, 99]).astype(str)
d["short"] = (d.air < 10).astype(float)


class FWL2:
    def __init__(self, P, extra):
        cols = [pd.get_dummies(P[c].astype(str), prefix=c, drop_first=True, dtype=float)
                for c in ["down", "dist_bin", "ytg_bin", "quarter", "def_ts", "off_ts", "rel", "dir"] + extra]
        M = np.hstack([np.ones((len(P), 1))] + [c.values for c in cols])
        q, r = np.linalg.qr(M); self.Q = q[:, np.abs(np.diag(r)) > 1e-8]
    def res(self, v):
        return v - self.Q @ (self.Q.T @ v)


def contrast(P, y, extra=(), n_perm=500):
    P = P.dropna(subset=[y, "air"]).reset_index(drop=True); fw = FWL2(P, list(extra)); y_r = fw.res(P[y].values.astype(float))
    R = {r: (P.rel == r).values.astype(float) for r in ["toward", "away", "middle"]}
    def st(dp):
        b, C = fit(fw, y_r, [dp * R["toward"], dp * R["away"], dp * R["middle"]])
        return b, b[0] - b[1], (b[0] - b[1]) / np.sqrt(C[0, 0] + C[1, 1] - 2 * C[0, 1])
    b, c, t = st(P.dp.values); vals = P.drop_duplicates("nfl_id").set_index("nfl_id").dp; nt = []
    for _ in range(n_perm):
        sp = pd.Series(rng.permutation(vals.values), index=vals.index); nt.append(st(P.nfl_id.map(sp).values)[2])
    return b[0], b[1], c, (1 + (np.abs(nt) >= abs(t)).sum()) / (n_perm + 1), len(P)


fmt = lambda r: f"toward {r[0]:+.4f} | away {r[1]:+.4f} | toward-away {r[2]:+.4f} (perm p {r[3]:.3f}, rows {r[4]})"
for lab, P in [("zone", d[d.team_coverage_man_zone == "ZONE_COVERAGE"]), ("all", d)]:
    print(f"\n=== {lab} coverage (per 10 Drive+ points) ===")
    print("  completion, no depth control      ", fmt(contrast(P, "completion")))
    print("  completion, air-yard bins held     ", fmt(contrast(P, "completion", ["air_bin"])))
    print("  EPA, air-yard bins held            ", fmt(contrast(P, "epa", ["air_bin"])))
    print("  share of throws that are short     ", fmt(contrast(P, "short")))
    for sl, sub in [("short throws (< 10 yd air)", P[P.air < 10]), ("throws >= 10 yd air", P[P.air >= 10])]:
        print(f"  {sl:28s} completion: " + fmt(contrast(sub, "completion")))
        print(f"  {sl:28s} EPA:        " + fmt(contrast(sub, "epa")))
