"""Why does throw-side EPA not move when throw-side completion does? (76b: zone completion toward a high-Drive+ CB's side
-2.05 pp per 10 points vs away; EPA contrast 0.000)

Same rows and model as 76b (its build code is executed unmodified -- 76b is hashed in the prereg). For each component,
the Drive+ slope (per 10 points) on throws toward his side, away from it, and the toward-away contrast with a
player-permutation p and the permutation SD of the contrast (= its noise level):
  epa, completion, interception, defensive pass interference (DPI), air yards, deep share, explosive (>= 20 yds),
  yards per completion, YAC proxy (yards gained - air yards, completions), EPA per completion, EPA per incompletion.
Plus: the EPA change implied by the completion change alone, and the minimum detectable EPA contrast (80% power).
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
src = open(f"{ROOT}/code/76b_toward_away.py").read()
ns = {}; exec(src.split("rows = []")[0], ns)
d, FWL, fit = ns["d"], ns["FWL"], ns["fit"]
rng = np.random.default_rng(139)
pp = ns["pp"]
desc = pp.drop_duplicates(["game_id", "play_id"]).set_index(["game_id", "play_id"])
key = pd.MultiIndex.from_arrays([d.game_id, d.play_id])
d["desc"] = desc.play_description.reindex(key).values
d["air"] = desc.pass_length.reindex(key).values
d["yds"] = desc.yards_gained.reindex(key).values
d["interception"] = (d.pass_result == "IN").astype(float)
d["dpi"] = d.desc.str.contains("Defensive Pass Interference", case=False).astype(float)
d["deep"] = d.desc.str.contains(r"pass (?:incomplete )?deep", case=False).astype(float)
d["explosive"] = (d.yds >= 20).astype(float)
C = d.pass_result == "C"
d["yds_if_c"] = np.where(C, d.yds, np.nan); d["yac_if_c"] = np.where(C, d.yds - d.air, np.nan)
d["epa_if_c"] = np.where(C, d.epa, np.nan); d["epa_if_i"] = np.where(d.pass_result == "I", d.epa, np.nan)
print("EPA by result on these throws:", d.groupby("pass_result").epa.mean().round(3).to_dict(), "| DPI rate", round(d.dpi.mean(), 4))


def analyze(P, y, n_perm=500):
    P = P.dropna(subset=[y]).reset_index(drop=True); fw = FWL(P); y_r = fw.res(P[y].values.astype(float))
    R = {r: (P.rel == r).values.astype(float) for r in ["toward", "away", "middle"]}
    def stats(dp):
        b, Cv = fit(fw, y_r, [dp * R["toward"], dp * R["away"], dp * R["middle"]])
        return b, b[0] - b[1], (b[0] - b[1]) / np.sqrt(Cv[0, 0] + Cv[1, 1] - 2 * Cv[0, 1])
    b, con, t = stats(P.dp.values)
    vals = P.drop_duplicates("nfl_id").set_index("nfl_id").dp; nc, nt = [], []
    for _ in range(n_perm):
        sp = pd.Series(rng.permutation(vals.values), index=vals.index); _, c_, t_ = stats(P.nfl_id.map(sp).values); nc.append(c_); nt.append(t_)
    return {"toward": b[0], "away": b[1], "toward-away": con, "perm p": (1 + (np.abs(nt) >= abs(t)).sum()) / (n_perm + 1), "noise SD": np.std(nc), "rows": len(P)}


OUT = ["epa", "completion", "interception", "dpi", "air", "deep", "explosive", "yds_if_c", "yac_if_c", "epa_if_c", "epa_if_i"]
pd.set_option("display.width", 240)
for lab, P in [("zone", d[d.team_coverage_man_zone == "ZONE_COVERAGE"]), ("all", d)]:
    rows = []
    for y in OUT:
        r = analyze(P, y); r["outcome"] = y; rows.append(r)
    R = pd.DataFrame(rows).set_index("outcome")
    R.to_csv(f"{ROOT}/results/77_epa_decomposition_{lab}.csv")
    print(f"\n=== {lab} coverage: Drive+ slope per 10 points (offense perspective) ===")
    print(R.round(4).to_string())
    Pt = P[P.rel == "toward"]
    gap = Pt[Pt.pass_result == "C"].epa.mean() - Pt[Pt.pass_result == "I"].epa.mean()
    implied = R.loc["completion", "toward-away"] * gap
    mde = 2.8 * R.loc["epa", "noise SD"]
    print(f"  EPA(complete) - EPA(incomplete) on toward throws: {gap:.3f} -> EPA change implied by the completion change alone: {implied:+.4f} per 10 pts")
    print(f"  observed EPA contrast {R.loc['epa', 'toward-away']:+.4f}; its noise SD {R.loc['epa', 'noise SD']:.4f}; minimum detectable (80% power) ~{mde:.3f}"
          f" = {mde / max(abs(implied), 1e-9):.1f}x the implied effect")
