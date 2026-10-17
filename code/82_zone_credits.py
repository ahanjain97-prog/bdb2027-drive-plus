"""Play-by-play coverage credits, ZONE coverage only (Drive+ is a zone trait).

Credits from 81b (PBU = named on an incompletion, INT, DPI, stop after a catch). Throw side from 76b/77 (play
description direction vs the CB's side at the snap; 99.9% validated). Samples, all ZONE_COVERAGE pass attempts:
  Z1 every zone snap of a graded pool DB                  Z2 lined up at CB / SLOT_CB
  Z3 match shells (Cover 3 / 4 / 6) at CB / slot          Z4 throws TO HIS SIDE (CB / slot, clear side)
  Z5 throws to his side, < 20 air yards (where the drive happens)
Outcomes per throw: ball play (PBU or INT), PBU, INT, completion, yards; per credited target: completion.
Model: y ~ Drive+ + position + alignment + season (+ defense team-season FE as a check); player-permutation p (500).
Benchmark: the PBU effect expected if all of the after-throw completion drop (77/80b) came from breakups.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(181)
PB = pd.read_parquet(f"{ROOT}/data/outcomes/81b_pbp_credits.parquet")
src = open(f"{ROOT}/code/77_epa_decomposition.py").read(); ns = {}; exec(src.split("def analyze")[0], ns)
side = ns["d"][["game_id", "play_id", "nfl_id", "rel", "air"]].drop_duplicates(["game_id", "play_id", "nfl_id"])
Z = PB[PB.team_coverage_man_zone == "ZONE_COVERAGE"].merge(side, on=["game_id", "play_id", "nfl_id"], how="left")
Z["completion"] = (Z.pass_result == "C").astype(float); Z["yds"] = Z.yards_gained
Z["def_ts"] = Z.team_abbr + Z.season.astype(str)
cs = Z["aln"] != "other"
MATCH = Z.team_coverage_type.isin(["COVER_3_ZONE", "COVER_4_ZONE", "COVER_6_ZONE"])
SAMPLES = [("Z1 all zone snaps", Z), ("Z2 zone at CB / slot", Z[cs]), ("Z3 match shells (C3/C4/C6) at CB / slot", Z[cs & MATCH]),
           ("Z4 zone throws to HIS side", Z[cs & (Z.rel == "toward")]), ("Z5 his side, < 20 air yds", Z[cs & (Z.rel == "toward") & (Z.air < 20)])]
print("zone throws:", len(Z), "| DBs", Z.nfl_id.nunique(), "| throws to his side (CB/slot):", int((cs & (Z.rel == "toward")).sum()))


def test(x, y, ctx, n_perm=500):
    x = x.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"
    mod = smf.ols(f, data=x); used = x.loc[mod.data.row_labels]
    m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id)[0]})
    t0 = smf.ols(f, data=x).fit().tvalues["dp"]; vals = x.drop_duplicates("nfl_id").set_index("nfl_id").dp; null = []
    for _ in range(n_perm):
        xx = x.copy(); xx["dp"] = xx.nfl_id.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=xx).fit().tvalues["dp"])
    return m.params["dp"], m.bse["dp"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(used), used.nfl_id.nunique(), used[y].mean()


rows = []
pd.set_option("display.width", 230)
for lab, x in SAMPLES:
    for y, sc, unit in [("ball_play", 100, "per 100 throws"), ("pbu", 100, "per 100 throws"), ("int_", 100, "per 100 throws"),
                        ("completion", 100, "pp"), ("yds", 1, "yds per throw"), ("comp_c", 100, "pp (credited targets)")]:
        for ctl, ctx in [("season", "is_s + C(aln) + C(season)"), ("+ team-season FE", "is_s + C(aln) + C(def_ts)")]:
            if lab.startswith("Z1") and ctl == "season":
                ctx = "is_s + C(aln) + C(season)"
            b, se, p, n, k, mu = test(x, y, ctx, n_perm=500 if ctl == "season" else 200)
            rows.append({"sample": lab, "outcome": y, "controls": ctl, "per 10 Drive+": b * sc, "SE": se * sc, "perm p": p, "mean": mu * sc, "n": n, "DBs": k, "unit": unit})
    print(f"done {lab}", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/82_zone_credits.csv", index=False)
for ctl in ["season", "+ team-season FE"]:
    print(f"\n=== zone only, controls: {ctl} (per 10 Drive+ points; player-permutation p) ===")
    T = R[R.controls == ctl].copy(); T["cell"] = T.apply(lambda r: f"{r['per 10 Drive+']:+.2f} (p {r['perm p']:.2f}) [mean {r['mean']:.1f}]", axis=1)
    print(T.pivot(index="sample", columns="outcome", values="cell")[["ball_play", "pbu", "int_", "completion", "yds", "comp_c"]].to_string())
    print("  n / DBs:", T.drop_duplicates("sample").set_index("sample")[["n", "DBs"]].to_dict("index"))
# benchmark: if the after-throw completion drop (-4.9 pp per 10 on targets of his man, 80b) were all breakups,
# PBU per throw to his side should rise by about 4.9 pp x (share of his-side throws that target his man ~ 0.19, from 77)
z4 = R[(R["sample"].str.startswith("Z4")) & (R.controls == "season") & (R.outcome == "pbu")].iloc[0]
exp_ = 4.9 * 0.19
print(f"\nbenchmark: expected PBU change on throws to his side if the after-throw effect were breakups ≈ +{exp_:.2f} per 100 throws; "
      f"observed {z4['per 10 Drive+']:+.2f} (SE {z4['SE']:.2f}) -> {abs(z4['per 10 Drive+'] - exp_) / z4['SE']:.1f} SE from the benchmark, {abs(z4['per 10 Drive+']) / z4['SE']:.1f} SE from zero")
