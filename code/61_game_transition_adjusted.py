"""Step 1b: context-adjusted in-game transition drive, capacity percentiles, and the bridge to coverage outcomes.

Each transition event's drive top speed / drive gain is residualized on its context (OLS on all CB events): coverage
(man / zone), the CB's assignment family, alignment (outside / slot), down, drive direction (downhill), before / after
the throw, turn time after the snap (binned), speed at the end of the backpedal (binned), and whether the drive was
cut short by the play ending (drive window < 1.0 s). Per CB: mean residual and 90th-percentile residual ("capacity").
Then: split-half reliability, correlation with combine drive speed, and correlation with the CB's coverage-rep outcomes
(closing after the break, separation at the throw) -> is the in-game metric a bridge?
"""
import hashlib, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
E = pd.read_parquet(f"{ROOT}/data/outcomes/cb_game_transitions.parquet")
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet").set_index(["game_id", "play_id", "nfl_id"])
key = pd.MultiIndex.from_arrays([E.game_id, E.play_id, E.nfl_id])
E["asg"] = meta.coverage_assignment.reindex(key).fillna("NA").str.replace("_LEFT|_RIGHT|_MIDDLE", "", regex=True).values
E["align"] = meta.lined_up_position.reindex(key).fillna("NA").values
E["down"] = meta.down.reindex(key).fillna(1).values
E["turn_bin"] = pd.cut(E.k_turn, [-1, 1, 2, 3, 4, 99], labels=False)
E["bp_bin"] = pd.cut(E.g_bp_speed, [0, 2, 3, 4, 99], labels=False)
E["cov"] = E.team_coverage_man_zone.fillna("NA")
E["half"] = E.game_id.map(lambda g: int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2)
CTX = "C(cov) + C(asg) + C(align) + C(down) + g_downhill + g_post_throw + C(turn_bin) + C(bp_bin)"
for y in ["g_drive_top", "g_drive_gain_07"]:
    d = E.dropna(subset=[y])
    m = smf.ols(f"{y} ~ {CTX}", data=d).fit()
    E.loc[d.index, y + "_res"] = m.resid
    print(f"context R^2 for {y}: {m.rsquared:.3f}")
drive = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_enriched.parquet")
cov = cov.dropna(subset=["wr_skill"])
out_cl = cov.groupby("nfl_id_d").db_closing.mean()[cov.groupby("nfl_id_d").size() >= 10]
dd = cov.dropna(subset=["sep_at_throw"]).copy()
dd["sep_res"] = smf.ols("sep_at_throw ~ wr_skill + C(team_coverage_man_zone) + C(route_ran)", data=dd).fit().resid
out_sep = dd.groupby("nfl_id_d").sep_res.mean()[dd.groupby("nfl_id_d").size() >= 10]
q90 = lambda x: np.nanpercentile(x, 90)
rows = []
for col in ["g_drive_top_res", "g_drive_gain_07_res"]:
    for agg_name, agg in [("mean", "mean"), ("p90", q90)]:
        for sub_name, sub in [("all", E), ("zone", E[E["cov"] == "ZONE_COVERAGE"]), ("before throw", E[E.g_post_throw == 0])]:
            g = sub.groupby("nfl_id")[col]; n = g.size()
            val = g.agg(agg)[n >= 20]
            h = sub.groupby(["nfl_id", "half"])[col].agg([agg_name if agg_name == "mean" else q90, "size"])
            h.columns = ["v", "size"]; h = h[h["size"] >= 10]["v"].unstack().dropna()
            r = np.corrcoef(h[0], h[1])[0, 1]
            j = lambda s2: pd.concat([val.rename("a"), s2.rename("b")], axis=1).dropna()
            rows.append({"metric": f"{col} {agg_name} [{sub_name}]", "CBs": len(val), "split_half": 2 * r / (1 + r),
                         "r_combine_drive": j(drive).corr().iloc[0, 1], "r_cov_closing": j(out_cl).corr().iloc[0, 1],
                         "r_cov_sep_at_throw_adj": j(out_sep).corr().iloc[0, 1], "n_cov": len(j(out_cl))})
R = pd.DataFrame(rows)
R.to_csv(f"{ROOT}/results/61_game_transition_adjusted.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(2).to_string(index=False))
P = E.groupby("nfl_id").agg(g_drive_res_mean=("g_drive_top_res", "mean"), g_drive_res_p90=("g_drive_top_res", q90),
                            g_gain_res_mean=("g_drive_gain_07_res", "mean"), n_events=("row", "size"))
P[P.n_events >= 20].to_parquet(f"{ROOT}/data/outcomes/cb_game_drive_player.parquet")
