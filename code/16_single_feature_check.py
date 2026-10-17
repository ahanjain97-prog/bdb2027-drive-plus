"""Do multi-feature blocks dilute a single strong combine feature? Same CV/ridge machinery as 15, one feature at a time."""
import sys
import numpy as np
sys.argv = ["x", "--n_rep", "10"]
exec(open("/Users/ahanjain/bdb_ssl/code/08_probe.py").read().split("tests = []")[0])
caps = pd.read_parquet(f"{ROOT}/data/game_caps.parquet"); caps = caps[caps.n_plays >= 80]
att = pd.read_parquet(f"{ROOT}/data/combine_attempt_caps.parquet")
c40 = att[att.drill_type == "FORTY_YARD_DASH"].groupby("nfl_id")[["top_speed", "accel_peak"]].max().add_prefix("c40_")
call = att.groupby("nfl_id")[["decel_peak", "lateral_peak", "amag_peak"]].max().add_prefix("call_")
rows = []
for grp in ["REC", "OL", "DL", "DB"]:
    ids = caps.index.intersection(base.index[base.grp == grp])
    ids = ids[base.loc[ids, "combine_weight"].notna()]
    sub = base.loc[ids, "sub"].values
    S0 = pd.DataFrame(index=ids)
    for s_ in sorted(set(sub))[1:]:
        S0[f"sub_{s_}"] = (sub == s_).astype(float)
    for c in ["combine_height", "combine_weight", "arm_length"]:
        S0[c] = base.loc[ids, c]
    B = S0.values.astype(float)
    feats = {"official_forty": base.loc[ids, ["forty"]].values, "official_10yd": base.loc[ids, ["ten_yd_split"]].values,
             "tracked_40_top_speed": c40.reindex(ids)[["c40_top_speed"]].values,
             "tracked_40_peak_accel": c40.reindex(ids)[["c40_accel_peak"]].values,
             "tracked_peak_decel": call.reindex(ids)[["call_decel_peak"]].values,
             "tracked_peak_lateral": call.reindex(ids)[["call_lateral_peak"]].values}
    for t in ["top_speed", "accel_peak", "decel_peak", "lateral_peak"]:
        y = caps.loc[ids, "g_" + t].values.astype(float)
        y = (y - y.mean()) / y.std()
        folds = make_folds(np.arange(len(y)), 10, 0)
        cv = CV(B, y, folds)
        for fn, P in feats.items():
            P = P.astype(float)
            rows.append({"group": grp, "game_target": t, "combine_feature": fn, "n": len(y), "miss": np.isnan(P).mean(),
                         "r2_size": cv.r2_base, "r2_size_plus": cv.r2_full(P), "delta": cv.r2_full(P) - cv.r2_base})
R = pd.DataFrame(rows)
R.to_csv(f"{ROOT}/results/16_single_feature_check.csv", index=False)
pd.set_option("display.width", 250)
print(R.pivot_table(index=["group", "game_target"], columns="combine_feature", values="delta").round(3).to_string())
print("\nsize-only R^2:"); print(R.groupby(["group", "game_target"]).r2_size.first().round(3).unstack().to_string())
