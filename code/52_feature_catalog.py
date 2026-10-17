"""Phase 2b: per-position feature catalog (reliable features only) + per-position outcome table.

Feature enters a unit's catalog if ANY reliability estimate >= REL_MIN:
  family composite        split-half over the family's drills                (50_trait_reliability.csv)
  drill feature           attempt 1 vs 2 test-retest                          (50_test_retest.csv)
  drill feature (events)  odd vs even breaks / turns within attempts          (50_within_attempt_reliability.csv)
Plus the frozen Hoop Turn Speed for DL and the official stopwatch results (reference only, marked kind=stopwatch).
Near-duplicates (|r| > .85 within the unit's players) are collapsed, keeping the more reliable feature.
Units: WR, TE, CB, S, EDGE, IDL, T, IOL (combine group: WR, TE, DB, DB, DL, DL, OL, OL).
Outcomes: library (40), coverage reps (41, DB = separation allowed, receiver = separation created; >= 8 reps),
tackle reps (42, >= 4 reps), DL/OL head-to-head reps (bend/30, >= 15 reps), box run fit (34 logic, >= 30 snaps).
Output data/features/catalog_{unit}.parquet, data/outcomes/outcomes_{unit}.parquet, results/52_catalog_summary.csv
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
REL_MIN = 0.4
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
TC = pd.read_parquet(f"{ROOT}/data/features/trait_composites.parquet")
PD = pd.read_parquet(f"{ROOT}/data/features/player_drill_features.parquet")
REL = pd.read_csv(f"{ROOT}/results/50_trait_reliability.csv")
TR = pd.read_csv(f"{ROOT}/results/50_test_retest.csv")
WI = pd.read_csv(f"{ROOT}/results/50_within_attempt_reliability.csv")
UNITS = {"WR": ("WR", ["WR"]), "TE": ("TE", ["TE"]), "CB": ("DB", ["CB"]), "S": ("DB", ["SS", "FS", "DB"]),
         "EDGE": ("DL", ["DE", "OLB"]), "IDL": ("DL", ["DT", "NT"]), "T": ("OL", ["T"]), "IOL": ("OL", ["G", "C"])}
UNIVERSAL = ["FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL"]
STOP = ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
sys.path.insert(0, f"{ROOT}/prereg")
from hoop_turn_speed import hoop_turn_speed
hoop = hoop_turn_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))

# reliability lookup for drill features
drill_rel = {}
for r in TR.itertuples():
    drill_rel[(r.drill, r.feature)] = max(drill_rel.get((r.drill, r.feature), -1), r.test_retest)
EVMAP = {("break", f): f"brk_mean_{f}" for f in ["entry_speed", "min_speed", "brake_time", "brake_decel", "brake_dist", "exit_burst", "reaccel_time", "cut_angle", "cut_retention"]}
EVMAP.update({("turn", "turn_speed"): "turn_speed", ("turn", "turn_lean"): "turn_lean"})
for r in WI.itertuples():
    col = EVMAP.get((r.kind, r.feature))
    if col:
        drill_rel[(r.drill, col)] = max(drill_rel.get((r.drill, col), -1), r.within_attempt_rel)
grp_of_drill = lambda d: d.split(":")[0] if ":" in d else "ALL"
summary = []
for unit, (cg, positions) in UNITS.items():
    ids = base.index[base.nfl_position.isin(positions)]
    feats, meta_rows = {}, []
    for r in REL[(REL.group == cg) & (REL.split_half_over_drills >= REL_MIN)].itertuples():
        col = f"{cg}|{r.family}|{r.feature}"
        if col in TC:
            feats[f"fam:{r.family}:{r.feature}"] = TC[col]
            meta_rows.append({"feature": f"fam:{r.family}:{r.feature}", "kind": "family_composite", "reliability": r.split_half_over_drills})
    for (d_, f), rel in drill_rel.items():
        if rel >= REL_MIN and (grp_of_drill(d_) == cg or d_ in UNIVERSAL) and f in PD.columns:
            v = PD[f].xs(d_, level="drill") if d_ in PD.index.get_level_values("drill") else None
            if v is not None:
                feats[f"drill:{d_}:{f}"] = v
                meta_rows.append({"feature": f"drill:{d_}:{f}", "kind": "drill_feature", "reliability": rel})
    if cg == "DL":
        feats["frozen:hoop_turn_speed"] = hoop; meta_rows.append({"feature": "frozen:hoop_turn_speed", "kind": "frozen_metric", "reliability": np.nan})
    Fm = pd.DataFrame(feats).reindex(ids)
    Fm = Fm.loc[:, Fm.notna().sum() >= 15]
    M = pd.DataFrame(meta_rows).drop_duplicates("feature").set_index("feature").reindex(Fm.columns)
    # collapse near-duplicates
    order = M.reliability.fillna(1.0).sort_values(ascending=False).index
    keep = []
    C = Fm.corr(min_periods=15).abs()
    for f in order:
        if all(not (C.loc[f, k] > .85) for k in keep):
            keep.append(f)
    Fm, M = Fm[keep], M.loc[keep]
    for c in STOP:
        Fm[f"stopwatch:{c}"] = base.loc[ids, c]; M.loc[f"stopwatch:{c}"] = ["stopwatch", np.nan]
    Fm["size:weight"] = base.loc[ids, "combine_weight"]; Fm["size:height"] = base.loc[ids, "combine_height"]; Fm["size:arm"] = base.loc[ids, "arm_length"]
    for c in ["size:weight", "size:height", "size:arm"]:
        M.loc[c] = ["size", np.nan]
    Fm.to_parquet(f"{ROOT}/data/features/catalog_{unit}.parquet"); M.to_csv(f"{ROOT}/data/features/catalog_{unit}_meta.csv")
    summary.append({"unit": unit, "players": len(ids), "tracking_features": int((M.kind.isin(["family_composite", "drill_feature", "frozen_metric"])).sum()),
                    "family": int((M.kind == "family_composite").sum()), "drill": int((M.kind == "drill_feature").sum())})

# ---------------- outcomes per unit
O = pd.read_parquet(f"{ROOT}/data/outcomes/player_outcomes.parquet")
lib = O.pivot_table(index="nfl_id", columns="outcome", values="value")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
cov = cov[cov.season_type.isin(["REG", "POST"]) & (cov.nullified != "Y")]
cm = ["sep_mean", "sep_at_break", "sep_gain_break", "sep_at_throw", "db_speed_deficit"]
dbc = cov.groupby("nfl_id_d")[cm].mean()[cov.groupby("nfl_id_d").size() >= 8].add_prefix("COV allowed ")
dbm = cov[cov.team_coverage_man_zone == "MAN_COVERAGE"].groupby("nfl_id_d")[cm].mean()[cov[cov.team_coverage_man_zone == "MAN_COVERAGE"].groupby("nfl_id_d").size() >= 6].add_prefix("COV allowed (man) ")
rcv = cov.groupby("nfl_id_r")[["sep_mean", "sep_gain_break", "sep_at_throw"]].mean()[cov.groupby("nfl_id_r").size() >= 8].add_prefix("COV created ")
tk = pd.read_parquet(f"{ROOT}/data/outcomes/tackle_reps.parquet")
tk = tk[tk.season_type.isin(["REG", "POST"])]
tkm = tk.groupby("nfl_id_tk")[["closing_speed", "pursuit_eff", "breakdown", "yds_after_arrival"]].mean()[tk.groupby("nfl_id_tk").size() >= 4].add_prefix("TACKLE ")
rep = pd.read_parquet(f"{ROOT}/data/bend/matchups.parquet")
rep = rep[rep.primary & rep.season_type.isin(["REG", "POST"]) & (rep.nullified != "Y")].copy()
rep["ppa_oe"] = rep.ppa - rep.groupby(["dl_align", "ol_align", "down"]).ppa.transform("mean")
dlr = pd.DataFrame({"H2H corner rate": rep.groupby("dl_id").corner_turned.mean(), "H2H blocker ppa_oe vs him": rep.groupby("dl_id").ppa_oe.mean()})[rep.groupby("dl_id").size() >= 15]
olr = pd.DataFrame({"H2H corner allowed rate": rep.groupby("ol_id").corner_turned.mean(), "H2H ppa_oe": rep.groupby("ol_id").ppa_oe.mean()})[rep.groupby("ol_id").size() >= 15]
UO = {"WR": [lib.filter(like="REC "), rcv], "TE": [lib.filter(like="REC "), rcv], "CB": [lib.filter(like="CB "), dbc, dbm, tkm],
      "S": [lib.filter(like="S "), dbc, dbm, tkm], "EDGE": [lib.filter(like="EDGE "), dlr], "IDL": [lib.filter(like="IDL "), dlr],
      "T": [lib.filter(regex="^T "), olr], "IOL": [lib.filter(like="IOL "), olr]}
for unit, parts in UO.items():
    ids = base.index[base.nfl_position.isin(UNITS[unit][1])]
    out = pd.concat(parts, axis=1).reindex(ids)
    out = out.loc[:, out.notna().sum() >= 15]
    out.to_parquet(f"{ROOT}/data/outcomes/outcomes_{unit}.parquet")
    s = next(x for x in summary if x["unit"] == unit)
    s["outcomes"] = out.shape[1]; s["outcome_list"] = "; ".join(f"{c} (n {out[c].notna().sum()})" for c in out.columns)
S = pd.DataFrame(summary); S.to_csv(f"{ROOT}/results/52_catalog_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 400)
print(S.drop(columns="outcome_list").to_string(index=False))
for r in S.itertuples():
    print(f"\n{r.unit} outcomes: {r.outcome_list}")
