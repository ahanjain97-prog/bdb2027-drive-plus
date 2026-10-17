"""Step 3 -- second drill, same phase: BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL.

Same decomposition as 72a (phase_utils; the break must be 50-135 deg). Prediction from 72b: speed 0.8-1.0 s after the
turn (and top) predicts zone-matching closing / CPOE; speed out of the turn and after ~2 steps does not.
Also: correlation with the 45-degree drill's phases (same trait?), and a two-drill composite (mean of the within-year
z of v_10 in both drills) -- does pooling drills raise reliability and prediction?
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/code")
from phase_utils import phase
rng = np.random.default_rng(79)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
d = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL")].copy()
d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"])
rows, angs = [], []
for eid, g in d.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    p = phase(t, g.x.values.astype(float), g.y.values.astype(float), 50, 135)
    p0 = phase(t, g.x.values.astype(float), g.y.values.astype(float), 0, 360)
    if p0:
        angs.append(p0["turn_angle"])
    if p:
        rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "yr": g.draft_year.iat[0], "t0": g.time.iat[0], **p})
A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"]); A["att"] = A.groupby("nfl_id").cumcount()
print(f"attempts {d.event_id.nunique()}, detected turn angle quantiles {np.percentile(angs, [10, 25, 50, 75, 90]).round(0)}, valid {len(A)}, players {A.nfl_id.nunique()}")
print("clip ends < 1.0 s after turn, share by year:", A.assign(c=A.clip_after < 1.0).groupby("yr").c.mean().round(2).to_dict())
FE = ["v_entry", "v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "top"]
print("medians by year:\n", A.groupby("yr")[FE].median().round(2).to_string())
two = A[A.att <= 1].pivot(index="nfl_id", columns="att", values=FE)
rel = {f: (round(two[f].dropna().corr().iloc[0, 1], 2), len(two[f].dropna())) for f in FE if two[f].dropna().shape[0] >= 15}
print("attempt 1 vs 2 r:", rel)
P = A.groupby("nfl_id")[FE].mean(); yr = base.draft_year.reindex(P.index)
Z9 = (P - P.groupby(yr).transform("mean")) / P.groupby(yr).transform("std")
Z9.to_parquet(f"{ROOT}/data/features/cb_bp90_parts.parquet")
Z45 = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cbs = base.index[base.nfl_position == "CB"]
J = Z9.reindex(cbs).add_prefix("bp90_").join(Z45.reindex(cbs)[["v_out", "v_04", "v_08", "v_10", "top"]].add_prefix("bp45_"))
print(f"\nBP-90 vs BP-45 phases (CBs with both: {J.dropna(subset=['bp90_v_10', 'bp45_v_10']).shape[0]}):")
print(J.corr().loc[["bp90_v_out", "bp90_v_04", "bp90_v_08", "bp90_v_10", "bp90_top"], ["bp45_v_out", "bp45_v_04", "bp45_v_08", "bp45_v_10", "bp45_top"]].round(2).to_string())
comp = pd.concat([Z9.v_10, Z45.v_10], axis=1).mean(1)          # two-drill composite (either drill if only one)
comp_top = pd.concat([Z9.top, Z45.top], axis=1).mean(1)

cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
SER = {**{f"bp90_{c}": Z9[c] for c in ["v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "top"]}, "bp45_top (frozen)": Z45.top, "bp45_v_10": Z45.v_10,
       "composite v_10 (both drills)": comp, "composite top (both drills)": comp_top}
for k_, s_ in SER.items():
    v = s_.reindex(cov.nfl_id_d.unique()); cov["z_" + k_] = cov.nfl_id_d.map((s_ - v.mean()) / v.std())
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def eff(d, y, ctx, term, n_perm=300):
    d = d.dropna(subset=[y, term] + (["wr_skill"] if "wr_skill" in ctx else [])).copy(); d["T_"] = d[term]; f = f"{y} ~ T_ + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues["T_"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")["T_"]; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["T_"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["T_"])
    return m.params["T_"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), d.nfl_id_d.nunique()


TASKS = [("zone-matching", "db_closing", "wr_skill + C(route_ran)"), ("zone-matching", "sep_gain_break", "wr_skill + C(route_ran)"),
         ("all", "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"), ("targeted", "cpoe", "rec_target_prop + C(team_coverage_man_zone)")]
sub = lambda u: cov[cov.cell == u] if u == "zone-matching" else (cov[cov.targeted == 1] if u == "targeted" else cov)
rows = []
for k_ in SER:
    for u, y, ctx in TASKS:
        b, p, n = eff(sub(u), y, ctx, "z_" + k_)
        rows.append({"metric": k_, "task": f"{u}:{y}", "b": b, "p_perm": p, "CBs": n})
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/73b_bp90.csv", index=False)
pd.set_option("display.width", 240)
print("\n=== per SD (player-permutation p) ===")
print(R.pivot_table(index="metric", columns="task", values="b", sort=False).round(3).to_string())
print(R.pivot_table(index="metric", columns="task", values="p_perm", sort=False).round(3).to_string())
print("CBs:", R.groupby("metric").CBs.max().to_dict())
