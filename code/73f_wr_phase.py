"""Step 7 -- the receiver side of the same phase, and the phase-matched matchup.

Combine: WR route drills where the receiver keeps running after the break (speed out, in, dagger, corner, post-corner,
slot sail, slot strike, slant, comeback): phase_utils (break 25-180 deg, turn start >= 3 yd/s). Per drill, within-year
z of v_out .. v_10 and top; player value = mean over his route drills; reliability = split-half over drills.
NFL (CB reps, WR receivers): (a) WR late speed -> separation created at the break / at the throw, zone-matching vs
man-matching, with WR size + forty as controls in a second model; (b) matched-phase matchup: WR v_10 x CB v_10 (and
x CB frozen top) on sep_gain_break and closing; player-permutation p on both sides.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/code")
from phase_utils import phase
rng = np.random.default_rng(103)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ROUTES = ["SPEED_OUT_ROUTE_LEFT", "IN_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "CORNER_ROUTE", "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT",
          "SLOT_STRIKE_ROUTE_LEFT", "SLANT_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT"]
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
wrs = base.index[base.nfl_position == "WR"]
d = ct[(ct.entity_type == "PLAYER") & ct.drill_name.isin(ROUTES) & ct.nfl_id.isin(wrs)].copy(); d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"])
FE = ["v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "top"]
rows = []
for eid, g in d.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    p = phase(t, g.x.values.astype(float), g.y.values.astype(float), 25, 180, start_speed=3.0)
    if p:
        rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill_name.iat[0], "yr": g.draft_year.iat[0], **p})
A = pd.DataFrame(rows)
print(f"route attempts {d.event_id.nunique()}, break detected {len(A)}, WRs {A.nfl_id.nunique()}")
print("per drill: n, median turn angle, share clip ends < 1.0 s after break:\n",
      A.groupby("drill").agg(n=("event_id", "size"), angle=("turn_angle", "median"), trunc=("clip_after", lambda s: (s < 1.0).mean())).round(2).to_string())
for f in FE:
    grp = A.groupby(["drill", "yr"])[f]; A[f + "_z"] = (A[f] - grp.transform("mean")) / grp.transform("std")
A["dr_rank"] = A.groupby("nfl_id").drill.rank(method="first"); A["half"] = A.dr_rank % 2
rel = {}
for f in FE:
    h = A.groupby(["nfl_id", "half"])[f + "_z"].mean().unstack().dropna()
    r = np.corrcoef(h[0], h[1])[0, 1]; rel[f] = (round(r, 2), round(2 * r / (1 + r), 2), len(h))
print("split-half over route drills (r, Spearman-Brown, n):", rel)
W = A.groupby("nfl_id")[[f + "_z" for f in FE]].mean().rename(columns=lambda c: "wr_" + c[:-2])
print("WR late speed vs size/forty:", W.join(base[["forty", "combine_weight", "combine_height"]]).corr().loc[["wr_v_04", "wr_v_10", "wr_top"], ["forty", "combine_weight", "combine_height"]].round(2).to_dict())
W.to_parquet(f"{ROOT}/data/features/wr_route_phase.parquet")

cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cv = cov[cov.nfl_id_r.isin(wrs)].copy()
for c in W.columns:
    v = W[c].reindex(cv.nfl_id_r.unique()); cv[c] = cv.nfl_id_r.map((W[c] - v.mean()) / v.std())
for c in ["v_10", "top"]:
    v = Z[c].reindex(cv.nfl_id_d.unique()); cv["cb_" + c] = cv.nfl_id_d.map((Z[c] - v.mean()) / v.std())
for c in ["forty", "combine_weight", "combine_height"]:
    s_ = base[c].reindex(cv.nfl_id_r.unique()); cv["wr_" + c] = cv.nfl_id_r.map(((base[c] - s_.mean()) / s_.std()).fillna(0))
tot = cv.groupby("nfl_id_d").sep_gain_break.agg(["sum", "count"])
cv["cb_skill"] = (cv.nfl_id_d.map(tot["sum"]) - cv.sep_gain_break.fillna(0)) / (cv.nfl_id_d.map(tot["count"]) - 1).replace(0, np.nan)
cv["cb_skill"] = cv.cb_skill.fillna(cv.sep_gain_break.mean())
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def perm_p(d, f, term, who, col, n_perm=300):
    t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates(who).set_index(who)[col]; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd[col] = dd[who].map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1)


print(f"\n=== (a) WR route-drill phase -> separation CREATED (CB reps, WR receivers: {cv.nfl_id_r.nunique()} WRs) ===")
for cell in ["zone-matching", "man-matching"]:
    for y in ["sep_gain_break", "sep_at_throw"]:
        out = []
        for c in ["wr_v_out", "wr_v_04", "wr_v_10", "wr_top"]:
            for ctl, lab in [("", ""), (" + wr_forty + wr_combine_weight + wr_combine_height", " |size+40")]:
                dd = cv[cv.cell == cell].dropna(subset=[y, c]); f = f"{y} ~ {c} + cb_skill + C(route_ran){ctl}"
                m = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": CL(dd)})
                out.append(f"{c[3:]}{lab} {m.params[c]:+.3f} (p {perm_p(dd, f, c, 'nfl_id_r', c, 200):.3f})")
        print(f"  {cell:13s} {y:14s} " + " | ".join(out))
print("\n=== (b) phase-matched matchup: WR late speed x CB late speed ===")
for cell in ["zone-matching", "man-matching"]:
    for y, ctx in [("sep_gain_break", "C(route_ran)"), ("db_closing", "C(route_ran)")]:
        for wc, cc in [("wr_v_10", "cb_v_10"), ("wr_v_10", "cb_top"), ("wr_top", "cb_top")]:
            dd = cv[cv.cell == cell].dropna(subset=[y, wc, cc]); f = f"{y} ~ {wc} * {cc} + {ctx}"
            m = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": CL(dd)}); term = f"{wc}:{cc}"
            p1, p2 = perm_p(dd, f, term, "nfl_id_d", cc), perm_p(dd, f, term, "nfl_id_r", wc)
            print(f"  {cell:13s} {y:14s} {wc} x {cc}: WR {m.params[wc]:+.3f} | CB {m.params[cc]:+.3f} | interaction {m.params[term]:+.3f} "
                  f"(perm p CB-side {p1:.3f}, WR-side {p2:.3f}) | reps {len(dd)}, WRs {dd.nfl_id_r.nunique()}, CBs {dd.nfl_id_d.nunique()}")
