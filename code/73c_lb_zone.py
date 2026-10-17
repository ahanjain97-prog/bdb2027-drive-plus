"""Step 4 -- linebackers: does the same late-drive phase translate for zone droppers?

Combine: SHORT_ZONE_BREAKS (40 LBs, ~3 attempts; 2023-24 only) and BACK_PEDAL_AND_REACT (59, 1 attempt): phase_utils
decomposition (break 25-180 deg, turn start >= 2 yd/s); attempt reliability (odd vs even attempt); within-year z.
NFL: LB coverage reps built like 41 (pool receiver WR/TE/RB route + pool LB on the same dropback, mean distance <= 3 yd
for >= 1 s): sep_gain_break, closing over 0.7 s after the receiver's break, sep_at_throw, completion when targeted.
Tests: each drill phase -> outcomes (all reps, zone reps), player-permutation p.
"""
import json, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/code")
from phase_utils import phase
rng = np.random.default_rng(101)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
FE = ["v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "top"]
PARTS = {}
for dr in ["SHORT_ZONE_BREAKS", "BACK_PEDAL_AND_REACT"]:
    d = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == dr)].copy(); d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"])
    rows = []
    for eid, g in d.groupby("event_id"):
        t = (g.time - g.time.iat[0]).dt.total_seconds().values
        p = phase(t, g.x.values.astype(float), g.y.values.astype(float), 25, 180, start_speed=2.0)
        if p:
            rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "yr": g.draft_year.iat[0], "t0": g.time.iat[0], **p})
    A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"]); A["att"] = A.groupby("nfl_id").cumcount()
    for f in FE:                                                       # z within year (and within break-angle tercile)
        A["ang"] = pd.qcut(A.turn_angle, 3, labels=False, duplicates="drop")
        grp = A.groupby(["yr", "ang"])[f]; A[f + "_z"] = (A[f] - grp.transform("mean")) / grp.transform("std")
    print(f"{dr}: attempts {len(A)}, players {A.nfl_id.nunique()}, clip ends < 1.0 s after turn: {(A.clip_after < 1.0).mean():.2f}")
    if A.att.max() >= 1:
        A["par"] = A.att % 2; rel = {}
        for f in FE:
            h = A.groupby(["nfl_id", "par"])[f + "_z"].mean().unstack().dropna()
            if len(h) >= 12:
                rel[f] = (round(np.corrcoef(h[0], h[1])[0, 1], 2), len(h))
        print("   odd vs even attempt r:", rel)
    PARTS[dr] = A.groupby("nfl_id")[[f + "_z" for f in FE]].mean().rename(columns=lambda c: c[:-2])

# ---------------- LB coverage reps
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet"); pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
m = meta.merge(pl, on="nfl_id"); db = (m.play_type == "DROPBACK") & m.season_type.isin(["REG", "POST"])
rec = m[db & m.nfl_position.isin(["WR", "TE", "RB", "FB"]) & m.route_ran.notna()]
lb = m[db & m.nfl_position.isin(["OLB", "ILB", "MLB", "LB"])]
P = rec[["game_id", "play_id", "idx", "nfl_id", "route_ran", "team_coverage_man_zone", "season"]].merge(
    lb[["game_id", "play_id", "idx", "nfl_id", "coverage_assignment"]], on=["game_id", "play_id"], suffixes=("_r", "_d"))
get = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A_, B_ = P.idx_r.values, P.idx_d.values
xr, yr, sr, lr, pr = get(A_, "x_rel"), get(A_, "y_field"), get(A_, "s"), get(A_, "a_lat"), get(A_, "post_pass")
xd, yd = get(B_, "x_rel"), get(B_, "y_field")
vv = V[A_] & V[B_]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE
route = vv & (k >= 5) & (k <= 35) & (pr < .5); dist = np.hypot(xr - xd, yr - yd); nl = route.sum(1)
sep_mean = np.where(nl > 0, np.where(route, dist, 0).sum(1) / np.maximum(nl, 1), np.nan)
keep = (nl >= 10) & (sep_mean <= 3.0)
P = P[keep].reset_index(drop=True); dist, vv, sr, lr, pr = dist[keep], vv[keep], sr[keep], lr[keep], pr[keep]
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1); hb = a.max(1) > 0
r_ = np.arange(len(P)); at = lambda arr, j: np.where(vv[r_, np.clip(j, 0, T - 1)] & (j < T), arr[r_, np.clip(j, 0, T - 1)], np.nan)
P["sep_gain_break"] = np.where(hb, at(dist, jb + 5) - at(dist, jb - 3), np.nan)
P["lb_closing"] = np.where(hb, (at(dist, jb) - at(dist, jb + 7)) / 0.7, np.nan)
thr = (pr >= .5) & vv & (k >= 0); jt = thr.argmax(1)
P["sep_at_throw"] = np.where(thr.any(1), at(dist, jt), np.nan)
o = pp.set_index(["game_id", "play_id", "nfl_id"]); kr = pd.MultiIndex.from_arrays([P.game_id, P.play_id, P.nfl_id_r])
P["targeted"] = o.target.reindex(kr).values; P["pass_result"] = o.pass_result.reindex(kr).values
P["completion"] = np.where(P.targeted == 1, (P.pass_result == "C").astype(float), np.nan)
tot = P.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
P["wr_skill"] = (P.nfl_id_r.map(tot["sum"]) - P.sep_gain_break.fillna(0)) / (P.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)
P["wr_skill"] = P.wr_skill.fillna(P.sep_gain_break.mean())
P.to_parquet(f"{ROOT}/data/outcomes/lb_coverage_reps.parquet", index=False)
print(f"\nLB coverage reps {len(P)}, LBs {P.nfl_id_d.nunique()}, receivers {P.nfl_id_r.nunique()}, targeted {int((P.targeted == 1).sum())} | man/zone:",
      P.team_coverage_man_zone.value_counts().to_dict())
for dr, Zp in PARTS.items():
    print(f"  LBs with reps AND {dr}: {len(set(P.nfl_id_d) & set(Zp.index))}")
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def eff(d, y, ctx, s_, n_perm=300):
    v = s_.reindex(d.nfl_id_d.unique()); d = d.assign(T_=d.nfl_id_d.map((s_ - v.mean()) / v.std())).dropna(subset=[y, "T_"])
    if d.nfl_id_d.nunique() < 8:
        return None
    f = f"{y} ~ T_ + {ctx}"; m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues["T_"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").T_; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["T_"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["T_"])
    return m.params["T_"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d), d.nfl_id_d.nunique()


forty = base.forty.where(base.forty.notna())
rows = []
for dr, Zp in list(PARTS.items()) + [("official 40 (reference, sign flipped)", pd.DataFrame({"top": -forty}))]:
    for f in Zp.columns:
        for lab, d, y, ctx in [("all", P, "lb_closing", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
                               ("all", P, "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
                               ("zone", P[P.team_coverage_man_zone == "ZONE_COVERAGE"], "lb_closing", "wr_skill + C(route_ran)"),
                               ("zone", P[P.team_coverage_man_zone == "ZONE_COVERAGE"], "sep_gain_break", "wr_skill + C(route_ran)"),
                               ("all", P, "sep_at_throw", "wr_skill + C(team_coverage_man_zone) + C(route_ran)")]:
            r = eff(d, y, ctx, Zp[f])
            if r:
                rows.append({"drill": dr, "phase": f, "task": f"{lab}:{y}", "b": r[0], "p_perm": r[1], "reps": r[2], "LBs": r[3]})
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/73c_lb.csv", index=False)
pd.set_option("display.width", 240)
print("\nper SD (player-permutation p in second table):")
print(R.pivot_table(index=["drill", "phase"], columns="task", values="b", sort=False).round(3).to_string())
print(R.pivot_table(index=["drill", "phase"], columns="task", values="p_perm", sort=False).round(3).to_string())
print("LBs per test:", R.groupby("drill").LBs.max().to_dict())
