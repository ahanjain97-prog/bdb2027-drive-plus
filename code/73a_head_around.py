"""Step 2 -- head-around: are high-drive CBs turned to the ball sooner? (the completion effect 72d could not explain)

In-game body orientation o is tracked. Standardized frame (X.npy): facing vector = (sin o, cos o). Ball origin is
approximated by the pocket: QB at x_rel = -7 (7 yd behind the LOS), y_field = 0 (field centre; hash error <= 3.3 yd).
  face(t)       cos(angle between the CB's facing and the bearing CB -> pocket): +1 eyes on the QB, -1 back to him
  face_release  at the pass-forward frame          face_arrive  at the arrival frame (72d, targeted reps only)
  t_open        time from release until face >= 0 (0 if already open; = flight if never) -- targeted reps
  open_by_arr   opened before arrival (0/1)
  face_05/10    face 0.5 / 1.0 s after release -- all CB reps with a throw (non-targeted too)
Tests: split-half reliability (is it a CB trait?); combine drive -> each (perm p, context man/zone + route + cushion);
completion ~ face + separation (does facing the ball matter beyond being close?); mediation of drive -> completion.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(73)
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
fl = pd.read_parquet(f"{ROOT}/data/outcomes/cb_targets_flight.parquet")[["game_id", "play_id", "nfl_id_d", "flight", "d_release", "d_arrive", "t_arrive_snap"]]
cov = cov.merge(fl, on=["game_id", "play_id", "nfl_id_d"], how="left")
for c in ["top", "v_10"]:
    v = Z[c].reindex(cov.nfl_id_d.unique()); cov["c_" + c] = cov.nfl_id_d.map((Z[c] - v.mean()) / v.std())
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
B = cov.idx_d.values
xd, yd, so, co, pp_ = g(B, "x_rel"), g(B, "y_field"), g(B, "sin_o"), g(B, "cos_o"), g(cov.idx_r.values, "post_pass")
vv = V[B]; T = X.shape[1]; r_ = np.arange(len(cov))
bx, by = -7.0 - xd, 0.0 - yd; nb = np.hypot(bx, by) + 1e-6
face = (so * bx + co * by) / nb
jr = np.where((pp_ >= .5).any(1), (pp_ >= .5).argmax(1), -1)
at = lambda arr, j: np.where((j >= 0) & (j < T) & vv[r_, np.clip(j, 0, T - 1)], arr[r_, np.clip(j, 0, T - 1)], np.nan)
cov["face_release"] = at(face, jr)
cov["face_05"] = at(face, np.where(jr >= 0, jr + 5, -1)); cov["face_10"] = at(face, np.where(jr >= 0, jr + 10, -1))
ja = np.where(cov.t_arrive_snap.notna(), np.round(cov.t_arrive_snap.fillna(0) * 10).astype(int) + PRE, -1)
cov["face_arrive"] = at(face, ja)
t_open = np.full(len(cov), np.nan)
for i in np.where((jr >= 0) & (ja > jr))[0]:
    seg = face[i, jr[i]:ja[i] + 1]; ok = vv[i, jr[i]:ja[i] + 1]
    hit = np.where(ok & (seg >= 0))[0]
    t_open[i] = hit[0] / 10 if len(hit) else (ja[i] - jr[i]) / 10
cov["t_open"] = t_open
cov["open_by_arr"] = np.where(cov.t_open.notna(), (cov.face_arrive >= 0).astype(float), np.nan)
tg = cov[cov.targeted == 1]
print(f"targeted reps with arrival {tg.face_arrive.notna().sum()}, all reps with a throw {cov.face_release.notna().sum()}")
print("medians (targeted):", tg[["face_release", "face_arrive", "t_open", "open_by_arr"]].median().round(2).to_dict(),
      "| share open by arrival", round(tg.open_by_arr.mean(), 2))
print("face at release by coverage:", cov.groupby("team_coverage_man_zone").face_release.mean().round(2).to_dict(),
      "| by cell:", cov.groupby("cell").face_release.mean().round(2).to_dict())

# reliability: split-half over each CB's reps
cov = cov.sort_values(["nfl_id_d", "game_id", "play_id"]); cov["half"] = cov.groupby("nfl_id_d").cumcount() % 2
for m_ in ["face_release", "face_05", "face_10"]:
    d = cov.dropna(subset=[m_, "cushion"])
    d = d.assign(res=smf.ols(f"{m_} ~ C(team_coverage_man_zone) + C(route_ran) + cushion", data=d).fit().resid)
    h = d.groupby(["nfl_id_d", "half"]).res.mean().unstack(); n = d.groupby(["nfl_id_d", "half"]).size().unstack()
    h = h[n.min(1) >= 8]
    r = np.corrcoef(h[0], h[1])[0, 1]
    print(f"  split-half reliability of {m_} (context-adjusted, CBs {len(h)}): r {r:+.2f}")
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def eff(d, y, ctx, term="c_top", n_perm=400, logit=False):
    d = d.dropna(subset=[y, term, "cushion"]).copy(); f = f"{y} ~ {term} + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd[term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return m.params[term], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d), d.nfl_id_d.nunique()


CTX = "C(team_coverage_man_zone) + C(route_ran) + cushion"
print("\n=== combine drive (per SD) -> facing the ball ===")
for y, d, lab in [("face_release", cov, "all reps w/ throw"), ("face_05", cov, "all reps w/ throw"), ("face_10", cov, "all reps w/ throw"),
                  ("face_05", cov[cov.cell == "zone-matching"], "zone-matching"), ("face_10", cov[cov.cell == "zone-matching"], "zone-matching"),
                  ("face_05", cov[cov.cell == "man-matching"], "man-matching"),
                  ("face_release", tg, "targeted"), ("face_arrive", tg, "targeted"), ("t_open", tg, "targeted"), ("open_by_arr", tg, "targeted")]:
    for term in ["c_top", "c_v_10"]:
        b, p, n, k = eff(d, y, CTX if lab != "zone-matching" and lab != "man-matching" else "C(route_ran) + cushion", term=term)
        print(f"  {lab:18s} {y:13s} {term:7s} {b:+.3f} (perm p {p:.3f}, reps {n}, CBs {k})")
print("\n=== completion (logit, targeted): does facing the ball matter beyond separation? ===")
d = tg.dropna(subset=["completion", "face_arrive", "d_release", "d_arrive", "c_top", "flight"]).copy()
for c in ["face_arrive", "face_release", "d_release", "d_arrive", "t_open"]:
    d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
for f in ["completion ~ c_top + flight + C(team_coverage_man_zone)",
          "completion ~ z_face_arrive + z_d_arrive + flight + C(team_coverage_man_zone)",
          "completion ~ c_top + z_face_arrive + z_d_arrive + flight + C(team_coverage_man_zone)",
          "completion ~ c_top + z_face_release + z_t_open + z_d_release + z_d_arrive + flight + C(team_coverage_man_zone)"]:
    m = smf.logit(f, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]})
    print(f"  {' + '.join(t for t in m.params.index if t.startswith(('c_', 'z_'))):55s} " + " | ".join(f"{t} {m.params[t]:+.3f} (p {m.pvalues[t]:.3f})" for t in m.params.index if t.startswith(("c_", "z_"))) + f"  n {len(d)}")
cov.to_parquet(f"{ROOT}/data/outcomes/cb_reps_face.parquet", index=False)
