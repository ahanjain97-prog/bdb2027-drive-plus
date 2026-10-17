"""Audit checks not yet run on Drive+.

(A) Selection on the outcome. The registered rep (41) keeps pairs with mean separation <= 3 yd over a window that includes
    the post-break period, so a badly beaten rep can drop out. Rebuild reps with a PRE-BREAK-ONLY criterion (mean
    separation <= 3 yd from 0.5 s after the snap to the receiver's break, >= 5 frames) and re-estimate the main claims.
(B) Season heterogeneity: formal drive x season interaction (2025 vs 2024), with a player-permutation p.
(C) Go routes: the "break" on a vertical is just the largest lateral wiggle; re-estimate without GO routes.
All on graded DBs, position control, clustered SEs, player-permutation p (300).
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(211)
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
GR = GR[GR.nfl_position.isin(["CB", "SS", "FS", "DB"])]
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")


def est(d, y, term, ctx, n_perm=300):
    d = d.dropna(subset=[y, term]).copy(); f = f"{y} ~ {term} + {ctx}"
    mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]})
    t0 = mod.fit().tvalues[term]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term.split(":")[0]]; null = []
    base_term = term.split(":")[0]
    for _ in range(n_perm):
        dd = used.copy(); dd[base_term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index))
        null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return m.params[term], m.pvalues[term], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(used), used.nfl_id_d.nunique()


def claims(R, label):
    R = R.copy(); R["open"] = np.where(R.sep_at_throw.notna(), (R.sep_at_throw >= 2).astype(float), np.nan)
    out = []
    for nm, d, y, ctx, sc in [("sep gained at break (all)", R, "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran) + is_s", 1),
                              ("zone-match closing", R[R.cell == "zone-matching"], "db_closing", "wr_skill + C(route_ran) + is_s", 1),
                              ("open at throw (all)", R, "open", "C(team_coverage_man_zone) + C(route_ran) + is_s", 100),
                              ("completion when targeted", R[R.targeted == 1], "completion", "C(team_coverage_man_zone) + C(route_ran) + is_s", 100)]:
        b, p, pp_, n, k = est(d, y, "dp", ctx); out.append(f"{nm}: {b * sc:+.3f} (perm p {pp_:.3f}; n {n}, DBs {k})")
    print(f"  [{label}]\n    " + "\n    ".join(out), flush=True)


# ---------------- (A) rebuild reps with a pre-break-only criterion
m = meta.merge(pl, on="nfl_id"); db = (m.play_type == "DROPBACK") & m.season_type.isin(["REG", "POST"]) & (m.nullified != "Y")
rec = m[db & m.nfl_position.isin(["WR", "TE"]) & m.route_ran.notna()]
dbk = m[db & m.nfl_id.isin(GR.index)]
P = rec[["game_id", "play_id", "idx", "nfl_id", "route_ran", "team_coverage_man_zone", "season"]].merge(
    dbk[["game_id", "play_id", "idx", "nfl_id"]], on=["game_id", "play_id"], suffixes=("_r", "_d"))
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = P.idx_r.values, P.idx_d.values
xr, yr, sr, lr, pr = g(A, "x_rel"), g(A, "y_field"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
xd, yd, sd = g(B, "x_rel"), g(B, "y_field"), g(B, "s")
vxr, vyr, vxd, vyd = g(A, "vx"), g(A, "vy"), g(B, "vx"), g(B, "vy")
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]; r_ = np.arange(len(P))
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1); hb = a.max(1) > 0
dist = np.hypot(xr - xd, yr - yd)
pre_w = vv & (k >= 5) & (cols < jb[:, None]); n_pre = pre_w.sum(1)
sep_pre = np.where(n_pre > 0, np.where(pre_w, dist, 0).sum(1) / np.maximum(n_pre, 1), np.nan)
keep = hb & (n_pre >= 5) & (sep_pre <= 3.0)
print(f"(A) pairs {len(P)}, pre-break-only reps {int(keep.sum())} (registered definition had 2,040 reps on graded DBs)")
P = P[keep].reset_index(drop=True); idx = np.where(keep)[0]; r_ = np.arange(len(P)); jb = jb[idx]
dist, vv, pr, sd, sr = dist[idx], vv[idx], pr[idx], sd[idx], sr[idx]; vxr, vyr, vxd, vyd = vxr[idx], vyr[idx], vxd[idx], vyd[idx]
at = lambda arr, j: np.where((j < T) & (j >= 0) & vv[r_, np.clip(j, 0, T - 1)], arr[r_, np.clip(j, 0, T - 1)], np.nan)
P["sep_gain_break"] = at(dist, jb + 5) - at(dist, jb - 3)
P["db_closing"] = (at(dist, jb) - at(dist, jb + 7)) / 0.7
thr = (pr >= .5) & vv & (k[:, :] >= 0)[:len(P)] if False else (pr >= .5) & vv & (np.arange(T)[None, :] >= PRE)
jt = thr.argmax(1); P["sep_at_throw"] = np.where(thr.any(1), dist[r_, jt], np.nan)
w = vv & (cols[:len(P)] if False else cols) >= (jb - 5)[:, None]
w = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
pa = np.where(w.sum(1) >= 3, np.where(w, cosv, 0).sum(1) / np.maximum(w.sum(1), 1), np.nan)
P["cell"] = np.where(pa >= .7, np.where(P.team_coverage_man_zone == "ZONE_COVERAGE", "zone-matching", "man-matching"), "other")
kr = pd.MultiIndex.from_arrays([P.game_id, P.play_id, P.nfl_id_r])
P["targeted"] = pp.target.reindex(kr).values; res = pp.pass_result.reindex(kr).values
P["completion"] = np.where(P.targeted == 1, (res == "C").astype(float), np.nan)
tot = P.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
P["wr_skill"] = ((P.nfl_id_r.map(tot["sum"]) - P.sep_gain_break.fillna(0)) / (P.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)).fillna(P.sep_gain_break.mean())
P["dp"] = P.nfl_id_d.map((GR.drive_plus - 100) / 10); P["is_s"] = (P.nfl_id_d.map(GR.nfl_position) != "CB").astype(int)
print(f"    reps {len(P)}, DBs {P.nfl_id_d.nunique()}, targets {int((P.targeted == 1).sum())}, cells {P.cell.value_counts().to_dict()}")
claims(P, "pre-break-only selection")

# ---------------- registered reps for (B) and (C)
R = pd.read_parquet(f"{ROOT}/data/outcomes/79_scored_coverage_reps.parquet")
R["dp"] = R.nfl_id_d.map((GR.drive_plus - 100) / 10)
claims(R, "registered 3-yd reps (reference)")
print("\n(B) season heterogeneity: drive x 2025 interaction (2024 + 2025 seasons; 2023 too thin)")
S = R[R.season.isin([2024, 2025])].copy(); S["s25"] = (S.season == 2025).astype(int); S["open"] = np.where(S.sep_at_throw.notna(), (S.sep_at_throw >= 2).astype(float), np.nan)
for nm, d, y, ctx, sc in [("sep gained at break (all)", S, "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran) + is_s + s25", 1),
                          ("zone-match closing", S[S.cell == "zone-matching"], "db_closing", "wr_skill + C(route_ran) + is_s + s25", 1),
                          ("open at throw (all)", S, "open", "C(team_coverage_man_zone) + C(route_ran) + is_s + s25", 100),
                          ("completion when targeted", S[S.targeted == 1], "completion", "C(team_coverage_man_zone) + C(route_ran) + is_s + s25", 100)]:
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + dp:s25 + {ctx}"
    mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]; m_ = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]})
    t0 = mod.fit().tvalues["dp:s25"]; vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(300):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp:s25"])
    print(f"    {nm}: 2024 slope {m_.params['dp'] * sc:+.3f}, 2025 minus 2024 {m_.params['dp:s25'] * sc:+.3f} (clustered p {m_.pvalues['dp:s25']:.3f}, perm p {(1 + (np.abs(null) >= abs(t0)).sum()) / 301:.3f}; DBs {used.nfl_id_d.nunique()})", flush=True)
print("\n(C) without GO routes")
claims(R[R.route_ran != "GO"], "registered reps, GO routes excluded")
