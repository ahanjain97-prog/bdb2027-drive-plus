"""Drive+ for every defensive back who ran the 45-degree transition drill (CBs + safeties + DBs), on the CB scale, and a
validation test for safeties when they play CORNER (lined up at SLOT_CB or CB).

Drive+ = 100 + 10 * (z - mean_CB) / sd_CB, where z = the frozen drive speed z-scored within combine year over every
player who ran the drill (72a), and mean_CB / sd_CB = its mean / SD over the pool CBs (identical CB grades to 75c).
Validation reps: all pool-receiver x pool-DB coverage reps (41) where the DB lined up at SLOT_CB or CB; recomputed:
closing over 0.7 s after the break, zone-matching cell (70), separation gained, completion when targeted.
Tests (player permutation): safeties-at-corner alone; CBs + safeties-at-corner pooled.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(131)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
Z45 = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cbs = base.index[base.nfl_position == "CB"]
ref = Z45.top.reindex(cbs).dropna(); MU, SD = ref.mean(), ref.std()
print(f"CB reference (within-year z of frozen drive): mean {MU:+.4f}, sd {SD:.4f}")
dbs = base.index[base.nfl_position.isin(["CB", "SS", "FS", "DB"])]
G = pd.DataFrame({"Drive+": 100 + 10 * (Z45.top.reindex(dbs) - MU) / SD}).dropna()
G["pos"] = base.nfl_position.reindex(G.index); G["name"] = base.display_name.reindex(G.index); G["class"] = base.draft_year.reindex(G.index)
G["tier"] = pd.cut(G["Drive+"], [-1e9, 90, 100, 110, 1e9], right=False, labels=["<90", "90-100", "100-110", "110+"])
print("graded DBs by position:", G.pos.value_counts().to_dict(), "| mean Drive+ by position:", G.groupby("pos")["Drive+"].mean().round(1).to_dict())

# ---------------- reps at corner / slot
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
al = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
al = al[al.season_type.isin(["REG", "POST"]) & (al.nullified != "Y") & al.lined_up_position_d.isin(["SLOT_CB", "CB"])].reset_index(drop=True)
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = al.idx_r.values, al.idx_d.values
vxr, vyr, sr, lr, pr = g(A, "vx"), g(A, "vy"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
vxd, vyd, sd = g(B, "vx"), g(B, "vy"), g(B, "s")
dist = np.hypot(g(A, "x_rel") - g(B, "x_rel"), g(A, "y_field") - g(B, "y_field"))
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]; r_ = np.arange(len(al))
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1); hb = a.max(1) > 0
at = lambda arr, j: np.where(hb & (j < T) & vv[r_, np.clip(j, 0, T - 1)], arr[r_, np.clip(j, 0, T - 1)], np.nan)
al["db_closing"] = (at(dist, jb) - at(dist, jb + 7)) / 0.7
w = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
pa = np.where(w.sum(1) >= 3, np.where(w, cosv, 0).sum(1) / np.maximum(w.sum(1), 1), np.nan)
al["cell"] = np.where(pa >= .7, np.where(al.team_coverage_man_zone == "ZONE_COVERAGE", "zone-matching", "man-matching"), "other")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
kr = pd.MultiIndex.from_arrays([al.game_id, al.play_id, al.nfl_id_r])
al["targeted"] = pp.target.reindex(kr).values; pres = pp.pass_result.reindex(kr).values
al["completion"] = np.where(al.targeted == 1, (pres == "C").astype(float), np.nan)
tot = al.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
al["wr_skill"] = ((al.nfl_id_r.map(tot["sum"]) - al.sep_gain_break.fillna(0)) / (al.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)).fillna(al.sep_gain_break.mean())
al["pos"] = al.nfl_id_d.map(base.nfl_position); al["is_s"] = (~al.pos.eq("CB")).astype(int)
al["dp"] = al.nfl_id_d.map((G["Drive+"] - 100) / 10)
al = al.dropna(subset=["dp"])
print(f"\nreps at CB/slot with a graded DB: {len(al)} (CBs {al[al.is_s == 0].nfl_id_d.nunique()}, safeties/DBs {al[al.is_s == 1].nfl_id_d.nunique()}, "
      f"safety reps {int(al.is_s.sum())}) | safety reps by cell: {al[al.is_s == 1].cell.value_counts().to_dict()}")
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def eff(d, y, ctx, n_perm=400):
    d = d.dropna(subset=[y]).copy()
    if d.nfl_id_d.nunique() < 8:
        return None
    f = f"{y} ~ dp + {ctx}"; m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues["dp"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"])
    return m.params["dp"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d), d.nfl_id_d.nunique()


print("\n=== Drive+ (per 10 points) when lined up at CB / slot ===")
for lab, d in [("safeties/DBs only", al[al.is_s == 1]), ("CBs only", al[al.is_s == 0]), ("pooled CB + safeties", al)]:
    for y, dd, ctx in [("db_closing", d[d.cell == "zone-matching"], "wr_skill + C(route_ran)" + (" + is_s" if lab.startswith("pooled") else "")),
                       ("sep_gain_break", d, "wr_skill + C(team_coverage_man_zone) + C(route_ran)" + (" + is_s" if lab.startswith("pooled") else "")),
                       ("completion", d[d.targeted == 1], "C(team_coverage_man_zone)" + (" + is_s" if lab.startswith("pooled") else ""))]:
        r = eff(dd, y, ctx)
        tag = "zone-matching closing" if y == "db_closing" else ("completion (targeted)" if y == "completion" else "sep gained (all)")
        print(f"  {lab:22s} {tag:22s} " + (f"{r[0]:+.3f} (perm p {r[1]:.3f}, reps {r[2]}, DBs {r[3]})" if r else "too few DBs"))
G["NFL reps at CB/slot"] = al.groupby("nfl_id_d").size()
G = G.sort_values("Drive+", ascending=False); G["Drive+"] = G["Drive+"].round(0)
G.to_csv(f"{ROOT}/results/76a_drive_plus_all_db.csv")
print("\ngrades by position and tier:\n", pd.crosstab(G.pos, G.tier).to_string())
print("\ntop 15 DBs:\n", G.head(15).to_string())
