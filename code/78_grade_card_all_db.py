"""Drive+ grade card on the biggest reasonable sample: all graded DBs (CB/SS/FS/DB), all alignments, 2023-25 REG+POST.

Grades = the FROZEN file prereg/drive_plus_frozen_2026-10-12.csv. Rep samples (both players must be in the 510-player
pool -- only pool players are tracked):
  S1  3-yd reps (41, the registered definition), CBs only        -- the original card, as a check
  S2  3-yd reps, all graded DBs, any alignment                    -- main expanded card
  S3  4-yd reps (mean distance <= 4 yd), all graded DBs           -- largest sample (sensitivity)
Per tier: DBs with reps (CB / S), reps, targets, completion % allowed, EPA per target (offense), zone-matching closing
(adj for receiver skill + route), separation gained at break (adj). Continuous slopes per 10 Drive+ points (+ position
and man/zone controls) with player-permutation p; top-vs-bottom tier bootstrap (by player). Season breakdown.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(151)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
GR = GR[GR.nfl_position.isin(["CB", "SS", "FS", "DB"])]
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
TIERS = ["110+", "100-110", "90-100", "<90"]


def build(fname):
    al = pd.read_parquet(f"{ROOT}/data/outcomes/{fname}.parquet")
    al = al[al.season_type.isin(["REG", "POST"]) & (al.nullified != "Y") & al.nfl_id_d.isin(GR.index)].reset_index(drop=True)
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
    kr = pd.MultiIndex.from_arrays([al.game_id, al.play_id, al.nfl_id_r])
    al["targeted"] = pp.target.reindex(kr).values
    res = pp.pass_result.reindex(kr).values
    al["completion"] = np.where(al.targeted == 1, (res == "C").astype(float), np.nan)
    al["epa_t"] = np.where(al.targeted == 1, pp.expected_points_added.reindex(kr).values, np.nan)
    tot = al.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
    al["wr_skill"] = ((al.nfl_id_r.map(tot["sum"]) - al.sep_gain_break.fillna(0)) / (al.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)).fillna(al.sep_gain_break.mean())
    al["pos"] = al.nfl_id_d.map(GR.nfl_position); al["is_s"] = (al.pos != "CB").astype(int)
    al["dp"] = al.nfl_id_d.map((GR.drive_plus - 100) / 10); al["tier"] = al.nfl_id_d.map(GR.tier)
    z = al[al.cell == "zone-matching"].dropna(subset=["db_closing"])
    al.loc[z.index, "close_adj"] = smf.ols("db_closing ~ wr_skill + C(route_ran)", data=z).fit().resid + z.db_closing.mean()
    s = al.dropna(subset=["sep_gain_break"])
    al.loc[s.index, "sep_adj"] = smf.ols("sep_gain_break ~ wr_skill + C(team_coverage_man_zone) + C(route_ran)", data=s).fit().resid + s.sep_gain_break.mean()
    return al


CL = lambda d: pd.factorize(d.nfl_id_d)[0]


def slope(d, y, ctx, n_perm=500):
    d = d.dropna(subset=[y]).copy(); f = f"{y} ~ dp + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues["dp"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"])
    return m.params["dp"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), d.nfl_id_d.nunique()


def boot(d, col, n=2000):
    d = d.dropna(subset=[col]); gm = d.groupby("nfl_id_d")[col].agg(["sum", "size"])
    ih = [i for i in d[d.tier == "110+"].nfl_id_d.unique()]; il = [i for i in d[d.tier == "<90"].nfl_id_d.unique()]
    est = gm.loc[ih, "sum"].sum() / gm.loc[ih, "size"].sum() - gm.loc[il, "sum"].sum() / gm.loc[il, "size"].sum(); out = []
    for _ in range(n):
        h, l = gm.loc[rng.choice(ih, len(ih))].sum(), gm.loc[rng.choice(il, len(il))].sum()
        out.append(h["sum"] / h["size"] - l["sum"] / l["size"])
    return est, np.percentile(out, [2.5, 97.5])


pd.set_option("display.width", 250)
reps3, reps4 = build("coverage_reps"), build("coverage_reps_4yd")
for lab, d in [("S1  3-yd reps, CBs only (original card)", reps3[reps3.pos == "CB"]), ("S2  3-yd reps, all graded DBs, any alignment", reps3),
               ("S3  4-yd reps, all graded DBs, any alignment (largest)", reps4)]:
    t = d[d.targeted == 1]
    print(f"\n==================== {lab} ====================")
    print(f"DBs with reps {d.nfl_id_d.nunique()} (CB {d[d.pos == 'CB'].nfl_id_d.nunique()}, S/DB {d[d.pos != 'CB'].nfl_id_d.nunique()}) | reps {len(d)} | targets {len(t)} | "
          f"by season: reps {d.season.value_counts().sort_index().to_dict()}, DBs {d.groupby('season').nfl_id_d.nunique().to_dict()}, targets {t.season.value_counts().sort_index().to_dict()}")
    rows = []
    for tr in TIERS:
        x, xt = d[d.tier == tr], t[t.tier == tr]
        rows.append({"Drive+ tier": tr, "DBs with reps": x.nfl_id_d.nunique(), "(CB / S)": f"{x[x.pos == 'CB'].nfl_id_d.nunique()} / {x[x.pos != 'CB'].nfl_id_d.nunique()}",
                     "reps": len(x), "targets": len(xt), "DBs targeted": xt.nfl_id_d.nunique(), "completion % allowed": 100 * xt.completion.mean(),
                     "EPA per target": xt.epa_t.mean(), "zone-match closing (adj)": x.close_adj.mean(), "sep gained at break (adj)": x.sep_adj.mean()})
    Tb = pd.DataFrame(rows).set_index("Drive+ tier")
    print(Tb.round(2).to_string())
    pos_ctl = " + is_s" if d.is_s.nunique() > 1 else ""
    out = []
    for nm, dd, y, ctx, sc in [("completion % (per 10 pts)", t, "completion", "C(team_coverage_man_zone)" + pos_ctl, 100),
                               ("EPA per target", t, "epa_t", "C(team_coverage_man_zone)" + pos_ctl, 1),
                               ("EPA per target, zone", t[t.team_coverage_man_zone == "ZONE_COVERAGE"], "epa_t", "1" + pos_ctl, 1),
                               ("zone-match closing", d[d.cell == "zone-matching"], "db_closing", "wr_skill + C(route_ran)" + pos_ctl, 1),
                               ("sep gained at break", d, "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)" + pos_ctl, 1)]:
        b, p, n = slope(dd, y, ctx)
        out.append(f"{nm}: {b * sc:+.3f} (perm p {p:.3f}, {n} DBs)")
    print("continuous, per 10 Drive+ points:\n   " + "\n   ".join(out))
    for nm, col, sc in [("completion % allowed", "completion", 100), ("EPA per target", "epa_t", 1)]:
        e, ci = boot(t, col)
        print(f"110+ minus <90, {nm}: {e * sc:+.2f} [{ci[0] * sc:+.2f}, {ci[1] * sc:+.2f}] (player bootstrap 95% CI)")
    Tb.to_csv(f"{ROOT}/results/78_grade_card_{lab.split()[0]}.csv")
