"""Outside-data extension (nflverse play-by-play + NGS receiving aggregates), kept separate from the registered tests.

Joins: BDB game_id == nflverse old_game_id, play_id == play_id. Adds to every throw-side row (76b/77: pass attempt x graded
pool CB on a clear side, 22,312 rows):
  cp / cpoe       nflverse completion probability for the throw and completion over expected (ALL targets, not only pool)
  air_yards       exact air yards (replaces short/deep parsed from text)
  rcv_sep         targeted receiver's season NGS average separation (ngs_api receiving, target-weighted) -- receiver quality
  rcv_cushion     his season average cushion
  qb_cpoe_loo     passer's season CPOE excluding this play -- QB quality
Tests (toward his side vs away, defense + offense team-season FE, player-permutation p), zone and all coverage:
  T1 nflverse CPOE as the outcome (completion minus cp) on all throws
  T2 completion with cp + receiver + QB controls          T3 EPA with the same controls
  T4 exact air yards toward vs away (the depth shift)     T5 the raw completion contrast on the same matched rows (reference)
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; NV = "/Users/ahanjain/nfl_data/raw"
rng = np.random.default_rng(241)
ns = {}; exec(open(f"{ROOT}/code/77_epa_decomposition.py").read().split("def analyze")[0], ns)
d = ns["d"].copy()
COLS = ["old_game_id", "play_id", "season", "week", "cp", "cpoe", "air_yards", "complete_pass", "epa", "receiver_player_id", "receiver_player_name", "passer_player_id", "pass_location", "yards_after_catch"]
pbp = pd.concat([pd.read_parquet(f"{NV}/pbp/play_by_play_{s}.parquet", columns=COLS) for s in [2023, 2024, 2025]], ignore_index=True)
pbp["game_id"] = pbp.old_game_id.astype(int); pbp = pbp.drop_duplicates(["game_id", "play_id"])
# QB season CPOE, leave-one-play-out
q = pbp.dropna(subset=["cpoe", "passer_player_id"]).groupby(["passer_player_id", "season"]).cpoe.agg(["sum", "count"])
pbp = pbp.join(q, on=["passer_player_id", "season"])
pbp["qb_cpoe_loo"] = np.where(pbp["count"] > 1, (pbp["sum"] - pbp.cpoe.fillna(0)) / (pbp["count"] - 1), np.nan)
# NGS receiving: season averages per receiver, target-weighted over weekly rows
ngs = pd.concat([pd.read_parquet(f"{NV}/ngs_api/ngs_api_receiving_{s}_{t}.parquet") for s in [2023, 2024, 2025] for t in ["REG", "POST"]], ignore_index=True)
ngs = ngs[(ngs.week > 0) & ngs.targets.gt(0)]
w = ngs.targets.astype(float); ngs["ws"] = ngs.avgSeparation * w; ngs["wc"] = ngs.avgCushion * w
rs = ngs.groupby(["player_gsisId", "season"]).agg(ws=("ws", "sum"), wc=("wc", "sum"), tw=("targets", "sum")).reset_index()
rs["rcv_sep"] = rs.ws / rs.tw; rs["rcv_cushion"] = rs.wc / rs.tw
pbp = pbp.merge(rs[["player_gsisId", "season", "rcv_sep", "rcv_cushion", "tw"]].rename(columns={"player_gsisId": "receiver_player_id", "tw": "rcv_targets"}), on=["receiver_player_id", "season"], how="left")
d = d.merge(pbp[["game_id", "play_id", "cp", "cpoe", "air_yards", "complete_pass", "qb_cpoe_loo", "rcv_sep", "rcv_cushion", "rcv_targets", "receiver_player_name", "epa"]].rename(columns={"epa": "epa_nv"}), on=["game_id", "play_id"], how="left")
print(f"throw-side rows {len(d)} | matched to nflverse {d.cp.notna().sum() + d.cp.isna().sum() - d.complete_pass.isna().sum()} | with cp {d.cp.notna().sum()} | with receiver NGS season separation {d.rcv_sep.notna().sum()} | with QB CPOE {d.qb_cpoe_loo.notna().sum()}")
chk = d.dropna(subset=["cp", "completion"]); print(f"sanity: BDB completion vs nflverse complete_pass agree on {(chk.completion == chk.complete_pass).mean():.3f} of {len(chk)}; mean cp {chk.cp.mean():.3f} vs completion {chk.completion.mean():.3f}")
d["cpoe_nv"] = d.completion - d.cp
d["rcv_sep_c"] = d.rcv_sep - d.rcv_sep.mean(); d["qb_c"] = d.qb_cpoe_loo - d.qb_cpoe_loo.mean(); d["cp_c"] = d.cp - d.cp.mean(); d["air_c"] = d.air_yards - d.air_yards.mean()


class FWL2:
    def __init__(self, P, cont):
        cols = [pd.get_dummies(P[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in ["down", "dist_bin", "ytg_bin", "quarter", "def_ts", "off_ts", "rel", "dir"]]
        M = np.hstack([np.ones((len(P), 1))] + [c.values for c in cols] + [P[cont].values.astype(float)] if cont else [np.ones((len(P), 1))] + [c.values for c in cols])
        q_, r_ = np.linalg.qr(M); self.Q = q_[:, np.abs(np.diag(r_)) > 1e-8]
    def res(self, v): return v - self.Q @ (self.Q.T @ v)


def fit(fw, y_r, cols):
    Xr = np.column_stack([fw.res(c) for c in cols]); b, *_ = np.linalg.lstsq(Xr, y_r, rcond=None)
    e = y_r - Xr @ b; s2 = e @ e / (len(e) - fw.Q.shape[1] - Xr.shape[1]); C = s2 * np.linalg.inv(Xr.T @ Xr); return b, C


def analyze(P, y, cont=(), n_perm=500):
    P = P.dropna(subset=[y] + list(cont)).reset_index(drop=True); fw = FWL2(P, list(cont)); y_r = fw.res(P[y].values.astype(float))
    R = {r: (P.rel == r).values.astype(float) for r in ["toward", "away", "middle"]}
    def st(dp):
        b, C = fit(fw, y_r, [dp * R["toward"], dp * R["away"], dp * R["middle"]]); con = b[0] - b[1]
        return b, con, con / np.sqrt(C[0, 0] + C[1, 1] - 2 * C[0, 1]), b[0] / np.sqrt(C[0, 0])
    b, con, t, tt = st(P.dp.values); vals = P.drop_duplicates("nfl_id").set_index("nfl_id").dp; nc, nt = [], []
    for _ in range(n_perm):
        sp = pd.Series(rng.permutation(vals.values), index=vals.index); _, _, t_, tt_ = st(P.nfl_id.map(sp).values); nc.append(t_); nt.append(tt_)
    return {"toward": b[0], "away": b[1], "con": con, "p_con": (1 + (np.abs(nc) >= abs(t)).sum()) / (n_perm + 1), "p_tow": (1 + (np.abs(nt) >= abs(tt)).sum()) / (n_perm + 1), "n": len(P), "cbs": P.nfl_id.nunique()}


CTRL = ["cp_c", "rcv_sep_c", "qb_c"]
for lab, P in [("zone", d[d.team_coverage_man_zone == "ZONE_COVERAGE"]), ("all coverages", d)]:
    print(f"\n=== {lab}: per 10 Drive+ points, toward his side vs away (team-season FE; player-permutation p) ===")
    base_rows = P.dropna(subset=["cp", "rcv_sep", "qb_cpoe_loo"])
    for nm, y, cont, sc in [("T5 completion, raw (matched rows)", "completion", [], 100), ("T1 nflverse CPOE (completion − cp)", "cpoe_nv", [], 100),
                            ("T2 completion | cp, receiver sep, QB CPOE", "completion", CTRL, 100), ("T2b completion | + exact air yards", "completion", CTRL + ["air_c"], 100),
                            ("T3 EPA | cp, receiver sep, QB CPOE", "epa", CTRL, 1), ("T4 exact air yards (depth shift)", "air_yards", [], 1),
                            ("T4b receiver season NGS separation", "rcv_sep", [], 1)]:
        r = analyze(base_rows if nm.startswith(("T2", "T3", "T1", "T5")) else P, y, cont)
        print(f"  {nm:42s} toward {r['toward'] * sc:+.3f} | away {r['away'] * sc:+.3f} | contrast {r['con'] * sc:+.3f} (p {r['p_con']:.3f}; toward alone p {r['p_tow']:.3f}) | rows {r['n']}, CBs {r['cbs']}", flush=True)
d.to_parquet(f"{ROOT}/data/outcomes/93_throwside_nflverse.parquet", index=False)
