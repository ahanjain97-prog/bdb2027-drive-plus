"""Throws TOWARD vs AWAY from the high-Drive+ corner: does the defense allow less EPA on throws to his side?

Unit: a pass attempt (C/I/IN) x a graded pool CB on defense, lined up at CB or SLOT_CB, whose side at the snap is clear
(standardized y_field >= +4 = the QB's left, <= -4 = the QB's right). Throw direction from the play description
("pass short/deep left|middle|right"), validated against where targeted pool receivers are 1 s after the throw.
  rel = toward (throw to his side) | away (opposite side) | middle
Model: y ~ Drive+ x rel + rel + down/distance/field/quarter + defense team-season FE + offense team-season FE.
  The Drive+ slope on AWAY throws is a built-in placebo (he is not involved); the claim is toward < away.
Outcomes (offense perspective): EPA, completion, success. Split: all, zone, man. Player-permutation p (shuffle Drive+
across CBs; FWL on a precomputed QR). Also: avoidance (is a high-Drive+ CB's side thrown at less?).
"""
import json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(137)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
G = pd.read_csv(f"{ROOT}/results/76a_drive_plus_all_db.csv", index_col=0) if False else None
Z45 = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cbs = base.index[base.nfl_position == "CB"]
ref = Z45.top.reindex(cbs).dropna(); DP = ((Z45.top.reindex(cbs) - ref.mean()) / ref.std()).dropna()      # Drive+ in SD units (10 pts)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
games = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season"]]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")[["game_id", "play_id", "nfl_id", "idx"]]
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
passes = pp[pp.pass_result.isin(["C", "I", "IN"])].drop_duplicates(["game_id", "play_id"]).copy()
passes["dir"] = passes.play_description.str.extract(r"pass (?:incomplete )?(?:short|deep) (left|middle|right)", flags=re.I)[0].str.lower()
# ---- validate the side convention with targeted pool receivers
tr = pp[(pp.target == 1) & (pp.team_abbr == pp.possession_team)].merge(meta, on=["game_id", "play_id", "nfl_id"]).merge(passes[["game_id", "play_id", "dir"]], on=["game_id", "play_id"])
tr = tr[tr.dir.isin(["left", "right"])]
idx = tr.idx.values; pp_ = np.asarray(X[idx][:, :, F.index("post_pass")]); yf = np.asarray(X[idx][:, :, F.index("y_field")])
jr = np.where((pp_ >= .5).any(1), (pp_ >= .5).argmax(1), -1); j1 = np.clip(jr + 10, 0, X.shape[1] - 1)
ok = (jr >= 0) & V[idx, j1]
y1 = yf[np.arange(len(idx)), j1]
agree = ((y1 > 0) == (tr.dir.values == "left"))[ok]
print(f"direction check: targeted receivers 1 s after the throw on the side the description says, {agree.mean():.1%} of {ok.sum()} throws")
LEFT_IS_POS = agree.mean() >= 0.5
# ---- CB rows
d = pp[pp.nfl_id.isin(DP.index) & (pp.team_abbr != pp.possession_team) & pp.lined_up_position.isin(["CB", "SLOT_CB"])][["game_id", "play_id", "nfl_id", "lined_up_position"]]
d = d.merge(passes[["game_id", "play_id", "dir", "team_abbr", "possession_team", "down", "yards_to_go", "yardline_side", "yardline_number", "quarter",
                    "expected_points_added", "pass_result", "team_coverage_man_zone"]].rename(columns={"team_abbr": "_t"}), on=["game_id", "play_id"])
d = d.merge(meta, on=["game_id", "play_id", "nfl_id"]).merge(games, on="game_id")
ys = np.asarray(X[d.idx.values][:, PRE, F.index("y_field")]); vs = V[d.idx.values, PRE]
d["y_snap"] = np.where(vs, ys, np.nan)
side_pos = "left" if LEFT_IS_POS else "right"; side_neg = "right" if LEFT_IS_POS else "left"
d["side"] = np.where(d.y_snap >= 4, side_pos, np.where(d.y_snap <= -4, side_neg, None))
d = d[d.side.notna() & d.dir.notna()].copy()
d["rel"] = np.where(d.dir == "middle", "middle", np.where(d.dir == d.side, "toward", "away"))
d["def_team"] = np.where(d._t == d.possession_team, None, d._t)
dteam = pp[["game_id", "play_id", "nfl_id", "team_abbr"]].rename(columns={"team_abbr": "def_team2"})
d = d.merge(dteam, on=["game_id", "play_id", "nfl_id"], how="left")
d["def_ts"] = d.def_team2 + d.season.astype(str); d["off_ts"] = d.possession_team + d.season.astype(str)
d["ytg_bin"] = pd.cut(np.where(d.yardline_side == d.possession_team, 100 - d.yardline_number, d.yardline_number), [-1, 10, 20, 40, 60, 80, 101]).astype(str)
d["dist_bin"] = pd.cut(d.yards_to_go, [0, 2, 5, 8, 10, 15, 99]).astype(str)
d["epa"] = d.expected_points_added; d["completion"] = (d.pass_result == "C").astype(float); d["success"] = (d.epa > 0).astype(float)
d["dp"] = d.nfl_id.map(DP); d = d.reset_index(drop=True)
print(f"rows (pass attempt x graded pool CB with a clear side): {len(d)}, CBs {d.nfl_id.nunique()}, plays {d[['game_id', 'play_id']].drop_duplicates().shape[0]}")
print("throws by rel:", d.rel.value_counts().to_dict(), "| by coverage:", d.team_coverage_man_zone.value_counts().to_dict())


class FWL:
    def __init__(self, P):
        cols = [pd.get_dummies(P[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in ["down", "dist_bin", "ytg_bin", "quarter", "def_ts", "off_ts", "rel", "dir"]]
        M = np.hstack([np.ones((len(P), 1))] + [c.values for c in cols])
        q, r = np.linalg.qr(M); self.Q = q[:, np.abs(np.diag(r)) > 1e-8]
    def res(self, v):
        return v - self.Q @ (self.Q.T @ v)


def fit(fw, y_r, cols):
    Xr = np.column_stack([fw.res(c) for c in cols]); b, *_ = np.linalg.lstsq(Xr, y_r, rcond=None)
    e = y_r - Xr @ b; s2 = e @ e / (len(e) - fw.Q.shape[1] - Xr.shape[1]); C = s2 * np.linalg.inv(Xr.T @ Xr)
    return b, C


def analyze(P, y, n_perm=1000):
    P = P.dropna(subset=[y]).reset_index(drop=True); fw = FWL(P); y_r = fw.res(P[y].values.astype(float))
    R = {r: (P.rel == r).values.astype(float) for r in ["toward", "away", "middle"]}
    def stats(dp):
        b, C = fit(fw, y_r, [dp * R["toward"], dp * R["away"], dp * R["middle"]])
        con = b[0] - b[1]; se = np.sqrt(C[0, 0] + C[1, 1] - 2 * C[0, 1])
        return b, b[0] / np.sqrt(C[0, 0]), con, con / se
    dp = P.dp.values; b, t_tow, con, t_con = stats(dp)
    vals = P.drop_duplicates("nfl_id").set_index("nfl_id").dp; nt, nc = [], []
    for _ in range(n_perm):
        sp = pd.Series(rng.permutation(vals.values), index=vals.index); _, a1, _, a2 = stats(P.nfl_id.map(sp).values); nt.append(a1); nc.append(a2)
    p_t = (1 + (np.abs(nt) >= abs(t_tow)).sum()) / (n_perm + 1); p_c = (1 + (np.abs(nc) >= abs(t_con)).sum()) / (n_perm + 1)
    return {"toward": b[0], "away": b[1], "middle": b[2], "toward-away": con, "perm p toward": p_t, "perm p toward-away": p_c, "rows": len(P), "CBs": P.nfl_id.nunique()}


rows = []
for cov_lab, P in [("all", d), ("zone", d[d.team_coverage_man_zone == "ZONE_COVERAGE"]), ("man", d[d.team_coverage_man_zone == "MAN_COVERAGE"])]:
    for y in ["epa", "completion", "success"]:
        r = analyze(P, y); r.update({"coverage": cov_lab, "outcome": y}); rows.append(r)
        print(f"  {cov_lab:5s} {y:10s} Drive+ slope per 10 pts: toward {r['toward']:+.4f} | away {r['away']:+.4f} | middle {r['middle']:+.4f} || "
              f"toward-away {r['toward-away']:+.4f} (perm p {r['perm p toward-away']:.3f}); toward alone perm p {r['perm p toward']:.3f} | rows {r['rows']}, CBs {r['CBs']}", flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/76b_toward_away.csv", index=False)
# ---- avoidance: is his side thrown at less?
A = d[d.rel.isin(["toward", "away"])].copy(); A["tow"] = (A.rel == "toward").astype(float)
fw = FWL(A.assign(rel="x")); y_r = fw.res(A.tow.values)
b, C = fit(fw, y_r, [A.dp.values]); t0 = b[0] / np.sqrt(C[0, 0]); vals = A.drop_duplicates("nfl_id").set_index("nfl_id").dp; null = []
for _ in range(1000):
    sp = pd.Series(rng.permutation(vals.values), index=vals.index); bb, CC = fit(fw, y_r, [A.nfl_id.map(sp).values]); null.append(bb[0] / np.sqrt(CC[0, 0]))
print(f"\navoidance: share of left/right throws going to his side {A.tow.mean():.3f}; per 10 Drive+ points {b[0]:+.4f} (perm p {(1 + (np.abs(null) >= abs(t0)).sum()) / 1001:.3f})")
