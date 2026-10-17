import warnings, contextlib, io
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/Users/ahanjain/bdb_ssl/code/26_pressure_glm_matchups.py").read().split("# (2) matchups")[0])
n = d.groupby("nfl_id").size(); d["w_eq"] = 1.0 / d.nfl_id.map(n)
dd = d.dropna(subset=["max_depth_live"]).copy(); dd["big"] = dd.nfl_id.map(n) >= 50
G = lambda q: {"groups": pd.factorize(q.nfl_id)[0]}
for lab, q, f, w in [("snap-weighted, no weight ctrl", dd, "max_depth_live ~ z_hoop + edge + C(blitz) + C(down) + C(ytg_b)", None),
                     ("snap-weighted, players >=50 REG snaps", dd[dd.big], "max_depth_live ~ z_hoop + edge + C(blitz) + C(down) + C(ytg_b)", None),
                     ("EQUAL player weight, players >=50", dd[dd.big], "max_depth_live ~ z_hoop + edge + C(blitz) + C(down) + C(ytg_b)", "w_eq"),
                     ("EQUAL player weight + body weight", dd[dd.big], "max_depth_live ~ z_hoop + z_w + edge + C(blitz) + C(down) + C(ytg_b)", "w_eq")]:
    m = (smf.wls(f, data=q, weights=q[w]) if w else smf.ols(f, data=q)).fit(cov_type="cluster", cov_kwds=G(q))
    print(f"{lab:40s} b/SD {m.params['z_hoop']:+.3f} yd  p {m.pvalues['z_hoop']:.4f}  players {q.nfl_id.nunique()}")
# is the player-level link carried by low-volume players? correlation by snap-volume tercile
pm = dd[dd.big].groupby("nfl_id").agg(pen=("max_depth_live_oe", "mean"), n=("max_depth_live_oe", "size"), hoop=("z_hoop", "first"))
pm["tert"] = pd.qcut(pm.n, 3, labels=["low", "mid", "high"])
print("player-level r(hoop, penetration_oe) by snap-volume tercile:", pm.groupby("tert").apply(lambda g: round(g.hoop.corr(g.pen), 2), include_groups=False).to_dict(),
      "| r(hoop, n snaps) = %.2f" % pm.hoop.corr(np.log(pm.n)))

# true 1v1 matchups: pool OL and pool DL on the same play, aligned within 1.5 yd laterally at the snap
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position"]]
mm = meta.merge(pl, on="nfl_id")
mm["role"] = np.where(mm.nfl_position.isin(["T", "G", "C"]), "OL", np.where(mm.nfl_position.isin(["DE", "OLB", "DT", "NT"]), "DL", None))
mm = mm[mm.role.notna()]
yi = json_f = __import__("json").load(open(f"{ROOT}/data/seq/vocab.json"))["feats"].index("y_field")
mm["y_snap"] = np.asarray(X[mm.idx.values, 10, yi], dtype=np.float32)
a = mm[mm.role == "OL"][["game_id", "play_id", "nfl_id", "y_snap", "lined_up_position", "play_type"]]
b = mm[mm.role == "DL"][["game_id", "play_id", "nfl_id", "y_snap", "lined_up_position"]]
j = a.merge(b, on=["game_id", "play_id"], suffixes=("_ol", "_dl"))
j["dy"] = (j.y_snap_ol - j.y_snap_dl).abs()
print(f"\nOL-DL pairs on the same play: {len(j)}; aligned within 1.5 yd at the snap: {(j.dy <= 1.5).sum()} "
      f"(dropbacks {((j.dy <= 1.5) & (j.play_type == 'DROPBACK')).sum()}); unique OL {j[j.dy <= 1.5].nfl_id_ol.nunique()}, unique DL {j[j.dy <= 1.5].nfl_id_dl.nunique()}")
print("by alignment:", j[j.dy <= 1.5].groupby(["lined_up_position_ol", "lined_up_position_dl"]).size().sort_values(ascending=False).head(6).to_dict())
