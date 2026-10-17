"""Film-room data: real NFL coverage reps of top-third vs bottom-third Drive+ CORNERS, by route and coverage cell.

Reps: registered 3-yd coverage reps (41), CBs only, cells zone-matching / man-matching (70's definition), Drive+ thirds
among CBs with reps. Per rep, frames from the snap to 1.5 s after the receiver's break (10 Hz, standardized frame:
offense moving up the field; mirrored so the receiver always aligns on the right side of the screen):
  wr      : x (lateral, screen), y (yards past the LOS);  db: x, y, o (body facing, deg clockwise from screen-up), s (speed)
  jb      : break frame index, jt: throw frame index (or null), sep: separation from break-1.0 s to break+1.5 s
  gain    : separation change from 0.3 s before the break to 1.0 s after (what makes a rep "typical" for its group)
Meta: CB / receiver names, Drive+, season, week, team, shell, targeted / completion. Output results/viz/rep_film.json
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
src = open(f"{ROOT}/code/78_grade_card_all_db.py").read(); ns = {}
exec(src.split('pd.set_option("display.width", 250)')[0], ns)
R = ns["build"]("coverage_reps"); X, V, F, PRE = ns["X"], ns["V"], ns["F"], ns["PRE"]
GR = ns["GR"]
R = R[(R.pos == "CB") & R.cell.isin(["zone-matching", "man-matching"]) & ~R.route_ran.isin(["FLAT", "SCREEN", "WHEEL"])].reset_index(drop=True)
cbids = R.nfl_id_d.unique(); dp = GR.drive_plus.reindex(cbids)
q = pd.qcut(dp, 3, labels=["bottom", "middle", "top"])
R["grp"] = R.nfl_id_d.map(q); R = R[R.grp.isin(["top", "bottom"])].reset_index(drop=True)
print("CB thirds cut points:", dp.quantile([1 / 3, 2 / 3]).round(1).to_dict(), "| reps:", R.groupby(["cell", "grp"]).size().to_dict())
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = R.idx_r.values, R.idx_d.values
xr, yr, sr, lr, pr = g(A, "x_rel"), g(A, "y_field"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
xd, yd, sd = g(B, "x_rel"), g(B, "y_field"), g(B, "s")
sor, cor, sod, cod = g(A, "sin_o"), g(A, "cos_o"), g(B, "sin_o"), g(B, "cos_o")
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
jb = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1).argmax(1)
dist = np.hypot(xr - xd, yr - yd)
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")[["game_id", "play_id", "nfl_id", "team_coverage_type"]]
shell = meta.set_index(["game_id", "play_id", "nfl_id"]).team_coverage_type
games = pd.read_parquet(f"{ROOT}/data/raw/games.parquet").set_index("game_id")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
players = pd.read_parquet(f"{ROOT}/data/raw/players.parquet").set_index("nfl_id")
SHELL = {"COVER_0_MAN": "Cover 0", "COVER_1_MAN": "Cover 1", "COVER_2_MAN": "2-Man", "COVER_2_ZONE": "Cover 2", "COVER_3_ZONE": "Cover 3",
         "COVER_4_ZONE": "Cover 4", "COVER_6_ZONE": "Cover 6", "PREVENT": "Prevent", "MISC": "Other"}
r2 = lambda v: None if not np.isfinite(v) else round(float(v), 2)
out = []
for i, r in R.iterrows():
    j_end = min(jb[i] + 15, T - 1)
    fr = [j for j in range(PRE, j_end + 1) if vv[i, j]]
    if len(fr) < 15 or jb[i] - PRE < 5 or not vv[i, min(jb[i] + 10, T - 1)]:
        continue
    j0 = fr[0]; j1 = fr[-1]
    if j1 - j0 + 1 != len(fr):                       # keep only contiguous frames
        continue
    flip = -1.0 if yr[i, PRE] > 0 else 1.0            # receiver on the screen's right
    sx = lambda yy: -yy * flip                        # screen x = -(y_field) so the QB's right is screen right
    r1 = lambda v: round(float(v), 1)
    def track(xx, yy, so=None, co=None, s=None):
        t = {"x": [r1(sx(yy[j])) for j in range(j0, j1 + 1)], "y": [r1(xx[j]) for j in range(j0, j1 + 1)]}
        if so is not None:
            # facing unit vector in the standardized frame = (downfield, y_field) = (sin o, cos o); on screen
            # (up = downfield, right = -y_field * flip) it is (-flip * cos o, sin o) -> clockwise angle from screen-up
            t["o"] = [int(round(v)) for v in np.degrees(np.arctan2(-flip * co[j0:j1 + 1], so[j0:j1 + 1]))]
            t["s"] = [r1(s[j]) for j in range(j0, j1 + 1)]
        return t
    jt = int((pr[i] >= .5).argmax()) if (pr[i] >= .5).any() else None
    sep = [r2(dist[i, j]) if (0 <= j < T and vv[i, j]) else None for j in range(jb[i] - 10, jb[i] + 16)]
    gain = dist[i, jb[i] + 10] - dist[i, jb[i] - 3] if vv[i, jb[i] - 3] else np.nan
    key = (r.game_id, r.play_id, r.nfl_id_d); g_ = games.loc[r.game_id] if r.game_id in games.index else None
    out.append({"cell": r.cell, "grp": r.grp, "route": r.route_ran, "cb": int(r.nfl_id_d), "cbName": GR.display_name.get(r.nfl_id_d, "?"),
                "dp": round(float(GR.drive_plus.get(r.nfl_id_d)), 1), "wrName": players.display_name.get(r.nfl_id_r, "?"),
                "wrPos": players.nfl_position.get(r.nfl_id_r, ""), "season": int(r.season), "week": None if g_ is None else int(g_.week),
                "team": pp.team_abbr.get(key, ""), "shell": SHELL.get(shell.get(key, ""), "") , "targeted": int(r.targeted == 1) if pd.notna(r.targeted) else 0,
                "complete": None if pd.isna(r.completion) else int(r.completion), "jb": int(jb[i] - j0), "jt": None if jt is None or jt < j0 or jt > j1 else int(jt - j0),
                "wr": track(xr[i], yr[i]), "db": track(xd[i], yd[i], sod[i], cod[i], sd[i]), "sep": [None if v is None else round(v, 1) for v in sep], "gain": r2(gain)})
print("reps exported:", len(out))
cnt = pd.DataFrame([{k_: o[k_] for k_ in ("cell", "grp", "route")} for o in out]).value_counts().unstack("grp").fillna(0).astype(int)
print(cnt.to_string())
js = json.dumps({"reps": out, "cuts": {k_: round(float(v), 1) for k_, v in dp.quantile([1 / 3, 2 / 3]).items()}}, separators=(",", ":"), allow_nan=False)
open(f"{ROOT}/results/viz/rep_film.json", "w").write(js); print("bytes", len(js))
