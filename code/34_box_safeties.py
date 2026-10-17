"""Box defenders as the linebacker stand-in: DBs (SS/FS/CB/DB) and the few ILB/MLB aligned off the ball in the box
(lined_up_position MLB / OLB / BOX_SAFETY) on run plays.

Game (post-handoff, defense attacks toward -x):
  trigger_time   handoff -> first frame moving downhill (vx <= -2 yd/s)
  downhill_speed max downhill speed (-vx) within 1.5 s of the handoff
  fit_depth      x relative to the LOS 1.5 s after the handoff (smaller = closer to the line)
  pursuit_bend   peak |lateral accel| at s > 3 within 2.5 s of the handoff
Outcomes (REG+POST): tackle (solo or assist) and TFL over expected (leave-player-out cell: alignment x down x
  distance bucket); on his tackles, yards gained by the offense (lower = better).
Combine (DB drills): "DB turn speed" = duration-weighted speed through turn segments (same definition as hoop
  turn speed) averaged over W drill, box drill, Teryl Austin 1/2 and backpedal-90-break; "break burst" = speed
  gained in the 0.5 s after the slowest point following the backpedal, in the backpedal 45 / 90 drills;
  stopwatch 3-cone and shuttle for comparison.
"""
import json, hashlib, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet")[["nfl_id", "nfl_position", "display_name"]]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
m = meta.merge(pl, on="nfl_id")
sel = m[(m.play_type == "RUN") & m.lined_up_position.isin(["MLB", "OLB", "BOX_SAFETY"]) &
        m.nfl_position.isin(["SS", "FS", "CB", "DB", "ILB", "MLB"])].copy()
print("box run snaps", len(sel), "players", sel.nfl_id.nunique(), flush=True)
idx = sel.idx.values
g = lambda n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
xr, vx, s, lat, ph = g("x_rel"), g("vx"), g("s"), g("a_lat"), g("post_handoff")
v = V[idx]
lat_s = lat.copy(); lat_s[:, 1:-1] = (lat[:, :-2] + lat[:, 1:-1] + lat[:, 2:]) / 3
hand = (ph > .5) & v
has_h = hand.any(1); h0 = np.where(has_h, hand.argmax(1), PRE)          # no handoff event (QB run etc.): use snap
cols = np.arange(X.shape[1])[None, :]
after = v & (cols >= h0[:, None])
w15 = after & (cols <= h0[:, None] + 15); w25 = after & (cols <= h0[:, None] + 25)
down = w25 & (vx <= -2.0)
sel["trigger_time"] = np.where(down.any(1), (down.argmax(1) - h0) / 10.0, np.nan)
sel["downhill_speed"] = np.where(w15.any(1), np.where(w15, -vx, -np.inf).max(1), np.nan)
i15 = np.minimum(h0 + 15, X.shape[1] - 1); r = np.arange(len(idx))
sel["fit_depth"] = np.where(v[r, i15], xr[r, i15], np.nan)
sel["pursuit_bend"] = np.where((w25 & (s > 3)).any(1), np.where(w25 & (s > 3), np.abs(lat_s), 0).max(1), np.nan)
o = pp.set_index(["game_id", "play_id", "nfl_id"])
key = pd.MultiIndex.from_arrays([sel.game_id, sel.play_id, sel.nfl_id])
for c in ["tackle", "assist", "tackle_for_loss", "yards_gained"]:
    sel[c] = pd.to_numeric(o[c].reindex(key).astype("object").replace({True: 1, False: 0}), errors="coerce").values
sel["tk"] = ((sel.tackle.fillna(0) + sel.assist.fillna(0)) > 0).astype(float)
sel["tfl"] = sel.tackle_for_loss.fillna(0).clip(0, 1)
sel["ytg_b"] = pd.cut(sel.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
ok = sel.season_type.isin(["REG", "POST"]) & (sel.nullified != "Y")


def loo_adjust(df, ycol, cells):
    key = df[cells].astype(str).agg("|".join, axis=1)
    tot = df.groupby(key)[ycol].transform("sum"); cnt = df.groupby(key)[ycol].transform("count")
    pk = key + "#" + df.nfl_id.astype(str)
    ptot = df.groupby(pk)[ycol].transform("sum"); pcnt = df.groupby(pk)[ycol].transform("count")
    return df[ycol] - np.where((cnt - pcnt) >= 20, (tot - ptot) / (cnt - pcnt), df[ycol].mean())


so = sel[ok].copy()
so["tk_oe"] = loo_adjust(so, "tk", ["lined_up_position", "down", "ytg_b"])
so["tfl_oe"] = loo_adjust(so, "tfl", ["lined_up_position", "down", "ytg_b"])
MIN = 30
gm = sel.groupby("nfl_id").agg(n=("idx", "size"), trigger_time=("trigger_time", "median"), downhill_speed=("downhill_speed", "mean"),
                               fit_depth=("fit_depth", "mean"), pursuit_bend=("pursuit_bend", lambda x: np.nanpercentile(x, 75)))
go = so.groupby("nfl_id").agg(n_o=("idx", "size"), tackles_oe=("tk_oe", "mean"), tfl_oe=("tfl_oe", "mean"))
tky = so[so.tk == 1].groupby("nfl_id").yards_gained.agg(["mean", "size"])
go["yds_on_his_tackles"] = tky["mean"].where(tky["size"] >= 8)
G = gm[gm.n >= MIN].join(go[go.n_o >= MIN], how="left")
print(f"players with >= {MIN} box run snaps: {len(G)} ({base.loc[G.index, 'nfl_position'].value_counts().to_dict()})")
# reliability of game measures (halves of games)
sel["half"] = sel.game_id.map(lambda x: int(hashlib.md5(f"half-{x}".encode()).hexdigest(), 16) % 2)
so["half"] = so.game_id.map(lambda x: int(hashlib.md5(f"half-{x}".encode()).hexdigest(), 16) % 2)
for c, df_, agg in [("trigger_time", sel, "median"), ("downhill_speed", sel, "mean"), ("fit_depth", sel, "mean"), ("tk_oe", so, "mean"), ("tfl_oe", so, "mean")]:
    w = df_.groupby(["nfl_id", "half"])[c].agg([agg, "size"]); w = w[w["size"] >= MIN // 2][agg].unstack().dropna()
    rr = np.corrcoef(w[0], w[1])[0, 1]
    print(f"  split-half reliability {c:15s} {2 * rr / (1 + rr):.2f} (n {len(w)})")
# validate game mechanics against outcomes (does a fast trigger / shallow fit go with more tackles?)
print("\nmechanics -> outcomes (player level):")
for a in ["trigger_time", "downhill_speed", "fit_depth", "pursuit_bend"]:
    for b in ["tackles_oe", "tfl_oe", "yds_on_his_tackles"]:
        d = G[[a, b]].dropna()
        if len(d) >= 15:
            rr, pv = stats.pearsonr(d[a], d[b]); print(f"  {a:15s} -> {b:18s} r {rr:+.2f} (p {pv:.3f}, n {len(d)})")

# ---- combine DB drills
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
TURN = ["W_DRILL", "BOX_DRILL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL"]
BREAK = ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL"]
c = ct[(ct.entity_type == "PLAYER") & (ct.drill_type == "SKILL_DRILLS_DB") & ct.drill_name.isin(TURN + BREAK)].sort_values(["event_id", "time"])
import sys; sys.path.insert(0, f"{ROOT}/prereg")
from hoop_turn_speed import attempt_value
rows = []
for eid, gg in c.groupby("event_id"):
    if len(gg) < 8:
        continue
    t = (gg.time - gg.time.iat[0]).dt.total_seconds().values; sp = gg.s.values.astype(float)
    rec = {"nfl_id": gg.nfl_id.iat[0], "drill": gg.drill_name.iat[0]}
    if rec["drill"] in TURN:
        rec["turn_speed"] = attempt_value(t, sp, gg.dir.values.astype(float))
    if rec["drill"] in BREAK:
        pk = int(np.argmax(sp[: max(3, len(sp) // 2)]))                       # backpedal speed peak (first half)
        lo = pk + int(np.argmin(sp[pk:])) if pk < len(sp) - 1 else pk          # slowest point = the transition
        hi = min(lo + 5, len(sp) - 1)
        rec["break_burst"] = sp[hi] - sp[lo]
    rows.append(rec)
C = pd.DataFrame(rows)
z = lambda x: (x - x.mean()) / x.std()
ts = C.dropna(subset=["turn_speed"]).groupby(["nfl_id", "drill"]).turn_speed.mean().unstack().apply(z)
bb = C.dropna(subset=["break_burst"]).groupby(["nfl_id", "drill"]).break_burst.mean().unstack().apply(z)
CB = pd.DataFrame({"db_turn_speed": ts.mean(1), "db_break_burst": bb.mean(1)})
CB["three_cone"] = base.three_cone; CB["short_shuttle"] = base.short_shuttle
print(f"\ncombine DB drill coverage among box players: turn {CB.db_turn_speed.reindex(G.index).notna().sum()}, burst {CB.db_break_burst.reindex(G.index).notna().sum()}, "
      f"3-cone {CB.three_cone.reindex(G.index).notna().sum()}, shuttle {CB.short_shuttle.reindex(G.index).notna().sum()}")
rows = []
for cm in CB.columns:
    for gmet in ["trigger_time", "downhill_speed", "fit_depth", "pursuit_bend", "tackles_oe", "tfl_oe", "yds_on_his_tackles"]:
        d = pd.concat([CB[cm], G[gmet], base.combine_weight], axis=1).reindex(G.index).dropna()
        if len(d) < 15:
            continue
        rr, pv = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
        Z = np.column_stack([np.ones(len(d)), d.combine_weight])
        rx = d.iloc[:, 0] - Z @ np.linalg.lstsq(Z, d.iloc[:, 0], rcond=None)[0]; ry = d.iloc[:, 1] - Z @ np.linalg.lstsq(Z, d.iloc[:, 1], rcond=None)[0]
        rows.append({"combine": cm, "game": gmet, "n": len(d), "r": rr, "p": pv, "r_partial_weight": np.corrcoef(rx, ry)[0, 1]})
R = pd.DataFrame(rows)
R.to_csv(f"{ROOT}/results/34_box_safeties.csv", index=False)
pd.set_option("display.width", 200)
print("\ncombine -> box run defense (players with >= %d box run snaps):" % MIN)
print(R.round(3).to_string(index=False))
