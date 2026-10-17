"""Step 1: the in-game version of the transition drill, measured on EVERY CB dropback snap (no receiver needed).

Frame of reference (01_build_sequences): offense moves toward +x, defenders line up at x > 0 relative to the LOS,
so a CB backpedals toward +x and drives downhill toward -x. Body orientation o is tracked in games; cos_od = cos(o - dir).
Per snap (frames after the snap, until the play is dead):
  backpedal   >= 3 consecutive frames with speed >= 1.5 yd/s and cos_od < -0.5 (moving opposite to where he faces)
  turn        first frame within 1.2 s after the backpedal ends with cos_od > 0.3 and speed >= 1 (facing his travel again)
  drive       from the turn: running max speed until it falls 0.5 yd/s below the max (<= 1.5 s)
  metrics     g_drive_top, g_drive_gain_07 (speed 0.7 s after the turn minus at the turn), g_trans_min (slowest speed
              between backpedal end and turn), g_trans_time, g_bp_speed (speed at the end of the backpedal),
              g_downhill (mean vx over the drive < -1: driving toward the LOS), g_post_throw (turn after the throw)
Break on the throw (snaps with a throw): speed at the throw, max speed within 1.5 s after it, speed gained in 0.5 s,
  heading change in 0.5 s.
Per CB: means over events (>= MIN events) by game halves -> split-half reliability; correlation with the frozen combine
drive speed; by coverage (man / zone) and by drive type (downhill / other, pre / post throw).
Output data/outcomes/cb_game_transitions.parquet (event level) and results/60_*.csv
"""
import hashlib, json, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
cb = meta[(meta.play_type == "DROPBACK") & meta.nfl_id.isin(base.index[base.nfl_position == "CB"])].reset_index(drop=True)
idx = cb.idx.values
get = lambda n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
s, cod, vx, vy, pp_, pd_ = get("s"), get("cos_od"), get("vx"), get("vy"), get("post_pass"), get("post_dead")
v = V[idx]; T = X.shape[1]
print("CB dropback snaps", len(cb), "CBs", cb.nfl_id.nunique(), flush=True)
ev, thr = [], []
for i in range(len(cb)):
    live = v[i] & (np.arange(T) >= PRE) & (pd_[i] < .5)
    bp = live & (s[i] >= 1.5) & (cod[i] < -0.5)
    # break on the throw
    jf = np.where(live & (pp_[i] > .5))[0]
    if len(jf):
        f0 = jf[0]; w = np.arange(f0, min(f0 + 15, T)); w = w[v[i, w]]
        if len(w) >= 6:
            h0 = np.arctan2(vx[i, f0], vy[i, f0]); h5 = np.arctan2(vx[i, w[min(5, len(w) - 1)]], vy[i, w[min(5, len(w) - 1)]])
            thr.append({"row": i, "t_throw": (f0 - PRE) / 10, "speed_at_throw": s[i, f0], "max_speed_after_throw": s[i, w].max(),
                        "gain_05": s[i, w[min(5, len(w) - 1)]] - s[i, f0],
                        "heading_change_05": np.degrees(abs(np.arctan2(np.sin(h5 - h0), np.cos(h5 - h0))))})
    j = PRE
    while j < T - 3:
        if not bp[j]:
            j += 1; continue
        e = j
        while e + 1 < T and bp[e + 1]:
            e += 1
        if e - j + 1 >= 3:
            cand = np.where(live[e + 1:e + 13] & (cod[i, e + 1:e + 13] > 0.3) & (s[i, e + 1:e + 13] >= 1.0))[0]
            if len(cand):
                jt = e + 1 + cand[0]
                top, kk = s[i, jt], jt
                while kk + 1 < min(jt + 15, T) and v[i, kk + 1]:
                    kk += 1
                    if s[i, kk] > top:
                        top = s[i, kk]
                    elif s[i, kk] < top - 0.5:
                        break
                j7 = min(jt + 7, T - 1)
                dv = np.arange(jt, kk + 1)
                ev.append({"row": i, "k_turn": (jt - PRE) / 10, "g_bp_speed": s[i, e], "g_trans_min": s[i, e:jt + 1].min(),
                           "g_trans_time": (jt - e) / 10, "g_drive_top": top, "g_drive_gain_07": s[i, j7] - s[i, jt] if v[i, j7] else np.nan,
                           "g_downhill": float(vx[i, dv].mean() < -1.0), "g_post_throw": float(pp_[i, jt] > .5)})
        j = e + 1
E = pd.DataFrame(ev).merge(cb[["game_id", "play_id", "nfl_id", "season", "season_type", "team_coverage_man_zone"]].reset_index().rename(columns={"index": "row"}), on="row")
TH = pd.DataFrame(thr).merge(cb[["game_id", "play_id", "nfl_id", "season", "season_type", "team_coverage_man_zone"]].reset_index().rename(columns={"index": "row"}), on="row")
E.to_parquet(f"{ROOT}/data/outcomes/cb_game_transitions.parquet", index=False); TH.to_parquet(f"{ROOT}/data/outcomes/cb_game_throw_breaks.parquet", index=False)
print(f"transition events {len(E)} on {E.row.nunique()} snaps ({E.row.nunique() / len(cb):.0%} of CB dropbacks); downhill share {E.g_downhill.mean():.2f}, "
      f"after the throw {E.g_post_throw.mean():.2f}; break-on-throw snaps {len(TH)}")
print(E[["g_bp_speed", "g_trans_min", "g_trans_time", "g_drive_top", "g_drive_gain_07"]].describe().round(2).T[["mean", "50%", "std"]].to_string())

drive = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))
half = lambda g: int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2
E["half"] = E.game_id.map(half); TH["half"] = TH.game_id.map(half)
MIN = 20
rows = []
def player_metric(df, col, label, sub=None):
    d = df if sub is None else df[sub(df)]
    g = d.groupby("nfl_id")[col]; n = g.size()
    val = g.mean()[n >= MIN]
    h = d.groupby(["nfl_id", "half"])[col].agg(["mean", "size"]); h = h[h["size"] >= MIN // 2]["mean"].unstack().dropna()
    rel = np.nan
    if len(h) >= 12:
        r = np.corrcoef(h[0], h[1])[0, 1]; rel = 2 * r / (1 + r)
    j = pd.concat([val.rename("g"), drive.rename("c")], axis=1).dropna()
    rows.append({"metric": label, "CBs": len(val), "events_median": int(n[n >= MIN].median()) if (n >= MIN).any() else 0,
                 "split_half_rel": rel, "r_with_combine_drive": j.corr().iloc[0, 1] if len(j) >= 10 else np.nan, "n_both": len(j)})
for col in ["g_drive_top", "g_drive_gain_07", "g_trans_min", "g_trans_time", "g_bp_speed"]:
    player_metric(E, col, f"{col} (all transitions)")
player_metric(E, "g_drive_top", "g_drive_top | downhill", lambda d: d.g_downhill == 1)
player_metric(E, "g_drive_top", "g_drive_top | zone", lambda d: d.team_coverage_man_zone == "ZONE_COVERAGE")
player_metric(E, "g_drive_top", "g_drive_top | man", lambda d: d.team_coverage_man_zone == "MAN_COVERAGE")
player_metric(E, "g_drive_top", "g_drive_top | after the throw", lambda d: d.g_post_throw == 1)
player_metric(E, "g_drive_top", "g_drive_top | before the throw", lambda d: d.g_post_throw == 0)
for col in ["max_speed_after_throw", "gain_05", "heading_change_05"]:
    player_metric(TH, col, f"throw break: {col}")
player_metric(TH, "max_speed_after_throw", "throw break: max speed | zone", lambda d: d.team_coverage_man_zone == "ZONE_COVERAGE")
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/60_game_transition_reliability.csv", index=False)
pd.set_option("display.width", 200)
print("\nper-CB in-game metrics (>= %d events): reliability across game halves and correlation with combine drive speed" % MIN)
print(R.round(2).to_string(index=False))
