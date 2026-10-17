"""Build fixed-length, play-standardized trajectory tensors for every tracked player-play.

Frame of reference (verified in 00_inspect): dir/o are compass degrees, 0 = +y, clockwise, so
vx = s*sin(dir), vy = s*cos(dir). Plays with play_direction == 'left' are rotated 180 degrees so the
offense always moves toward +x. x is expressed relative to the line of scrimmage, y relative to the
player's own snap position (plus absolute field width as a separate channel).

Window: 10 frames before the snap .. 69 frames after (T = 80, 10 Hz). Missing frames are masked.
`a` in the raw data is an unsigned magnitude, so signed along-track and lateral accel are derived here.

Outputs (data/seq/):
  X.npy        float16 [N, T, F]  frame features (unstandardized)
  valid.npy    bool    [N, T]
  ctx_cat.npy  int16   [N, C]     categorical context codes (0 = UNK/missing)
  ctx_num.npy  float32 [N, K]
  meta.parquet one row per sequence (keys, split, context labels)
  vocab.json   context vocabularies + feature names
"""
import json, os, hashlib
import duckdb, numpy as np, pandas as pd

D = "/Users/ahanjain/bdb_ssl/data/raw"
OUT = "/Users/ahanjain/bdb_ssl/data/seq"
os.makedirs(OUT, exist_ok=True)
PRE, POST = 10, 69
T = PRE + 1 + POST

FEATS = ["x_rel", "y_rel", "y_field", "vx", "vy", "s", "a_long", "a_lat", "a_mag",
         "sin_o", "cos_o", "sin_od", "cos_od", "o_rate",
         "t_snap", "pre_snap", "post_pass", "post_handoff", "post_dead"]
O_FEATS = ["sin_o", "cos_o", "sin_od", "cos_od", "o_rate"]          # dropped for orientation-dropout / combine
TARGET_FEATS = FEATS[:14]                                           # what the masked model reconstructs

con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='6GB'")

# play-level context
con.execute(f"""
CREATE TABLE ctx AS
SELECT p.game_id, p.play_id, p.nfl_id, g.season, g.season_type, g.week,
       CASE WHEN p.team_abbr = p.possession_team THEN 'OFF' ELSE 'DEF' END AS side,
       p.play_direction,
       CASE WHEN p.yardline_side IS NULL THEN 60.0
            WHEN p.yardline_side = p.possession_team THEN 10.0 + p.yardline_number
            ELSE 110.0 - p.yardline_number END AS x_los,
       p.lined_up_position, p.route_ran, p.coverage_assignment, p.team_coverage_man_zone,
       p.team_coverage_type, p.offense_formation, p.receiver_alignment,
       CASE WHEN p.pass_result IN ('C','I','IN','S','R') THEN 'DROPBACK' ELSE 'RUN' END AS play_type,
       p.in_motion_at_ball_snap::VARCHAR AS in_motion, p.blitzing::VARCHAR AS blitzing,
       p.play_nullified_by_penalty AS nullified, p.down, p.yards_to_go, p.quarter,
       (p.pre_snap_home_score - p.pre_snap_visitor_score)
         * CASE WHEN p.possession_team = g.home_team_abbr THEN 1 ELSE -1 END AS score_diff_off
FROM '{D}/player_play.parquet' p JOIN '{D}/games.parquet' g USING (game_id)
WHERE p.play_direction IS NOT NULL
""")

# frames: standardize, align to snap, derive kinematics with window functions
con.execute(f"""
CREATE TABLE fr AS
WITH raw AS (
  SELECT t.game_id, t.play_id, t.nfl_id, t."time", t.event, c.x_los, c.play_direction,
         CASE WHEN c.play_direction = 'left' THEN 120 - t.x ELSE t.x END AS xs,
         CASE WHEN c.play_direction = 'left' THEN 53.3 - t.y ELSE t.y END AS ys,
         t.s, t.a,
         radians(CASE WHEN c.play_direction = 'left' THEN (t.dir + 180) % 360 ELSE t.dir END) AS th,
         radians(CASE WHEN c.play_direction = 'left' THEN (t.o + 180) % 360 ELSE t.o END) AS ob
  FROM '{D}/game_tracking_*.parquet' t
  JOIN ctx c USING (game_id, play_id, nfl_id)
), snap AS (
  SELECT game_id, play_id, nfl_id, min("time") AS ts FROM raw WHERE event = 'ball_snap' GROUP BY ALL
), ev AS (
  SELECT game_id, play_id, nfl_id,
         min("time") FILTER (WHERE event IN ('pass_forward','pass_shovel')) AS t_pass,
         min("time") FILTER (WHERE event IN ('handoff','run','snap_direct','qb_sneak','run_attempt')) AS t_hand,
         min("time") FILTER (WHERE event IN ('tackle','out_of_bounds','touchdown','qb_slide','qb_kneel',
                                            'qb_sack','qb_strip_sack','pass_outcome_incomplete','qb_spike')) AS t_dead
  FROM raw GROUP BY ALL
), r AS (
  SELECT raw.*, round((epoch(raw."time") - epoch(snap.ts)) * 10)::INT AS k,
         ev.t_pass, ev.t_hand, ev.t_dead
  FROM raw JOIN snap USING (game_id, play_id, nfl_id) JOIN ev USING (game_id, play_id, nfl_id)
), w AS (
  SELECT *, s * sin(th) AS vx, s * cos(th) AS vy FROM r WHERE k BETWEEN -{PRE} AND {POST}
), d AS (
  SELECT *,
    (coalesce(lead(s) OVER pw, s) - coalesce(lag(s) OVER pw, s))
      / (0.1 * (CASE WHEN lead(s) OVER pw IS NULL OR lag(s) OVER pw IS NULL THEN 1 ELSE 2 END)) AS a_long,
    (coalesce(lead(vx) OVER pw, vx) - coalesce(lag(vx) OVER pw, vx))
      / (0.1 * (CASE WHEN lead(vx) OVER pw IS NULL OR lag(vx) OVER pw IS NULL THEN 1 ELSE 2 END)) AS ax,
    (coalesce(lead(vy) OVER pw, vy) - coalesce(lag(vy) OVER pw, vy))
      / (0.1 * (CASE WHEN lead(vy) OVER pw IS NULL OR lag(vy) OVER pw IS NULL THEN 1 ELSE 2 END)) AS ay,
    lead(ob) OVER pw AS ob_next, lag(ob) OVER pw AS ob_prev,
    first(ys) OVER (PARTITION BY game_id, play_id, nfl_id ORDER BY abs(k), k
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS y_snap
  FROM w WINDOW pw AS (PARTITION BY game_id, play_id, nfl_id ORDER BY k)
)
SELECT game_id, play_id, nfl_id, k::SMALLINT AS k,
  (xs - x_los)::FLOAT AS x_rel, (ys - y_snap)::FLOAT AS y_rel, (ys - 26.65)::FLOAT AS y_field,
  vx::FLOAT AS vx, vy::FLOAT AS vy, s::FLOAT AS s, a_long::FLOAT AS a_long,
  (CASE WHEN s > 0.5 THEN (vx * ay - vy * ax) / s ELSE 0 END)::FLOAT AS a_lat, a::FLOAT AS a_mag,
  sin(ob)::FLOAT AS sin_o, cos(ob)::FLOAT AS cos_o, sin(ob - th)::FLOAT AS sin_od, cos(ob - th)::FLOAT AS cos_od,
  (degrees(atan2(sin(coalesce(ob_next, ob) - coalesce(ob_prev, ob)), cos(coalesce(ob_next, ob) - coalesce(ob_prev, ob))))
     / (0.1 * (CASE WHEN ob_next IS NULL OR ob_prev IS NULL THEN 1 ELSE 2 END)))::FLOAT AS o_rate,
  (k / 10.0)::FLOAT AS t_snap, (k < 0)::FLOAT AS pre_snap,
  (t_pass IS NOT NULL AND "time" >= t_pass)::FLOAT AS post_pass,
  (t_hand IS NOT NULL AND "time" >= t_hand)::FLOAT AS post_handoff,
  (t_dead IS NOT NULL AND "time" >= t_dead)::FLOAT AS post_dead
FROM d
""")
n_fr = con.execute("SELECT count(*) FROM fr").fetchone()[0]
print("windowed frames", n_fr)

# sequences that have a snap frame -> stable index
seq = con.execute("""
  SELECT c.*, f.n_frames, f.n_post FROM ctx c JOIN
   (SELECT game_id, play_id, nfl_id, count(*) n_frames, count(*) FILTER (WHERE k >= 0) n_post
    FROM fr GROUP BY ALL) f USING (game_id, play_id, nfl_id)
  ORDER BY game_id, play_id, nfl_id""").df()
seq = seq[seq.n_post >= 10].reset_index(drop=True)
seq["idx"] = np.arange(len(seq), dtype=np.int64)
con.register("seq_idx", seq[["game_id", "play_id", "nfl_id", "idx"]])
N = len(seq)
print("sequences", N)

X = np.zeros((N, T, len(FEATS)), dtype=np.float16)
valid = np.zeros((N, T), dtype=bool)
cols = ", ".join(FEATS)
# stream in chunks of sequence index to bound memory
CH = 40000
for lo in range(0, N, CH):
    df = con.execute(f"""SELECT s.idx, f.k, {cols} FROM fr f JOIN seq_idx s USING (game_id, play_id, nfl_id)
                         WHERE s.idx >= {lo} AND s.idx < {lo + CH}""").fetchnumpy()
    i = df["idx"].astype(np.int64); k = df["k"].astype(np.int64) + PRE
    F = np.stack([df[c].astype(np.float32) for c in FEATS], axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    X[i, k] = np.clip(F, -6e4, 6e4).astype(np.float16)
    valid[i, k] = True
    print(f"  filled {min(lo + CH, N)}/{N}", flush=True)

# categorical context
CAT = ["side", "lined_up_position", "route_ran", "coverage_assignment", "team_coverage_man_zone",
       "team_coverage_type", "offense_formation", "receiver_alignment", "play_type", "in_motion",
       "blitzing", "nullified", "season_type", "down"]
vocab = {}
ctx_cat = np.zeros((N, len(CAT)), dtype=np.int16)
for j, c in enumerate(CAT):
    v = seq[c].astype("string").fillna("<NA>").replace("", "<NA>")
    counts = v.value_counts()
    keep = [x for x in counts.index if x != "<NA>" and counts[x] >= 50]
    m = {x: i + 1 for i, x in enumerate(sorted(keep))}
    vocab[c] = m
    ctx_cat[:, j] = v.map(m).fillna(0).astype(np.int16).values

num = pd.DataFrame({
    "ytg": seq.yards_to_go.clip(0, 30).fillna(10) / 10.0,
    "x_los": (seq.x_los - 60.0) / 50.0,
    "quarter": seq.quarter.fillna(2).clip(1, 5) / 4.0,
    "score_diff": seq.score_diff_off.fillna(0).clip(-28, 28) / 14.0,
})
ctx_num = num.values.astype(np.float32)

# train/val split by game (hash, ~10% val)
seq["split"] = np.where(seq.game_id.map(lambda g: int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 10 == 0),
                        "val", "train")

np.save(f"{OUT}/X.npy", X); np.save(f"{OUT}/valid.npy", valid)
np.save(f"{OUT}/ctx_cat.npy", ctx_cat); np.save(f"{OUT}/ctx_num.npy", ctx_num)
seq.to_parquet(f"{OUT}/meta.parquet", index=False)
json.dump({"feats": FEATS, "o_feats": O_FEATS, "target_feats": TARGET_FEATS, "cat": CAT, "vocab": vocab,
           "num": list(num.columns), "pre": PRE, "post": POST}, open(f"{OUT}/vocab.json", "w"), indent=1)

# sanity: LOS-relative x at snap by side, frame coverage, feature ranges
snapx = X[:, PRE, 0].astype(np.float32)
for sd in ("OFF", "DEF"):
    m = (seq.side == sd).values & valid[:, PRE]
    print(sd, "x_rel at snap quantiles", np.percentile(snapx[m], [1, 10, 50, 90, 99]).round(2))
print("valid frames per seq quantiles", np.percentile(valid.sum(1), [1, 10, 50, 90, 99]))
print("snap frame valid share", valid[:, PRE].mean().round(4))
Xf = X[valid].astype(np.float32)
print(pd.DataFrame({"mean": Xf.mean(0), "std": Xf.std(0), "p01": np.percentile(Xf, 1, axis=0),
                    "p99": np.percentile(Xf, 99, axis=0)}, index=FEATS).round(3).to_string())
print(seq.split.value_counts().to_string())
