"""Direct combine -> NFL comparison of the SAME physical quantity, same units, same formulas.

Combine (tracking, every drill attempt) and game (every post-snap frame) both use:
  speed s (yd/s); along-track accel a_long = ds/dt (central difference); lateral accel
  a_lat = (vx*ay - vy*ax)/s for s > 0.5 (central differences of the velocity vector); |a| = raw `a` column.
Combine capacity  = max over the player's attempts (top speed and accel from the 40-yard dash; decel / lateral /
                    |a| from all drills).
Game capacity     = high quantile over the player's plays of each play's maximum (p99 for top speed, p95 for the
                    rest; >= MIN_PLAYS plays), so one noisy frame cannot set it.
Per position group: level comparison (game / combine), Pearson + Spearman r across players, partial r given
body weight, and a per-player table (combine value, game value, predicted game value, residual).
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from scipy import stats

ROOT = "/Users/ahanjain/bdb_ssl"
D = f"{ROOT}/data/raw"
MIN_PLAYS = 80
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")

# ---------------- combine side
ct = pd.read_parquet(f"{D}/combine_tracking.parquet")
ct = ct[ct.entity_type == "PLAYER"].sort_values(["event_id", "time"])


def attempt_caps(g):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(t) < 5:
        return None
    s = g.s.values.astype(float)
    th = np.radians(g.dir.values.astype(float))
    vx, vy = s * np.sin(th), s * np.cos(th)
    along = np.gradient(s, t)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    lat = np.where(s > 0.5, np.abs(vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    return {"top_speed": s.max(), "accel_peak": along.max(), "decel_peak": -along.min(),
            "lateral_peak": lat.max(), "amag_peak": g.a.max()}


rows = []
for eid, g in ct.groupby("event_id"):
    c = attempt_caps(g)
    if c:
        rows.append({"nfl_id": g.nfl_id.iat[0], "drill_type": g.drill_type.iat[0], "t0": g.time.iat[0], **c})
att = pd.DataFrame(rows)
forty = att[att.drill_type == "FORTY_YARD_DASH"].groupby("nfl_id")[["top_speed", "accel_peak"]].max()
alld = att.groupby("nfl_id")[["decel_peak", "lateral_peak", "amag_peak"]].max()
comb = forty.join(alld, how="outer")
comb.columns = ["c_" + c for c in comb.columns]

# combine reliability: alternate attempts in time order -> two parallel halves per player
att = att.sort_values(["nfl_id", "t0"])
att["half"] = att.groupby("nfl_id").cumcount() % 2
f2 = att[att.drill_type == "FORTY_YARD_DASH"].copy()
f2["half"] = f2.groupby("nfl_id").cumcount() % 2
c_half = pd.concat([f2.groupby(["nfl_id", "half"])[["top_speed", "accel_peak"]].max(),
                    att.groupby(["nfl_id", "half"])[["decel_peak", "lateral_peak", "amag_peak"]].max()], axis=1)

# ---------------- game side (same formulas; per-play maxima from the sequence tensors)
meta = pd.read_parquet(f"{ROOT}/data/seq/meta.parquet")
voc = json.load(open(f"{ROOT}/data/seq/vocab.json"))
F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r")
V = np.load(f"{ROOT}/data/seq/valid.npy")
post = V.copy(); post[:, :PRE] = False
ch = lambda n: np.where(post, np.asarray(X[:, :, F.index(n)], dtype=np.float32), np.nan)
import hashlib
half = meta.game_id.map(lambda g: int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2).values
play = pd.DataFrame({"nfl_id": meta.nfl_id.values, "half": half,
                     "top_speed": np.nanmax(ch("s"), 1), "accel_peak": np.nanmax(ch("a_long"), 1),
                     "decel_peak": -np.nanmin(ch("a_long"), 1), "lateral_peak": np.nanmax(np.abs(ch("a_lat")), 1),
                     "amag_peak": np.nanmax(ch("a_mag"), 1)})
q = {"top_speed": .99, "accel_peak": .95, "decel_peak": .95, "lateral_peak": .95, "amag_peak": .95}
gp = play.groupby("nfl_id")
game = pd.DataFrame({k: gp[k].quantile(v) for k, v in q.items()})
gh = play.groupby(["nfl_id", "half"])
g_half = pd.DataFrame({k: gh[k].quantile(v) for k, v in q.items()})
g_half["n"] = gh.size()
g_half = g_half[g_half.n >= MIN_PLAYS // 2]
game["n_plays"] = gp.size()
game = game[game.n_plays >= MIN_PLAYS]
game.columns = ["g_" + c if c != "n_plays" else c for c in game.columns]

df = comb.join(game, how="inner").join(base[["display_name", "grp", "sub", "combine_weight"]])
df = df[df.grp.notna()]
QUANT = list(q)


def split_half_rel(tab, ids, k):
    """Correlation between the two halves across players, Spearman-Brown stepped up to the full measure."""
    w = tab[k].unstack("half").reindex(ids).dropna()
    if len(w) < 15 or w.shape[1] < 2:
        return np.nan, len(w)
    r = np.corrcoef(w[0], w[1])[0, 1]
    return 2 * r / (1 + r), len(w)


out, per_player = [], []
units = [("ALL", df)] + [(g, df[df.grp == g]) for g in ["REC", "OL", "DL", "DB"]] + \
        [(s_, df[df["sub"] == s_]) for s_ in ["WR", "TE", "T", "IOL", "EDGE", "IDL", "CB", "S"]]
for grp, d in units:
    for k in QUANT:
        m = d[[f"c_{k}", f"g_{k}", "combine_weight", "sub", "display_name"]].dropna()
        if len(m) < 20:
            continue
        cx, gy, w = m[f"c_{k}"].values, m[f"g_{k}"].values, m.combine_weight.values
        r, p = stats.pearsonr(cx, gy)
        rho, prho = stats.spearmanr(cx, gy)
        # partial r given body weight (+ subgroup dummies within the group)
        Z = np.column_stack([np.ones(len(m)), w] + [(m["sub"] == s).astype(float) for s in sorted(m["sub"].unique())[1:]])
        rc = cx - Z @ np.linalg.lstsq(Z, cx, rcond=None)[0]
        rg = gy - Z @ np.linalg.lstsq(Z, gy, rcond=None)[0]
        pr, pp = stats.pearsonr(rc, rg)
        slope, icpt = np.polyfit(cx, gy, 1)
        rel_c, n_rc = split_half_rel(c_half, m.index, k)
        rel_g, n_rg = split_half_rel(g_half, m.index, k)
        ceil = np.sqrt(max(rel_c, 1e-6) * max(rel_g, 1e-6)) if np.isfinite(rel_c) and np.isfinite(rel_g) else np.nan
        out.append({"group": grp, "quantity": k, "n": len(m), "combine_mean": cx.mean(), "game_mean": gy.mean(),
                    "rel_combine": rel_c, "n_rel_c": n_rc, "rel_game": rel_g, "r_ceiling": ceil,
                    "r_disattenuated": r / ceil if ceil and ceil > 0.1 else np.nan,
                    "game_over_combine": gy.mean() / cx.mean(), "combine_sd": cx.std(), "game_sd": gy.std(),
                    "pearson_r": r, "p": p, "spearman": rho, "partial_r_weight_sub": pr, "p_partial": pp,
                    "slope": slope, "intercept": icpt})
        if grp in ("WR", "TE", "T", "IOL", "EDGE", "IDL", "CB", "S"):
            pred = icpt + slope * cx
            for pid, a, b, pr_ in zip(m.index, cx, gy, pred):
                per_player.append({"nfl_id": pid, "name": base.at[pid, "display_name"], "sub": base.at[pid, "sub"],
                                   "group": grp, "quantity": k, "combine": a, "game": b, "game_pred_from_combine": pr_,
                                   "game_minus_pred": b - pr_})
R = pd.DataFrame(out)
R.to_csv(f"{ROOT}/results/13_translation_direct.csv", index=False)
pp_ = pd.DataFrame(per_player)
pp_.to_csv(f"{ROOT}/results/13_translation_per_player.csv", index=False)
pd.set_option("display.width", 250)
cols = ["group", "quantity", "n", "combine_mean", "game_mean", "game_over_combine", "pearson_r", "p", "partial_r_weight_sub",
        "rel_combine", "n_rel_c", "rel_game", "r_ceiling", "r_disattenuated"]
print(R[cols].round(3).to_string(index=False))
print("\nplayers with combine tracking AND >= %d game plays: %d" % (MIN_PLAYS, len(df)))
for k in ["top_speed", "lateral_peak"]:
    t = pp_[(pp_.quantity == k) & (pp_.group == "WR")].sort_values("game_minus_pred")
    print(f"\nWR {k}: plays slower than combine predicts -> faster")
    print(t.head(4)[["name", "sub", "combine", "game", "game_pred_from_combine", "game_minus_pred"]].round(2).to_string(index=False))
    print(t.tail(4)[["name", "sub", "combine", "game", "game_pred_from_combine", "game_minus_pred"]].round(2).to_string(index=False))
# capacities for the translation model (15_translation_model.py)
game.to_parquet(f"{ROOT}/data/game_caps.parquet")
att.to_parquet(f"{ROOT}/data/combine_attempt_caps.parquet", index=False)
