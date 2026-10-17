"""How much of a player's NFL game movement can his combine data predict?

Unit: player (combine tracking + >= 80 tracked NFL plays), within position group (REC/OL/DL/DB, subgroup dummies).
Game-side profile (29 targets, standardized within group):
  5 capacities  (game top speed p99, accel/decel/lateral/|a| p95 of play maxima)    -- 13_translation_direct
  16 style      (context-adjusted mean of the hand-built per-play physics, all games)
  8 learned     (PCs of the mirror-invariant SSL game signature, fitted within group, label-free)
Combine sources (each added on top of S0 = subgroup + height + weight + arm length, which is shown as reference):
  stopwatch  official 40 / 10-yd / vertical / broad / 3-cone / shuttle / bench
  physics    per drill: top speed, peak accel/decel/lateral/|a|, time to 90% top speed, duration, path length
  minirocket random-convolution features per drill
  ssl        the GAME-pretrained encoder applied to combine drills (shared space)
  all        stopwatch + physics + ssl
Large blocks -> per drill label-free PCA(4), concatenated, -> PCA(8). Ridge with separate block scale (c=0 nests
S0), exact LOO alpha, repeated 5-fold CV. p = permutation of block rows within subgroup.
Lineup test: out-of-fold predicted 29-dim profile; percentile rank of the player's own NFL profile among all
players in the group by distance (0.5 = chance, 1 = always closest), top-1 and top-5 hit rates.
"""
import sys, json
import numpy as np
sys.argv = ["x", "--n_rep", "5"]
exec(open("/Users/ahanjain/bdb_ssl/code/08_probe.py").read().split("tests = []")[0])
from joblib import Parallel, delayed

ROOT = "/Users/ahanjain/bdb_ssl"
N_PERM, MIN_PLAYS, COV = 100, 80, 0.5

# ---------------- game profile
caps = pd.read_parquet(f"{ROOT}/data/game_caps.parquet")
caps = caps[caps.n_plays >= MIN_PLAYS].drop(columns="n_plays")


def pooled(block, prefix):
    s = pd.read_parquet(f"{ROOT}/data/sig/{block}.parquet")
    s = s[(s.kind == "resid") & s.part.isin(["h0", "h1"])]
    cols = [c for c in s.columns if c.startswith("f")]
    w = s[cols].mul(s.n, axis=0); w["nfl_id"] = s.nfl_id; w["n"] = s.n
    g = w.groupby("nfl_id").sum()
    out = g[cols].div(g.n, axis=0)
    out.columns = [f"{prefix}{c}" for c in cols]
    return out[g.n >= MIN_PLAYS]


phys_names = json.load(open(f"{ROOT}/data/blocks/phys_names.json"))
style = pooled("phys", "style_"); style.columns = [f"style_{n}" for n in phys_names]
sslg = pooled("ssl", "sslg_")

# ---------------- combine sources
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[ct.entity_type == "PLAYER"].sort_values(["event_id", "time"])
ct["drill"] = (ct.drill_type + ":" + ct.drill_name).str.replace("SKILL_DRILLS_", "")
rows = []
for eid, g in ct.groupby("event_id"):
    s = g.s.values.astype(float)
    if len(s) < 8 or s.max() < 1.0:
        continue
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    th = np.radians(g.dir.values.astype(float))
    vx, vy = s * np.sin(th), s * np.cos(th)
    al = np.gradient(s, t); ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    lat = np.where(s > 0.5, np.abs(vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    on = int(np.argmax(s >= 1.0))
    t90 = t[int(np.argmax(s >= 0.9 * s.max()))] - t[on]
    rows.append({"nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "vmax": s.max(), "acc": al.max(), "dec": -al.min(),
                 "lat": lat.max(), "amag": g.a.max(), "t90": t90, "dur": t[-1] - t[on], "path": g.dis.sum()})
cphys = pd.DataFrame(rows).groupby(["nfl_id", "drill"]).mean()
cssl = pd.read_parquet(f"{ROOT}/data/blocks/combine_ssl_attempt.parquet")
cssl["drill"] = cssl.drill.str.replace("SKILL_DRILLS_", "")
ecols = [c for c in cssl.columns if c.startswith("e") and c[1:].isdigit()]
cssl = cssl.groupby(["nfl_id", "drill"])[ecols].mean()
mr = pd.read_parquet(f"{ROOT}/data/sig/combine_mr.parquet").set_index("nfl_id")
STOP = ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
SIZE = ["combine_height", "combine_weight", "arm_length"]


def zfill(M):
    M = np.asarray(M, float)
    mu = np.nanmean(M, 0); mu = np.where(np.isnan(mu), 0, mu)
    M = np.where(np.isnan(M), mu, M)
    sd = M.std(0); sd[sd < 1e-9] = 1
    return (M - mu) / sd


def drill_block(long, ids, k_drill=4):
    """Per drill (>= COV of these players did it): label-free PCA(k) on players who ran it; NaN for the rest."""
    parts = []
    for d in long.index.get_level_values("drill").unique():
        sub = long.xs(d, level="drill")
        have = sub.index.intersection(ids)
        if len(have) < COV * len(ids) or len(have) < 20:
            continue
        Z = zfill(sub.loc[have].values)
        U, sv, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        k = min(k_drill, Z.shape[1])
        pcs = pd.DataFrame(U[:, :k] * sv[:k], index=have, columns=[f"{d}_{j}" for j in range(k)])
        parts.append(pcs.reindex(ids))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ids)


def run_one(grp, tname, y, B, Pblk, sub, seed=0):
    folds = make_folds(np.arange(len(y)), args.n_rep, seed)
    cv = CV(B, y, folds)
    obs = cv.r2_full(Pblk) - cv.r2_base
    rs = np.random.default_rng(seed + 7); null = []
    for _ in range(N_PERM):
        perm = np.arange(len(y))
        for s_ in np.unique(sub):
            ix = np.where(sub == s_)[0]; perm[ix] = rs.permutation(ix)
        null.append(cv.r2_full(Pblk[perm]) - cv.r2_base)
        if len(null) == 30 and np.mean(np.array(null) >= obs) > 0.3:
            break
    null = np.array(null)
    return {"r2_size_only": cv.r2_base, "r2_with_source": cv.r2_base + obs, "delta": obs,
            "p": (1 + (null >= obs).sum()) / (1 + len(null))}


def oof_predictions(B, Pblk, Y, n_rep=5):
    """Out-of-fold predicted profile (averaged over repeats), same model class as the R^2 runs."""
    pred = np.zeros_like(Y)
    for r in range(n_rep):
        for f in make_folds(np.arange(len(Y)), 1, 100 + r):
            for tr, te in f:
                Btr, Bte = _std(tr, te, B, lambda m: np.nanmedian(m, 0))
                if Pblk is None:
                    for j in range(Y.shape[1]):
                        pred[te, j] += ridge_loo(Btr, Y[tr, j], Bte) / n_rep
                else:
                    Ptr, Pte = _std(tr, te, Pblk, lambda m: np.nanmean(m, 0))
                    for j in range(Y.shape[1]):
                        pred[te, j] += ridge_two_penalty(Btr, Ptr, Y[tr, j], Bte, Pte) / n_rep
    return pred


def lineup(pred, Y):
    d = ((pred[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    own = np.diag(d)
    pct = ((d > own[:, None]).sum(1)) / (len(Y) - 1)
    rank = (d < own[:, None]).sum(1)
    return pct.mean(), (rank == 0).mean(), (rank < 5).mean()


results, lineups = [], []
for grp in ["REC", "OL", "DL", "DB"]:
    ids = base.index[(base.grp == grp)]
    prof = caps.join(style, how="inner").join(sslg, how="inner")
    ids = prof.index.intersection(ids)
    ids = ids[base.loc[ids, "combine_weight"].notna()]
    # learned game signature -> within-group PCA(8), label-free
    G = zfill(prof.loc[ids, sslg.columns].values)
    U, sv, _ = np.linalg.svd(G - G.mean(0), full_matrices=False)
    ssl_pcs = pd.DataFrame(U[:, :8] * sv[:8], index=ids, columns=[f"learned_pc{j+1}" for j in range(8)])
    Yd = pd.concat([prof.loc[ids, list(caps.columns) + list(style.columns)], ssl_pcs], axis=1)
    has_comb = cphys.index.get_level_values("nfl_id").unique()
    ids = ids[ids.isin(has_comb)]
    Yd = Yd.loc[ids]
    Y = zfill(Yd.values)
    sub = base.loc[ids, "sub"].values
    S0 = pd.DataFrame(index=ids)
    for s_ in sorted(set(sub))[1:]:
        S0[f"sub_{s_}"] = (sub == s_).astype(float)
    for c in SIZE:
        S0[c] = base.loc[ids, c]
    B = S0.values.astype(float)
    stop = base.loc[ids, STOP].copy()
    for c in STOP:
        if stop[c].isna().any():
            stop[c + "_miss"] = stop[c].isna().astype(float)
    blocks = {"stopwatch": stop.values.astype(float),
              "physics": block_pcs(drill_block(cphys, ids).values, 8),
              "minirocket": block_pcs(mr.reindex(ids)[combine_cols(ids)].values, 8),
              "ssl": block_pcs(drill_block(cssl, ids).values, 8)}
    blocks["all"] = np.column_stack([zfill(blocks["stopwatch"]), blocks["physics"], blocks["ssl"]])
    print(f"{grp}: players {len(ids)}  subgroups {dict(pd.Series(sub).value_counts())}", flush=True)
    jobs = [(grp, t, Y[:, j], B, blocks[src], sub, src) for j, t in enumerate(Yd.columns) for src in blocks]
    res = Parallel(n_jobs=4)(delayed(run_one)(g_, t_, y_, B_, P_, s_) for g_, t_, y_, B_, P_, s_, _ in jobs)
    for (g_, t_, *_rest, src), r in zip(jobs, res):
        kind = "capacity" if t_ in caps.columns else ("learned" if t_.startswith("learned") else "style")
        results.append({"group": g_, "target": t_, "kind": kind, "source": src, "n": len(ids), **r})
    # lineup test
    base_pred = oof_predictions(B, None, Y)
    lineups.append({"group": grp, "source": "size_only", **dict(zip(["mean_pct", "top1", "top5"], lineup(base_pred, Y))),
                    "chance_top1": 1 / len(ids), "chance_top5": 5 / len(ids), "n": len(ids)})
    for src, P in blocks.items():
        lp = lineup(oof_predictions(B, P, Y), Y)
        lineups.append({"group": grp, "source": src, **dict(zip(["mean_pct", "top1", "top5"], lp)),
                        "chance_top1": 1 / len(ids), "chance_top5": 5 / len(ids), "n": len(ids)})

R = pd.DataFrame(results); L = pd.DataFrame(lineups)
R.to_csv(f"{ROOT}/results/15_translation_model.csv", index=False)
L.to_csv(f"{ROOT}/results/15_translation_lineup.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("\n=== mean R^2 by group x target kind x source (size-only reference in first column) ===")
s = R.groupby(["group", "kind", "source"]).agg(r2_size=("r2_size_only", "mean"), r2_src=("r2_with_source", "mean"),
                                               delta=("delta", "mean"), n_p05=("p", lambda p: int((p < .05).sum())),
                                               targets=("p", "size"))
print(s.round(3).to_string())
print("\n=== lineup test (pick the player's NFL movement profile out of his position group) ===")
print(L.round(3).to_string(index=False))
print("\n=== targets where a combine source adds >= .05 R^2 over size (p < .05) ===")
hit = R[(R.delta >= .05) & (R.p < .05)].sort_values(["group", "delta"], ascending=[True, False])
print(hit[["group", "target", "kind", "source", "n", "r2_size_only", "r2_with_source", "delta", "p"]].round(3).to_string(index=False))
