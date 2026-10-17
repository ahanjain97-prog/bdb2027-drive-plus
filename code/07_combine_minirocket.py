"""Combine-drill block: MiniRocket random-convolution features per drill attempt -> player x drill.

Channels are rotation/mirror invariant (drills run in both directions): speed, |accel| (raw, unsigned),
signed along-track accel (ds/dt), |lateral accel| (s * heading rate). Each attempt is resampled to 64 points
over its clip; log clip duration is kept as an extra scalar. Per drill, features are averaged over a
player's attempts and reduced with a label-free PCA fitted on every player who ran that drill
(transductive but outcome-free). Output: data/sig/combine_mr.parquet (one row per player, all drills).
"""
import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket

D = "/Users/ahanjain/bdb_ssl/data/raw"
L, K_PC, MIN_PLAYERS = 64, 10, 30

ct = pd.read_parquet(f"{D}/combine_tracking.parquet")
ct = ct[ct.entity_type == "PLAYER"].copy()
ct["drill"] = ct.drill_type + ":" + ct.drill_name
ct = ct.sort_values(["event_id", "time"])


def attempt_series(g):
    s = g.s.values.astype(float)
    a = g.a.values.astype(float)
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    if len(s) < 5 or t[-1] <= 0:
        return None
    along = np.gradient(s, t)
    th = np.unwrap(np.radians(g.dir.values.astype(float)))
    lat = np.abs(s * np.gradient(th, t))
    lat = np.where(s > 0.5, lat, 0.0)
    grid = np.linspace(0, t[-1], L)
    X = np.stack([np.interp(grid, t, c) for c in (s, a, along, lat)])
    return X, np.log(t[-1])


rows = []
for drill, gd in ct.groupby("drill"):
    if gd.nfl_id.nunique() < MIN_PLAYERS:
        continue
    series, pid, dur = [], [], []
    for eid, g in gd.groupby("event_id"):
        r = attempt_series(g)
        if r is None:
            continue
        series.append(r[0]); dur.append(r[1]); pid.append(g.nfl_id.iat[0])
    X = np.stack(series).astype(np.float32)
    X = np.clip(X, -30, 30)
    X = X / (X.reshape(len(X), 4, -1).std(axis=(0, 2))[None, :, None] + 1e-6)
    mr = MiniRocket(n_kernels=10000, random_state=0)
    F = mr.fit_transform(X).astype(np.float32)
    df = pd.DataFrame(F); df["nfl_id"] = pid; df["dur"] = dur
    P = df.groupby("nfl_id").mean()
    dur_p = P.pop("dur")
    Z = (P.values - P.values.mean(0)) / (P.values.std(0) + 1e-6)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :K_PC] * S[:K_PC]
    ev = (S[:K_PC] ** 2).sum() / (S ** 2).sum()
    tag = drill.replace("SKILL_DRILLS_", "").replace(":", "_")
    out = pd.DataFrame(pcs, index=P.index, columns=[f"{tag}__pc{i}" for i in range(K_PC)])
    out[f"{tag}__logdur"] = dur_p
    rows.append(out)
    print(f"{drill:60s} players {len(P):4d} attempts {len(X):4d} top-{K_PC} PCA var {ev:.2f}", flush=True)

sig = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "nfl_id"})
sig.to_parquet("/Users/ahanjain/bdb_ssl/data/sig/combine_mr.parquet", index=False)
print("combine_mr", sig.shape)
