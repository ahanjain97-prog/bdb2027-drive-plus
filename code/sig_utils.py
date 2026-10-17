"""Aggregate per-play feature matrices into player x part signatures.

Context adjustment: each play's features minus the mean of OTHER players' plays in the same context cell
(side | lined-up position | play type | route or coverage assignment), backing off to coarser cells when the
leave-player-out cell has < MIN_N plays. The signature is the player's mean residual, i.e. how he moves
relative to what other players do when asked to do the same thing. Raw (unadjusted) means are kept too.
"""
import hashlib
import numpy as np
import pandas as pd

MIN_N = 30
SEQ = "/Users/ahanjain/bdb_ssl/data/seq"


def load_meta():
    m = pd.read_parquet(f"{SEQ}/meta.parquet")
    pl = pd.read_parquet("/Users/ahanjain/bdb_ssl/data/raw/players.parquet")[["nfl_id", "draft_year"]]
    m = m.merge(pl, on="nfl_id", how="left")
    assert (m.idx.values == np.arange(len(m))).all()
    m["half"] = m.game_id.map(lambda g: "h%d" % (int(hashlib.md5(f"half-{g}".encode()).hexdigest(), 16) % 2))
    assign = np.where(m.side == "OFF", m.route_ran.fillna("none"), m.coverage_assignment.fillna("none"))
    lp = m.lined_up_position.fillna("NA")
    m["cell1"] = m.side + "|" + lp + "|" + m.play_type + "|" + assign
    m["cell2"] = m.side + "|" + lp + "|" + m.play_type
    m["cell3"] = m.side + "|" + lp
    m["cell4"] = m.side
    return m


def loo_context_means(E, meta):
    """Leave-player-out context-cell means for every play, with fine -> coarse backoff."""
    N, d = E.shape
    pid = pd.factorize(meta.nfl_id)[0]
    mu = np.zeros_like(E)
    done = np.zeros(N, dtype=bool)
    level = np.zeros(N, dtype=np.int8)
    for li, col in enumerate(["cell1", "cell2", "cell3", "cell4"], start=1):
        ck = pd.factorize(meta[col])[0]
        cp = pd.factorize(pd.Series(ck).astype(np.int64) * (pid.max() + 1) + pid)[0]
        S_c = pd.DataFrame(E).groupby(ck).sum().values
        S_cp = pd.DataFrame(E).groupby(cp).sum().values
        n_c = np.bincount(ck)
        n_cp = np.bincount(cp)
        loo_n = n_c[ck] - n_cp[cp]
        use = ~done & (loo_n >= MIN_N)
        mu[use] = (S_c[ck[use]] - S_cp[cp[use]]) / loo_n[use, None]
        level[use] = li
        done |= use
    assert done.all(), "some plays had no leave-player-out reference"
    return mu, level


def signatures(E, meta, block, parts=("half", "season", "rookie")):
    """Return long DataFrame: nfl_id, part, kind (raw/resid), n, f0..f{d-1}."""
    E = np.asarray(E, dtype=np.float32)
    med = np.nanmedian(E, axis=0)
    E = np.where(np.isnan(E), med, E)
    mu, level = loo_context_means(E, meta)
    R = E - mu
    print(f"[{block}] context level used: " + ", ".join(f"cell{l}={np.mean(level == l):.3f}" for l in range(1, 5)))
    cols = [f"f{i}" for i in range(E.shape[1])]
    out = []
    for part in parts:
        if part == "half":
            key = meta.half.values
        elif part == "season":
            key = meta.season.astype(str).values
        elif part == "rookie":
            key = np.where(meta.season == meta.draft_year, "rookie", "other")
        for kind, M in (("raw", E), ("resid", R)):
            df = pd.DataFrame(M, columns=cols)
            df["nfl_id"] = meta.nfl_id.values
            df["part"] = key
            g = df.groupby(["nfl_id", "part"])
            agg = g[cols].mean()
            agg["n"] = g.size()
            agg = agg.reset_index()
            agg["kind"] = kind
            if part == "rookie":
                agg = agg[agg.part == "rookie"]
            out.append(agg)
    sig = pd.concat(out, ignore_index=True)
    sig["block"] = block
    return sig[["block", "nfl_id", "part", "kind", "n"] + cols]
