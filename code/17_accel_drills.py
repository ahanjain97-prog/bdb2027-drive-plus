"""Which combine drill movements carry in-game acceleration? Partial r (given subgroup + height + weight + arm)
between each per-drill tracking feature and NFL game peak acceleration (p95 of play maxima), within group. BH within group."""
import numpy as np, pandas as pd
from scipy import stats
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
caps = pd.read_parquet(f"{ROOT}/data/game_caps.parquet"); caps = caps[caps.n_plays >= 80]
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[ct.entity_type == "PLAYER"].sort_values(["event_id", "time"])
ct["drill"] = (ct.drill_type + ":" + ct.drill_name).str.replace("SKILL_DRILLS_", "")
rows = []
for eid, g in ct.groupby("event_id"):
    s = g.s.values.astype(float)
    if len(s) < 8 or s.max() < 1.0:
        continue
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    al = np.gradient(s, t); on = int(np.argmax(s >= 1.0))
    # acceleration build-up: mean ds/dt over the first 0.5 / 1.0 s after onset, speed reached at +1 s
    i5, i10 = min(on + 5, len(s) - 1), min(on + 10, len(s) - 1)
    rows.append({"nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "peak_accel": al.max(), "peak_decel": -al.min(),
                 "top_speed": s.max(), "accel_0_0.5s": (s[i5] - s[on]) / max(t[i5] - t[on], 1e-6),
                 "accel_0_1s": (s[i10] - s[on]) / max(t[i10] - t[on], 1e-6), "speed_at_1s": s[i10],
                 "t_to_90pct": t[int(np.argmax(s >= .9 * s.max()))] - t[on], "mean_speed": s[on:].mean()})
cp = pd.DataFrame(rows).groupby(["nfl_id", "drill"]).mean()
out = []
for grp in ["REC", "OL", "DL", "DB"]:
    ids = caps.index.intersection(base.index[base.grp == grp])
    ids = ids[base.loc[ids, "combine_weight"].notna()]
    sub = base.loc[ids, "sub"]
    for d in cp.index.get_level_values("drill").unique():
        x = cp.xs(d, level="drill")
        have = x.index.intersection(ids)
        if len(have) < max(25, 0.5 * len(ids)):
            continue
        Z = np.column_stack([np.ones(len(have)), base.loc[have, ["combine_height", "combine_weight"]].values] +
                            [(sub.loc[have] == s_).astype(float).values for s_ in sorted(sub.unique())[1:]])
        Z = np.where(np.isnan(Z), np.nanmean(Z, 0), Z)
        y = caps.loc[have, "g_accel_peak"].values
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        for f in x.columns:
            v = x.loc[have, f].values
            rv = v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
            r, p = stats.pearsonr(rv, ry)
            out.append({"group": grp, "drill": d, "feature": f, "n": len(have), "partial_r": r, "p": p})
R = pd.DataFrame(out)
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)
R["q_bh_group"] = R.groupby("group").p.transform(bh)
R.to_csv(f"{ROOT}/results/17_accel_drills.csv", index=False)
pd.set_option("display.width", 250)
for grp in ["REC", "OL", "DL", "DB"]:
    t = R[R.group == grp].sort_values("p").head(8)
    print(f"\n{grp}: tests {len(R[R.group == grp])}, q<.10: {(R[R.group == grp].q_bh_group < .10).sum()}")
    print(t.round(3).to_string(index=False))
