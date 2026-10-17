"""Robustness of the hoop -> pass-rush penetration link (DL pooled, unit-adjusted).
Checks: beyond weight + every stopwatch test; per draft class; leave-one-out influence; pressure itself;
the mediation chain (hoop -> NFL arc speed -> penetration); and whether 'hoop time' is a clip-length artifact
(compare with the time to complete the figure-8 measured from the path: second loop's exit)."""
import hashlib, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/Users/ahanjain/bdb_ssl/code/23_bend_deep.py").read())     # defines base, cb, unit, GM, ...

# figure-8 completion from the path: time from onset to the end of the last turn segment (not the clip end)
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
hoop = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "RUN_THE_HOOP_DRILL")].sort_values(["event_id", "time"])
T = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_turns.parquet")
T = T[T.family == "hoop"]
rows = []
for eid, g in hoop.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; s = g.s.values
    on = t[int(np.argmax(s >= 1.0))]
    tt = T[T.event_id == eid]
    if len(tt) < 2:
        continue
    end_turn = (tt.t_start + tt.dur).max()
    rows.append({"nfl_id": g.nfl_id.iat[0], "clip_after_last_turn": t[-1] - end_turn, "fig8_time": end_turn - on,
                 "turn_speed": np.average(tt.sust_speed, weights=tt.dur)})
H = pd.DataFrame(rows).groupby("nfl_id").mean()
cb2 = cb.join(H, how="left")
print("clip time after last turn: median %.2f s, sd %.2f" % (H.clip_after_last_turn.median(), H.clip_after_last_turn.std()))
print("r(hoop clip time, fig8 path time) = %.2f" % cb2[["c_hoop_time", "fig8_time"]].dropna().corr().iloc[0, 1])

ids = base.index[base.nfl_position.isin(["DE", "OLB", "DT", "NT"])]
STOP = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump"]


def full(xname, x, yname, y, extra=()):
    d = pd.concat([x.rename("x"), y.rename("y"), unit.rename("unit"), base.combine_weight.rename("w"),
                   base.draft_year.rename("yr")] + [base[c] for c in STOP], axis=1).reindex(ids).dropna(subset=["x", "y"])
    r0 = stats.pearsonr(d.x, d.y)
    def pc(cols):
        Z = d[cols].copy()
        for c in cols:
            if Z[c].isna().any():
                Z[c + "_m"] = Z[c].isna().astype(float); Z[c] = Z[c].fillna(Z[c].median())
        Z = np.column_stack([np.ones(len(d)), Z.values])
        rx = d.x - Z @ np.linalg.lstsq(Z, d.x, rcond=None)[0]; ry = d.y - Z @ np.linalg.lstsq(Z, d.y, rcond=None)[0]
        return stats.pearsonr(rx, ry)
    pu = pc(["unit"]); pw = pc(["unit", "w"]); ps = pc(["unit", "w"] + STOP)
    by = {yr: (stats.pearsonr(g.x, g.y)[0] if len(g) >= 8 else np.nan, len(g)) for yr, g in d.groupby("yr")}
    loo = []
    for i in range(len(d)):
        dd = d.drop(d.index[i]); loo.append(stats.pearsonr(dd.x, dd.y)[0])
    print(f"\n{xname} -> {yname}: n {len(d)}  raw r {r0[0]:+.3f} (p {r0[1]:.4f}) | +unit {pu[0]:+.3f} (p {pu[1]:.4f}) | "
          f"+weight {pw[0]:+.3f} (p {pw[1]:.4f}) | +all stopwatch {ps[0]:+.3f} (p {ps[1]:.4f})")
    print("   by draft class: " + ", ".join(f"{k}: {v[0]:+.2f} (n {v[1]})" for k, v in by.items()) +
          f" | leave-one-out r range [{min(loo):+.3f}, {max(loo):+.3f}]")


for xn in ["fig8_time", "turn_speed", "c_hoop_sust_speed"]:
    for yn in ["o_penetration_oe", "o_depth2s_oe", "o_pressure_oe", "g_arc_speed", "o_run_tfl_oe", "o_run_depth_oe", "o_run_tackle_oe"]:
        full(xn, cb2[xn], yn, GM[yn])
# mediation chain on players with all three
d = pd.concat([cb2.turn_speed, GM.g_arc_speed, GM.o_penetration_oe, GM.o_pressure_oe, unit], axis=1).reindex(ids).dropna()
d.columns = ["hoop", "arc", "pen", "poe", "unit"]
import numpy.linalg as la
def ols(y, X):
    X = np.column_stack([np.ones(len(X)), X]); b = la.lstsq(X, y, rcond=None)[0]; return b
print("\nmediation (n %d): hoop->pen total b %.3f; with NFL arc speed in model: hoop b %.3f, arc b %.3f" % (
    len(d), ols(d.pen, np.column_stack([d.hoop, d.unit]))[1], *ols(d.pen, np.column_stack([d.hoop, d.arc, d.unit]))[1:3]))
print("r(penetration_oe, pressure_oe) player level = %.2f" % d[["pen", "poe"]].corr().iloc[0, 1])
