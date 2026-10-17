"""Is the drive effect a break on the THROWN BALL? (72c: drive's completion effect is largest on throws released < 0.5 s
after the receiver's break, i.e. the closing must happen while the ball is in the air.)

Targeted CB coverage reps; raw game tracking (both players are pool players) -> frames of pass_forward (release) and
pass_arrived (arrival; else first pass_outcome_* frame):
  flight        arrival - release (s)
  d_release     receiver-CB distance at release      d_arrive  at arrival (how tight he is at the catch point)
  close_flight  (d_release - d_arrive) / flight  (yd/s, + = closing on the ball)
  cb_v_flight   CB mean speed during the flight      cb_v_arrive  CB speed at arrival
Tests: combine drive (frozen) -> each (rep level, permutation p); d_arrive / close_flight -> completion; mediation of the
drive -> completion effect through d_release (at release) vs close_flight (during flight); and the gap clock redone with
time from the break to ARRIVAL.
"""
import duckdb, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(71)
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
tg = cov[cov.targeted == 1].copy()
v = Z["top"].reindex(cov.nfl_id_d.unique()); tg["c_top"] = tg.nfl_id_d.map((Z["top"] - v.mean()) / v.std())
keys = pd.concat([tg[["game_id", "play_id", "nfl_id_r"]].rename(columns={"nfl_id_r": "nfl_id"}),
                  tg[["game_id", "play_id", "nfl_id_d"]].rename(columns={"nfl_id_d": "nfl_id"})]).drop_duplicates()
con = duckdb.connect(); con.register("k", keys)
tr = con.sql(f"""select t.game_id, t.play_id, t.nfl_id, t.time, t.x, t.y, t.s, t.event from read_parquet(['{ROOT}/data/raw/game_tracking_2023.parquet',
     '{ROOT}/data/raw/game_tracking_2024.parquet','{ROOT}/data/raw/game_tracking_2025.parquet']) t join k using (game_id, play_id, nfl_id)""").df()
tr["time"] = pd.to_datetime(tr.time); tr = tr.sort_values(["game_id", "play_id", "nfl_id", "time"])
print(f"targeted reps {len(tg)}, tracking rows {len(tr)}")
ARR = {"pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete", "pass_outcome_touchdown", "pass_outcome_interception", "pass_tipped"}
out = []
for (gid, pid), g in tr.groupby(["game_id", "play_id"]):
    for _, r in tg[(tg.game_id == gid) & (tg.play_id == pid)].iterrows():
        R, D = g[g.nfl_id == r.nfl_id_r].set_index("time"), g[g.nfl_id == r.nfl_id_d].set_index("time")
        idx = R.index.intersection(D.index)
        if len(idx) < 5:
            continue
        R, D = R.loc[idx], D.loc[idx]
        ev = R.event.fillna(D.event)
        rel = ev.index[ev == "pass_forward"]; arr = ev.index[ev.isin(["pass_arrived"])]
        if not len(arr):
            arr = ev.index[ev.isin(ARR)]
        if not len(rel) or not len(arr) or arr[0] <= rel[0]:
            continue
        t0, t1 = rel[0], arr[0]
        dist = np.hypot(R.x - D.x, R.y - D.y)
        fl = (t1 - t0).total_seconds(); w = (D.index >= t0) & (D.index <= t1)
        snap = ev.index[ev == "ball_snap"]
        out.append({"game_id": gid, "play_id": pid, "nfl_id_d": r.nfl_id_d, "flight": fl, "d_release": dist[t0], "d_arrive": dist[t1],
                    "close_flight": (dist[t0] - dist[t1]) / fl, "cb_v_flight": D.s[w].mean(), "cb_v_arrive": D.s[t1],
                    "t_arrive_snap": (t1 - snap[0]).total_seconds() if len(snap) else np.nan})
F = pd.DataFrame(out)
tg = tg.merge(F, on=["game_id", "play_id", "nfl_id_d"], how="inner")
tg["gap_arrive"] = tg.t_arrive_snap - tg.t_break
print(f"reps with release + arrival {len(tg)}, CBs {tg.nfl_id_d.nunique()} | medians:", tg[["flight", "d_release", "d_arrive", "close_flight", "gap_arrive"]].median().round(2).to_dict())
tg.to_parquet(f"{ROOT}/data/outcomes/cb_targets_flight.parquet", index=False)
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def eff(d, y, ctx, term="c_top", n_perm=400):
    d = d.dropna(subset=[y, term]).copy(); f = f"{y} ~ {term} + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd[term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return m.params[term], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d)


CTX = "C(team_coverage_man_zone) + flight"
print("\n=== combine drive (per SD) -> ball-flight mechanics (targeted reps) ===")
for y in ["d_release", "d_arrive", "close_flight", "cb_v_flight", "cb_v_arrive"]:
    b, p, n = eff(tg, y, CTX)
    print(f"  {y:13s} {b:+.3f} (perm p {p:.3f}, n {n})")
print("\n=== completion (logit) on the ball-flight mechanics, standardized ===")
d = tg.dropna(subset=["completion", "d_release", "close_flight", "c_top"]).copy()
for c in ["d_release", "close_flight", "d_arrive"]:
    d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
for f in ["completion ~ c_top + C(team_coverage_man_zone)",
          "completion ~ c_top + z_d_release + C(team_coverage_man_zone)",
          "completion ~ c_top + z_d_release + z_close_flight + flight + C(team_coverage_man_zone)",
          "completion ~ c_top + z_d_arrive + flight + C(team_coverage_man_zone)"]:
    m = smf.logit(f, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]})
    print(f"  {f.split('~')[1].replace(' + C(team_coverage_man_zone)', ''):55s} " + " | ".join(f"{t} {m.params[t]:+.3f} (p {m.pvalues[t]:.3f})" for t in m.params.index if t.startswith(("c_top", "z_", "flight"))))
print("\n=== the drive effect by time from the receiver's break to the ball's ARRIVAL ===")
d["gap_bin"] = pd.cut(d.gap_arrive, [-9, 1.0, 1.6, 99], labels=["<1.0 s", "1.0-1.6 s", ">1.6 s"])
for gb, dd in d.groupby("gap_bin"):
    if len(dd) < 40:
        continue
    m = smf.logit("completion ~ c_top + C(team_coverage_man_zone)", data=dd).fit(disp=0)
    b, p, n = eff(dd, "close_flight", CTX, n_perm=200)
    print(f"  arrival {gb:9s} after break: completion logit c_top {m.params['c_top']:+.3f} (p {m.pvalues['c_top']:.3f}) | drive -> closing in flight {b:+.3f} (perm p {p:.3f}) | n {len(dd)}")
