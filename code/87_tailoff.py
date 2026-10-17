"""Why do some drill attempts tail off? Isolate them and characterize the slowdown (no ball is tracked in this drill).

Per attempt of the 45-degree drill (frozen phase marks): after the drive top, the rest of the clip.
  assessable  >= 0.4 s of clip after the top            tail_off  speed falls >= 1.5 yd/s below the top before the clip ends
  t_decel     time (from turn end) when speed first drops 0.5 below the top (the frozen stop rule)
  d_decel     yards run from the turn end to that moment     dir_change  heading change from the top to the slowest point
  kind        "comes back / turns" (dir_change > 45 deg: looks like adjusting to a ball or the drill's end cone) vs
              "straight-line pull-up" (dir_change <= 45 deg: brakes on the same line)
Questions: how common by combine year; when/where it starts (consistent spot = drill design, variable = player);
is it repeatable within a player (attempt 1 vs 2); does it pull Drive+ down (early pull-ups cap the top); and does an
"early pull-up" tendency say anything on Sundays (zone-match closing, completion when targeted).
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.argv = ["x", "none"]; ns = {}; exec(open(f"{ROOT}/code/80_visuals.py").read(), ns)
att, GR, base = ns["drill_attempts"](), ns["GR"], ns["base"]
rows = []
for r in att:
    s, x, y, t, kt, j1 = r["s"], r["x"], r["y"], r["t"], r["jt"], r["j1"]
    rest = s[kt:]; after = t[-1] - t[kt]
    rec = {"nfl_id": r["nfl_id"], "event_id": r["event_id"], "top": r["top"], "t_top": t[kt], "clip_after_top": after, "assessable": after >= 0.4}
    if rec["assessable"]:
        drop = s[kt] - rest.min(); jmin = kt + int(np.argmin(rest))
        dec = np.where(rest < s[kt] - 0.5)[0]
        jd = kt + dec[0] if len(dec) else None
        h = np.degrees(np.unwrap(np.arctan2(np.gradient(x), np.gradient(y))))
        rec.update({"spd_drop": drop, "tail_off": drop >= 1.5, "t_decel": t[jd] if jd is not None else np.nan,
                    "d_decel": np.hypot(np.diff(x[j1:jd + 1]), np.diff(y[j1:jd + 1])).sum() if jd is not None else np.nan,
                    "dir_change": abs(h[jmin] - h[kt]), "end_speed": s[-1], "min_after": rest.min()})
    rows.append(rec)
A = pd.DataFrame(rows)
A["yr"] = A.nfl_id.map(base.draft_year); A["pos"] = A.nfl_id.map(base.nfl_position); A["dp"] = A.nfl_id.map(GR.drive_plus)
A["kind"] = np.where(~A.assessable, "clip ends near the top", np.where(~A.tail_off.fillna(False), "holds speed", np.where(A.dir_change > 45, "slows and turns", "straight-line pull-up")))
pd.set_option("display.width", 220)
print("=== how common (all DBs, 224 attempts) ===")
print(pd.crosstab(A.yr, A.kind, margins=True).to_string())
T = A[A.tail_off == True]
print(f"\n=== the {len(T)} tail-offs: where and when they start ===")
print(f"  slowdown starts {T.t_decel.median():.2f} s after the turn (IQR {T.t_decel.quantile(.25):.2f}-{T.t_decel.quantile(.75):.2f}), "
      f"after {T.d_decel.median():.1f} yd of drive (IQR {T.d_decel.quantile(.25):.1f}-{T.d_decel.quantile(.75):.1f})")
print(f"  for comparison, attempts that hold speed reach their top at {A[A.kind == 'holds speed'].t_top.median():.2f} s after the turn")
print(f"  speed lost: median {T.spd_drop.median():.1f} yd/s; heading change while slowing: median {T.dir_change.median():.0f} deg; "
      f"share turning > 45 deg: {(T.dir_change > 45).mean():.2f}; share turning > 90 deg: {(T.dir_change > 90).mean():.2f}")
print("  by year:", T.groupby("yr").agg(n=("spd_drop", "size"), t_decel=("t_decel", "median"), d_decel=("d_decel", "median"), turn=("dir_change", "median")).round(2).to_dict("index"))
# repeatability
P = A[A.assessable].copy(); P["k"] = P.groupby("nfl_id").cumcount()
two = P[P.k <= 1].pivot(index="nfl_id", columns="k", values="tail_off").dropna().astype(float)
if len(two) >= 10:
    both, one, none = int((two.sum(1) == 2).sum()), int((two.sum(1) == 1).sum()), int((two.sum(1) == 0).sum())
    pt = two.values.astype(float).mean(); exp_both = len(two) * pt * pt
    print(f"\n=== repeatable? players with two assessable attempts: {len(two)} | tail off both times {both} (chance alone ~{exp_both:.1f}), once {one}, never {none}")
    print(f"  attempt-to-attempt agreement r {np.corrcoef(two[0].astype(float), two[1].astype(float))[0, 1]:+.2f}")
    t2 = P[P.k <= 1].pivot(index="nfl_id", columns="k", values="t_decel").dropna()
    if len(t2) >= 8:
        print(f"  when both tail off, slowdown timing attempt 1 vs 2: r {t2.corr().iloc[0, 1]:+.2f} (n {len(t2)})")
# does it pull Drive+ down?
print("\n=== does it pull Drive+ down? (within-year z of the attempt's top) ===")
A["top_z"] = A.groupby("yr").top.transform(lambda v: (v - v.mean()) / v.std())
print(A.groupby("kind").agg(attempts=("top", "size"), top_z=("top_z", "mean"), t_top=("t_top", "median")).round(2).to_string())
early = A[(A.tail_off == True) & (A.t_decel < 0.8)]
print(f"  early pull-ups (slowing before 0.8 s, i.e. before steps 3-5): {len(early)} attempts, mean top z {early.top_z.mean():+.2f}")
# (tail-off is universal in the long-clip years, so a per-player tail-off rate has no variance to test on Sundays)
A.to_csv(f"{ROOT}/results/87_tailoff_attempts.csv", index=False)
print("\nexamples (CBs, biggest drops):")
ex = A[(A.pos == "CB") & (A.tail_off == True)].sort_values("spd_drop", ascending=False).head(6)
print(ex.assign(name=ex.nfl_id.map(GR.display_name))[["name", "yr", "dp", "top", "t_decel", "d_decel", "spd_drop", "dir_change", "kind"]].round(2).to_string(index=False))
