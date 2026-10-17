"""Improvements that stay inside the current data.

(a) Recover rejected combine attempts: the frozen rule skips an attempt when no turn is found or the max deviation is
    outside 25-90 deg. Relaxed detector (start speed 2.5 yd/s, deviation 20-110 deg, onset at 0.8 yd/s). Count recovered
    attempts, check they agree with the frozen attempt of the same player, and test validity with them included.
(b) Multi-frame drive index: per attempt, mean within-year z of v_06, v_08, v_10, top, d_10 (72a). Attempt 1 vs 2
    reliability vs the single-frame top; validity.
(c) Large-N separation criterion: NGS separation_at_pass_forward of the targeted receiver on every pass with a graded
    CB on a clear side (76b/77 rows). Drive+ slope on throws TOWARD his side vs AWAY (placebo), zone and all, with
    defense/offense team-season FE and player-permutation p.
Validity target throughout: the composite coverage score on the wide reps (90), leave-one-class-out MSE reduction.
"""
import sys, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; rng = np.random.default_rng(233)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
cbs = base.index[base.nfl_position == "CB"]
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet"); W["cls"] = W.nfl_id_d.map(base.draft_year)
for y, sgn in {"sep_gain_break": -1, "db_closing": +1, "sep_at_throw": -1, "open": -1}.items():
    d = W.dropna(subset=[y, "wr_skill"]); W.loc[d.index, y + "_r"] = smf.ols(f"{y} ~ wr_skill + C(zone) + C(route_ran) + is_s", data=d).fit().resid
    W[y + "_z"] = sgn * (W[y + "_r"] - W[y + "_r"].mean()) / W[y + "_r"].std()
W["composite"] = W[[c for c in W.columns if c.endswith("_z")]].mean(axis=1, skipna=True)


def zclass(s):
    yr = base.draft_year.reindex(s.index); return (s - s.groupby(yr).transform("mean")) / s.groupby(yr).transform("std")


def oos(d, y, col):
    d = d.dropna(subset=[y, col]).copy(); gain = tot = 0.0
    for c in sorted(d.cls.unique()):
        a, b = d[d.cls != c], d[d.cls == c]; m0 = smf.ols(f"{y} ~ is_s", data=a).fit(); m1 = smf.ols(f"{y} ~ is_s + {col}", data=a).fit()
        gain += ((b[y] - m0.predict(b)) ** 2).sum() - ((b[y] - m1.predict(b)) ** 2).sum(); tot += ((b[y] - m0.predict(b)) ** 2).sum()
    return 100 * gain / tot


def validity(series, label):
    v = series.reindex(W.nfl_id_d.unique()); W["T_"] = W.nfl_id_d.map((series - v.mean()) / v.std())
    d = W.dropna(subset=["composite", "T_"]); m = smf.ols("composite ~ T_ + is_s", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]})
    t = W[W.targeted == 1]
    print(f"    {label:34s} composite slope {m.params['T_']:+.3f} (t {m.tvalues['T_']:+.1f}) | LOCO composite {oos(W, 'composite', 'T_'):+.2f}% | LOCO completion {oos(t, 'completion', 'T_'):+.2f}% | DBs {d.nfl_id_d.nunique()}")


# ---------------- (a) recover rejected attempts
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
d = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")].copy()
d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"])


def top_of(t, x, y, on_speed=1.0, start_speed=3.0, lo=25, hi=90):
    if len(t) < 12: return None, "short"
    dt = np.median(np.diff(t)); vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    if not (s >= on_speed).any(): return None, "no onset"
    on = int(np.argmax(s >= on_speed)); run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
    if len(run) < 4: return None, "no backpedal"
    dev = np.abs(hd - np.median(hd[run[:4]])); st = np.where((np.arange(len(s)) > run[3]) & (s >= start_speed) & (dev > 10))[0]
    if not len(st): return None, "no turn"
    j0 = st[0]; w = slice(j0, min(j0 + 15, len(s))); dmax = dev[w].max()
    if dmax < lo or dmax > hi: return None, f"angle {dmax:.0f}"
    j1 = j0 + int(np.argmax(dev[w] >= 0.9 * dmax)); k, jend, top = j1, min(j1 + 20, len(s) - 1), s[j1]
    while k < jend:
        k += 1
        if s[k] > top: top = s[k]
        elif s[k] < top - 0.5: break
    return float(top), "ok"


rows = []
for eid, g in d.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; x, y = g.x.values.astype(float), g.y.values.astype(float)
    f, why = top_of(t, x, y); r, why2 = top_of(t, x, y, on_speed=0.8, start_speed=2.5, lo=20, hi=110)
    rows.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "t0": g.time.iat[0], "frozen": f, "why": why, "relaxed": r, "why2": why2})
A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"])
rej = A[A.frozen.isna()]
print(f"(a) attempts {len(A)}; rejected by the frozen rule {len(rej)}: {rej.why.str.replace(r'angle .*', 'angle out of range', regex=True).value_counts().to_dict()}")
rec = rej[rej.relaxed.notna()]
print(f"    recovered by the relaxed detector: {len(rec)} (reasons left: {rej[rej.relaxed.isna()].why2.value_counts().to_dict()})")
A["best"] = A.frozen.fillna(A.relaxed)
one = A.groupby("nfl_id").frozen.count(); gained = A[A.nfl_id.isin(one[one == 1].index) & A.frozen.isna() & A.relaxed.notna()]
print(f"    single-attempt players who gain a second attempt: {gained.nfl_id.nunique()} of {int((one == 1).sum())}")
pair = A[A.nfl_id.isin(gained.nfl_id)].groupby("nfl_id").agg(fro=("frozen", "max"), rel=("relaxed", lambda v: v[A.loc[v.index, 'frozen'].isna()].max()))
pair = pair.dropna()
if len(pair) >= 5:
    print(f"    agreement of the recovered attempt with the player's frozen attempt: r {np.corrcoef(zclass(pair.fro), zclass(pair.rel))[0, 1]:+.2f} (n {len(pair)}); "
          f"recovered attempts are on average {(pair.rel - pair.fro).mean():+.2f} yd/s vs the frozen one")
    print("    (the frozen two-attempt agreement is r +0.49)")
P_fro = zclass(A.dropna(subset=["frozen"]).groupby("nfl_id").frozen.mean()); P_best = zclass(A.dropna(subset=["best"]).groupby("nfl_id").best.mean())
print("    validity:")
validity(P_fro, "frozen (reference)"); validity(P_best, "frozen + recovered attempts")

# ---------------- (b) multi-frame drive index
AT = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts_attempts.parquet")
AT["yr"] = AT.nfl_id.map(base.draft_year)
for c in ["v_06", "v_08", "v_10", "top", "d_10"]:
    AT[c + "_z"] = AT.groupby("yr")[c].transform(lambda v: (v - v.mean()) / v.std())
AT["index"] = AT[["v_06_z", "v_08_z", "v_10_z", "top_z", "d_10_z"]].mean(axis=1, skipna=True)
two = AT[AT.att <= 1].pivot(index="nfl_id", columns="att", values=["top_z", "index"]).dropna()
print(f"\n(b) attempt 1 vs 2 agreement: single-frame top r {np.corrcoef(two['top_z'][0], two['top_z'][1])[0, 1]:+.2f} | multi-frame index r {np.corrcoef(two['index'][0], two['index'][1])[0, 1]:+.2f} (n {len(two)})")
validity(zclass(AT.groupby("nfl_id").top.mean()), "single-frame top"); validity(AT.groupby("nfl_id")["index"].mean(), "multi-frame drive index")

# ---------------- (c) NGS separation on throws to his side
sys.argv = ["x"]; ns = {}; exec(open(f"{ROOT}/code/77_epa_decomposition.py").read().split("OUT = [")[0], ns)
dd, analyze, pp = ns["d"], ns["analyze"], ns["pp"]
tg = pp[pp.target == 1].drop_duplicates(["game_id", "play_id"]).set_index(["game_id", "play_id"])
key = pd.MultiIndex.from_arrays([dd.game_id, dd.play_id]); dd["ngs_sep"] = tg.separation_at_pass_forward.reindex(key).values
dd["ngs_open"] = np.where(dd.ngs_sep.notna(), (dd.ngs_sep >= 2).astype(float), np.nan)
print(f"\n(c) throws with NGS separation of the targeted receiver: {int(dd.ngs_sep.notna().sum())} of {len(dd)} rows; toward-side rows {int(((dd.rel == 'toward') & dd.ngs_sep.notna()).sum())}")
for lab, P in [("zone", dd[dd.team_coverage_man_zone == "ZONE_COVERAGE"]), ("zone, < 20 air yds", dd[(dd.team_coverage_man_zone == "ZONE_COVERAGE") & (dd.air < 20)]), ("all coverages", dd)]:
    for y, sc in [("ngs_sep", 1), ("ngs_open", 100)]:
        r = analyze(P, y, n_perm=500)
        print(f"    {lab:20s} {y:9s} per 10 Drive+: toward {r['toward'] * sc:+.3f} | away {r['away'] * sc:+.3f} | toward-away {r['toward-away'] * sc:+.3f} (perm p {r['perm p']:.3f}; noise SD {r['noise SD'] * sc:.3f}) | rows {r['rows']}", flush=True)
