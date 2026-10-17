"""Harden the Drive+ measurement: robustness of the grade to every construction choice, and which version is most reliable.

Per attempt of the 45-degree drill (all DBs), the frozen phase logic with a configurable speed source / smoothing:
  V0  frozen: Savitzky-Golay window 7, top = running max for <= 2 s after the turn (stop 0.5 below)
  V1  truncation-safe: top searched only within 1.0 s of the turn end (every year's clips cover this)
  V2  regressed for attempts: V0 z x reliability(k attempts) -- Spearman-Brown from the single-attempt r
  V3  best attempt instead of the mean
  V4  SG window 5 (less smoothing)      V5  SG window 9 (more smoothing)      V6  vendor speed column s (no smoothing)
For each: single-attempt test-retest r (attempt 1 vs 2), rank correlation of player grades with V0, and validity:
  zone-matching closing (3-yd reps), completion when targeted (3-yd), ball-play rate per pass snap (81b play-by-play credits)
  -- slope per SD with player-permutation p. Output data/features/drive_plus_variants.parquet
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(179)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
d = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")].copy()
d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"])


def attempt(t, x, y, sraw, win=7, source="sg", window_after=20):
    if len(t) < 12:
        return None
    dt = np.median(np.diff(t))
    vx = savgol_filter(x, win, 2, deriv=1, delta=dt); vy = savgol_filter(y, win, 2, deriv=1, delta=dt)
    s = np.hypot(vx, vy) if source == "sg" else sraw
    hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
    on = int(np.argmax(s >= 1.0)); run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
    if len(run) < 4:
        return None
    dev = np.abs(hd - np.median(hd[run[:4]]))
    st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
    if not len(st):
        return None
    j0 = st[0]; w = slice(j0, min(j0 + 15, len(s))); dmax = dev[w].max()
    if dmax < 25 or dmax > 90:
        return None
    j1 = j0 + int(np.argmax(dev[w] >= 0.9 * dmax))
    k, jend, top = j1, min(j1 + window_after, len(s) - 1), s[j1]
    while k < jend:
        k += 1
        if s[k] > top:
            top = s[k]
        elif s[k] < top - 0.5:
            break
    return float(top)


CFG = {"V0": dict(), "V1": dict(window_after=10), "V4": dict(win=5), "V5": dict(win=9), "V6": dict(source="raw")}
rows = []
for eid, g in d.groupby("event_id"):
    t = (g.time - g.time.iat[0]).dt.total_seconds().values; x, y, sr = g.x.values.astype(float), g.y.values.astype(float), g.s.values.astype(float)
    rec = {"event_id": eid, "nfl_id": g.nfl_id.iat[0], "yr": g.draft_year.iat[0], "t0": g.time.iat[0]}
    for k_, kw in CFG.items():
        rec[k_] = attempt(t, x, y, sr, **kw)
    rows.append(rec)
A = pd.DataFrame(rows).sort_values(["nfl_id", "t0"])
for k_ in CFG:
    A[k_] = pd.to_numeric(A[k_])
A["att"] = A.groupby("nfl_id").cumcount()
print("valid attempts per variant:", {k_: int(A[k_].notna().sum()) for k_ in CFG}, "| players:", A.dropna(subset=["V0"]).nfl_id.nunique())
nv = A.dropna(subset=["V0"]).groupby("nfl_id").size()
print("players by number of valid frozen attempts:", nv.value_counts().sort_index().to_dict())
cbs = base.index[base.nfl_position == "CB"]


def zclass(s):
    yr = base.draft_year.reindex(s.index); return (s - s.groupby(yr).transform("mean")) / s.groupby(yr).transform("std")


P = pd.DataFrame(index=A.nfl_id.unique())
rel1 = {}
for k_ in CFG:
    v = A.dropna(subset=[k_]); v = v.assign(att2=v.groupby("nfl_id").cumcount())
    two = v[v.att2 <= 1].pivot(index="nfl_id", columns="att2", values=k_).dropna()
    rel1[k_] = np.corrcoef(zclass(two[0]), zclass(two[1]))[0, 1]
    P[k_] = zclass(v.groupby("nfl_id")[k_].mean())
r1 = rel1["V0"]; k = A.dropna(subset=["V0"]).groupby("nfl_id").size()
P["n_att"] = k
P["V2"] = P.V0 * (k * r1 / (1 + (k - 1) * r1)).reindex(P.index)
P["V3"] = zclass(A.dropna(subset=["V0"]).groupby("nfl_id").V0.max())
rel1["V2"] = rel1["V0"]; rel1["V3"] = np.nan
VAR = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
for v in VAR:                                                     # Drive+ scale: CB reference
    ref = P[v].reindex(cbs).dropna(); P[v + "_plus"] = 100 + 10 * (P[v] - ref.mean()) / ref.std()
P.to_parquet(f"{ROOT}/data/features/drive_plus_variants.parquet")
LAB = {"V0": "frozen", "V1": "top within 1.0 s (truncation-safe)", "V2": "regressed for # attempts", "V3": "best attempt",
       "V4": "less smoothing (SG 5)", "V5": "more smoothing (SG 9)", "V6": "vendor speed, no smoothing"}

# ---------------- validity
reps = pd.read_parquet(f"{ROOT}/data/outcomes/79_scored_coverage_reps.parquet")
pbp = pd.read_parquet(f"{ROOT}/data/outcomes/81b_pbp_credits.parquet")
CL = lambda x, c: pd.factorize(x[c])[0]


def slope(x, y, idcol, term, ctx, n_perm=300):
    x = x.dropna(subset=[y, term]).copy(); f = f"{y} ~ {term} + {ctx}"
    t0 = smf.ols(f, data=x).fit().tvalues[term]; b = smf.ols(f, data=x).fit().params[term]
    vals = x.drop_duplicates(idcol).set_index(idcol)[term]; null = []
    for _ in range(n_perm):
        xx = x.copy(); xx[term] = xx[idcol].map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=xx).fit().tvalues[term])
    return b, (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1)


out = []
for v in VAR:
    z = (P[v + "_plus"] - 100) / 10
    rr = reps.assign(T=reps.nfl_id_d.map(z)); pb = pbp.assign(T=pbp.nfl_id.map(z))
    row = {"variant": LAB[v], "single-attempt r": rel1[v], "rank r with frozen (CBs)": spearmanr(P[v].reindex(cbs), P["V0"].reindex(cbs), nan_policy="omit")[0],
           "CBs graded": int(P[v].reindex(cbs).notna().sum())}
    for nm, x, y, idc, ctx, sc in [("zone-match closing", rr[rr.cell == "zone-matching"], "db_closing", "nfl_id_d", "wr_skill + C(route_ran) + is_s", 1),
                                   ("completion (targets)", rr[rr.targeted == 1], "completion", "nfl_id_d", "C(team_coverage_man_zone) + C(route_ran) + is_s", 100),
                                   ("ball plays per 100 snaps", pb, "ball_play", "nfl_id", "is_s + C(aln) + zone + C(season)", 100)]:
        b, p = slope(x, y, idc, "T", ctx)
        row[f"{nm}: per 10"] = b * sc; row[f"{nm}: p"] = p
    out.append(row); print("  ", {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items()}, flush=True)
R = pd.DataFrame(out).set_index("variant"); R.to_csv(f"{ROOT}/results/81a_metric_hardening.csv")
pd.set_option("display.width", 250)
print("\n" + R.round(3).to_string())
