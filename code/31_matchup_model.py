"""Does the rusher's combine bend vs the blocker's combine quickness predict who wins the rep?

Reps: primary engaged DL-OL reps (30_matchups.py), REG + POST, not nullified.
Rusher: hoop turn speed (duration-weighted mean speed through the Run-the-Hoop turns, averaged over attempts).
Blocker: lateral quickness = mean speed through the five-yard wave (slide & shuffle) drill; also mirror-drill speed,
arm length. All combine traits z-scored across the players in the rep sample.
Outcomes: blocker's peak pressure probability allowed (ppa), rusher turned the corner (geometric), blocker pressure
allowed, rusher pressure. Linear models (binary outcomes as linear probability, coefficients in probability points),
two-way cluster-robust SE (rusher and blocker). Context: alignments, down, distance bucket, blitz.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
rep = pd.read_parquet(f"{ROOT}/data/bend/matchups.parquet")
rep = rep[rep.primary & rep.season_type.isin(["REG", "POST"]) & (rep.nullified != "Y")].copy()
rep["ytg_b"] = pd.cut(rep.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)

# rusher: hoop turn speed
T = pd.read_parquet(f"{ROOT}/data/bend/combine_bend_turns.parquet")
T = T[T.family == "hoop"]
hoop = T.groupby(["nfl_id", "event_id"]).apply(lambda d: np.average(d.sust_speed, weights=d.dur), include_groups=False).groupby("nfl_id").mean()
# blocker: wave / mirror speed
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
w = ct[(ct.entity_type == "PLAYER") & ct.drill_name.isin(["FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", "PASS_PRO_MIRROR_DRILL", "PASS_PRO-MIRROR_DRILL"])]
rows = []
for eid, g in w.sort_values(["event_id", "time"]).groupby("event_id"):
    s = g.s.values.astype(float); on = int(np.argmax(s >= 1.0))
    rows.append({"nfl_id": g.nfl_id.iat[0], "fam": "wave" if "WAVE" in g.drill_name.iat[0] else "mirror", "speed": s[on:].mean()})
q = pd.DataFrame(rows).groupby(["nfl_id", "fam"]).speed.mean().unstack()

rep["dl_hoop"] = rep.dl_id.map(hoop); rep["ol_wave"] = rep.ol_id.map(q["wave"]); rep["ol_mirror"] = rep.ol_id.map(q["mirror"])
rep["ol_arm"] = rep.ol_id.map(base.arm_length); rep["dl_arm"] = rep.dl_id.map(base.arm_length)
rep["dl_w"] = rep.dl_id.map(base.combine_weight); rep["ol_w"] = rep.ol_id.map(base.combine_weight)
rep["dl_ten"] = rep.dl_id.map(base.ten_yd_split); rep["ol_ten"] = rep.ol_id.map(base.ten_yd_split)


def zp(col, id_col):
    """z-score a player trait across the distinct players in the rep sample (not weighted by reps)."""
    v = rep.drop_duplicates(id_col).set_index(id_col)[col]
    return (rep[col] - v.mean()) / v.std()


for c, idc in [("dl_hoop", "dl_id"), ("ol_wave", "ol_id"), ("ol_mirror", "ol_id"), ("ol_arm", "ol_id"), ("dl_arm", "dl_id"),
               ("dl_w", "dl_id"), ("ol_w", "ol_id"), ("dl_ten", "dl_id"), ("ol_ten", "ol_id")]:
    rep["z_" + c] = zp(c, idc)
rep["advantage"] = rep.z_dl_hoop - rep.z_ol_wave
for c in ["corner_turned", "ol_pressure_allowed", "dl_pressure"]:
    rep[c] = rep[c].astype(float)
CTX = "C(dl_align) + C(ol_align) + C(down) + C(ytg_b) + C(blitzing)"


def fit(y, rhs, data, label):
    import re
    used = [c for c in data.columns if re.search(rf"\b{re.escape(c)}\b", rhs)]
    d = data.dropna(subset=[y] + used + ["down", "ytg_b", "blitzing", "dl_align", "ol_align"])
    g = np.column_stack([pd.factorize(d.dl_id)[0], pd.factorize(d.ol_id)[0]])
    mdl = smf.ols(f"{y} ~ {rhs} + {CTX}", data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
    terms = [t for t in mdl.params.index if t.startswith("z_") or t == "advantage" or ":" in t]
    out = {"label": label, "outcome": y, "reps": len(d), "rushers": d.dl_id.nunique(), "blockers": d.ol_id.nunique(), "mean_y": d[y].mean()}
    for t in terms:
        out[t] = f"{mdl.params[t]:+.4f} (p {mdl.pvalues[t]:.3f})"
    return out


res = []
for y in ["ppa", "corner_turned", "ol_pressure_allowed", "dl_pressure"]:
    res.append(fit(y, "z_dl_hoop + z_ol_wave", rep, "all reps: bend vs quickness"))
    res.append(fit(y, "z_dl_hoop * z_ol_wave", rep, "all reps: + interaction"))
    res.append(fit(y, "z_dl_hoop + z_ol_wave + z_dl_w + z_ol_w + z_ol_arm", rep, "all reps: + both weights + OL arm"))
    res.append(fit(y, "advantage", rep, "all reps: advantage = rusher bend z - blocker quickness z"))
    te = rep[(rep.ol_align == "T") & (rep.dl_align == "EDGE")]
    res.append(fit(y, "z_dl_hoop + z_ol_wave", te, "tackle vs edge: bend vs quickness"))
    res.append(fit(y, "z_dl_hoop + z_ol_wave + z_dl_w + z_ol_w + z_ol_arm", te, "tackle vs edge: + weights + OL arm"))
    ii = rep[(rep.dl_align == "INTERIOR_LINE") & rep.ol_align.isin(["G", "C"])]
    res.append(fit(y, "z_dl_hoop + z_ol_wave", ii, "interior: bend vs quickness"))
R = pd.DataFrame(res)
R.to_csv(f"{ROOT}/results/31_matchup_model.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
for y in R.outcome.unique():
    print(f"\n===== outcome: {y} =====")
    print(R[R.outcome == y].drop(columns="outcome").dropna(axis=1, how="all").to_string(index=False))
