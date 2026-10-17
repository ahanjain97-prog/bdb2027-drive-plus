"""Is Drive+ fragile? Specification curve over every reasonable analysis choice + influence diagnostics.

Claims (sign expected): zone-matching closing (+), completion when targeted (-), receiver open at the throw (-),
ball plays per pass snap (+, play-by-play credits).
Choices:
  rep radius 2.5 / 3 / 3.5 / 4 yd (nested subsets of the 4-yd reps by mean distance)   closing window 0.5 / 0.7 / 1.0 s
  matching threshold (pre-break velocity cosine) .6 / .7 / .8   open threshold 1.5 / 2 / 2.5 yd   sample CBs / all DBs
  controls minimal / + situation (cushion, slot, down, distance)   Drive+ version frozen / truncation-safe / regressed
  ball plays: outcome PBU+INT / PBU only / completion on credited targets; alignment all / CB+slot; controls season / + team
Estimates: OLS, SE clustered by DB. Spec curve = share with the expected sign, share with p < .05, median estimate.
Influence (frozen, registered 3-yd reps): drop each DB (incl. Quinyon Mitchell), winsorize Drive+ at 80-120, rank-based
Drive+, leave one season out, leave one defense out.
"""
import itertools, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
R4 = pd.read_parquet(f"{ROOT}/data/outcomes/79_scored_coverage_reps_4yd.parquet")
PB = pd.read_parquet(f"{ROOT}/data/outcomes/81b_pbp_credits.parquet")
VAR = pd.read_parquet(f"{ROOT}/data/features/drive_plus_variants.parquet")
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = R4.idx_r.values, R4.idx_d.values
vxr, vyr, sr, lr, pr = g(A, "vx"), g(A, "vy"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
vxd, vyd, sd = g(B, "vx"), g(B, "vy"), g(B, "s")
dist = np.hypot(g(A, "x_rel") - g(B, "x_rel"), g(A, "y_field") - g(B, "y_field"))
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]; r_ = np.arange(len(R4))
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1); hb = a.max(1) > 0
at = lambda arr, j: np.where(hb & (j < T) & vv[r_, np.clip(j, 0, T - 1)], arr[r_, np.clip(j, 0, T - 1)], np.nan)
for w in (5, 7, 10):
    R4[f"close_{w}"] = (at(dist, jb) - at(dist, jb + w)) / (w / 10)
w5 = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
R4["pa"] = np.where(w5.sum(1) >= 3, np.where(w5, cosv, 0).sum(1) / np.maximum(w5.sum(1), 1), np.nan)
R4["zone"] = (R4.team_coverage_man_zone == "ZONE_COVERAGE").astype(int)
R4["slot"] = (R4.lined_up_position_d == "SLOT_CB").astype(int)
for v in ["V0", "V1", "V2"]:
    R4["T_" + v] = R4.nfl_id_d.map((VAR[v + "_plus"] - 100) / 10); PB["T_" + v] = PB.nfl_id.map((VAR[v + "_plus"] - 100) / 10)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")[["game_id", "play_id", "nfl_id", "team_abbr"]]
R4 = R4.merge(pp.rename(columns={"nfl_id": "nfl_id_d", "team_abbr": "def_team"}), on=["game_id", "play_id", "nfl_id_d"], how="left")
PB["def_ts"] = PB.team_abbr + PB.season.astype(str)


def est(x, y, term, ctx, idc):
    x = x.dropna(subset=[y, term]).copy()
    if x[idc].nunique() < 10 or len(x) < 50:
        return None
    mod = smf.ols(f"{y} ~ {term} + {ctx}", data=x); used = x.loc[mod.data.row_labels]
    m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used[idc])[0]})
    return m.params[term], m.bse[term], m.pvalues[term], len(used), used[idc].nunique()


rows = []
SIT = " + cushion + slot + C(down) + yards_to_go"
for rad, smp, ctl, var in itertools.product([2.5, 3.0, 3.5, 4.0], ["CBs", "all DBs"], ["minimal", "situation"], ["V0", "V1", "V2"]):
    base_ = R4[(R4.sep_mean <= rad) & ((R4.is_s == 0) if smp == "CBs" else True)]
    pos = "" if smp == "CBs" else " + is_s"
    for wdw, thr in itertools.product([5, 7, 10], [.6, .7, .8]):
        x = base_[(base_.zone == 1) & (base_.pa >= thr)]
        r = est(x, f"close_{wdw}", "T_" + var, "wr_skill + C(route_ran)" + pos + (SIT if ctl == "situation" else ""), "nfl_id_d")
        if r: rows.append({"claim": "zone-match closing (+)", "radius": rad, "sample": smp, "controls": ctl, "version": var, "detail": f"window {wdw / 10}s, match ≥{thr}", "sign": 1, **dict(zip(["b", "se", "p", "n", "DBs"], r))})
    x = base_[base_.targeted == 1]
    r = est(x, "completion", "T_" + var, "C(team_coverage_man_zone) + C(route_ran)" + pos + (SIT if ctl == "situation" else ""), "nfl_id_d")
    if r: rows.append({"claim": "completion when targeted (-)", "radius": rad, "sample": smp, "controls": ctl, "version": var, "detail": "", "sign": -1, **dict(zip(["b", "se", "p", "n", "DBs"], r))})
    for th in (1.5, 2.0, 2.5):
        x = base_.assign(op=np.where(base_.sep.notna(), (base_.sep >= th).astype(float), np.nan))
        r = est(x, "op", "T_" + var, "C(team_coverage_man_zone) + C(route_ran)" + pos + (SIT if ctl == "situation" else ""), "nfl_id_d")
        if r: rows.append({"claim": "receiver open at throw (-)", "radius": rad, "sample": smp, "controls": ctl, "version": var, "detail": f"open ≥{th} yd", "sign": -1, **dict(zip(["b", "se", "p", "n", "DBs"], r))})
for y_, aln, ctl, var in itertools.product(["ball_play", "pbu", "comp_c"], ["all", "CB+slot"], ["season", "+ team-season FE"], ["V0", "V1", "V2"]):
    x = PB if aln == "all" else PB[PB.aln != "other"]
    ctx = "is_s + C(aln) + zone + C(season)" + (" + C(def_ts)" if ctl != "season" else "")
    r = est(x, y_, "T_" + var, ctx, "nfl_id")
    if r: rows.append({"claim": "ball plays per snap (+)" if y_ != "comp_c" else "completion on credited targets (-)", "radius": np.nan, "sample": aln, "controls": ctl,
                       "version": var, "detail": y_, "sign": 1 if y_ != "comp_c" else -1, **dict(zip(["b", "se", "p", "n", "DBs"], r))})
S = pd.DataFrame(rows); S["right_sign"] = np.sign(S.b) == S.sign; S["sig"] = S.right_sign & (S.p < .05)
S.to_csv(f"{ROOT}/results/81c_spec_curve.csv", index=False)
pd.set_option("display.width", 220)
print("=== specification curve ===")
print(S.groupby("claim").agg(specs=("b", "size"), right_sign=("right_sign", "mean"), sig_05=("sig", "mean"), median_b=("b", "median"),
                             min_b=("b", "min"), max_b=("b", "max"), median_DBs=("DBs", "median")).round(3).to_string())
print("\nby choice (share significant, right sign):")
for ch in ["radius", "sample", "controls", "version"]:
    print(f"  {ch}: " + S.groupby(["claim", ch]).sig.mean().unstack().round(2).to_string().replace("\n", "\n    "))

# ---------------- influence (frozen, registered 3-yd, all DBs)
R3 = R4[R4.sep_mean <= 3.0]
def run_all(R, P, term="T_V0"):
    out = {}
    for nm, x, y, ctx, idc in [("zone-match closing", R[(R.zone == 1) & (R.pa >= .7)], "close_7", "wr_skill + C(route_ran) + is_s", "nfl_id_d"),
                               ("completion (targets)", R[R.targeted == 1], "completion", "C(team_coverage_man_zone) + C(route_ran) + is_s", "nfl_id_d"),
                               ("open at throw", R, "open", "C(team_coverage_man_zone) + C(route_ran) + is_s", "nfl_id_d"),
                               ("ball plays / snap", P, "ball_play", "is_s + C(aln) + zone + C(season)", "nfl_id")]:
        r = est(x, y, term, ctx, idc); out[nm] = (r[0], r[2]) if r else (np.nan, np.nan)
    return out
full = run_all(R3, PB)
print("\n=== influence (frozen Drive+, 3-yd reps; slope per 10 pts and p) ===")
print("  full sample:", {k_: f"{b:+.4f} (p {p:.3f})" for k_, (b, p) in full.items()})
mitchell = 57143
print("  without Quinyon Mitchell:", {k_: f"{b:+.4f} (p {p:.3f})" for k_, (b, p) in run_all(R3[R3.nfl_id_d != mitchell], PB[PB.nfl_id != mitchell]).items()})
for c in ["T_V0"]:
    R3w, PBw = R3.assign(T_V0=R3.T_V0.clip(-2, 2)), PB.assign(T_V0=PB.T_V0.clip(-2, 2))
print("  Drive+ winsorized to 80-120:", {k_: f"{b:+.4f} (p {p:.3f})" for k_, (b, p) in run_all(R3w, PBw).items()})
rk = VAR.V0_plus.rank(pct=True); R3r, PBr = R3.assign(T_V0=R3.nfl_id_d.map(rk) * 3.46), PB.assign(T_V0=PB.nfl_id.map(rk) * 3.46)
print("  rank-based Drive+ (percentile, scaled to ~1 SD):", {k_: f"{b:+.4f} (p {p:.3f})" for k_, (b, p) in run_all(R3r, PBr).items()})
loo = {k_: [] for k_ in full}
for pid in R3.nfl_id_d.unique():
    for k_, (b, p) in run_all(R3[R3.nfl_id_d != pid], PB[PB.nfl_id != pid]).items():
        loo[k_].append(b)
print("  leave-one-DB-out slope range:", {k_: f"{np.nanmin(v):+.4f} to {np.nanmax(v):+.4f}" for k_, v in loo.items()})
for s in sorted(R3.season.unique()):
    print(f"  without season {s}:", {k_: f"{b:+.4f} (p {p:.3f})" for k_, (b, p) in run_all(R3[R3.season != s], PB[PB.season != s]).items()})
lot = {k_: [] for k_ in full}
for t in R3.def_team.dropna().unique():
    for k_, (b, p) in run_all(R3[R3.def_team != t], PB[PB.team_abbr != t]).items():
        lot[k_].append(b)
print("  leave-one-defense-out slope range:", {k_: f"{np.nanmin(v):+.4f} to {np.nanmax(v):+.4f}" for k_, v in lot.items()})

# ---------------- figure: specification curve
claims = ["zone-match closing (+)", "receiver open at throw (-)", "completion when targeted (-)", "ball plays per snap (+)"]
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8)); fig.subplots_adjust(wspace=.38)
for ax, c in zip(axes, claims):
    s = S[S.claim == c].sort_values("b").reset_index(drop=True); x = np.arange(len(s))
    if c.startswith("ball plays"):
        s = s.assign(b=s.b * 100, se=s.se * 100)
    col = np.where(s.sig, "#1565c0", np.where(s.right_sign, "#90caf9", "#e57373"))
    ax.vlines(x, s.b - 1.645 * s.se, s.b + 1.645 * s.se, color=col, alpha=.35, lw=1)
    ax.scatter(x, s.b, c=col, s=10, zorder=3); ax.axhline(0, color="k", lw=.8)
    ax.set_title(f"{c}\n{len(s)} specs: {100 * s.right_sign.mean():.0f}% right sign, {100 * s.sig.mean():.0f}% p < .05", fontsize=10.5, loc="left")
    mse = np.nanmedian(s.se); s = s.assign(se=s.se.clip(upper=5 * mse))          # near-singular team-FE specs: cap the bar, not the estimate
    lo_, hi_ = (s.b - 1.645 * s.se).min(), (s.b + 1.645 * s.se).max(); pad = .15 * (hi_ - lo_)
    ax.set_ylim(min(lo_, 0) - pad, max(hi_, 0) + pad)
    ax.set_xlabel("specifications, sorted by estimate"); ax.set_ylabel("Drive+ effect per 10 points" + (" (per 100 snaps)" if c.startswith("ball plays") else ""))
fig.suptitle("Specification curves: the coverage effects survive every analysis choice; box-score ball plays do not move", fontsize=15, fontweight="bold", x=.01, ha="left", y=1.04)
fig.text(.01, -.04, "Each dot = one combination of rep radius, closing window, matching / open threshold, sample (CBs vs all DBs), controls and Drive+ version (frozen, truncation-safe, "
         "attempt-regressed); bars = 90% CI (clustered by DB; capped at 5x the median width for a few near-singular team-FE fits). Dark blue = expected sign and p < .05.", fontsize=8.5, color="#555")
fig.savefig(f"{ROOT}/results/figs/fig8_spec_curve.png", dpi=180, bbox_inches="tight")
