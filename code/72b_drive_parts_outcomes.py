"""Which phase of the drill drive carries the NFL effect? + time-matched translation.

Components from 72a (within-year z, then standardized over the CBs in the outcome reps). Outcomes:
  zone-matching reps (70's cell, primary): db_closing, sep_gain_break, sep_at_throw
  all CB reps: sep_gain_break;  targeted: cpoe
(1) Each component alone, per SD, player-permutation p (300).
(2) Horse race inside one model: v_out (out of the turn) + step2_gain (first two steps) + late_gain (after) -- these
    three add up to top, so the coefficients say which piece of the top speed matters.
(3) Time-matched translation: CB speed in NFL reps at 0.0 .. 1.0 s after the RECEIVER's break and closing rate per window
    (0-0.3, 0.3-0.6, 0.6-1.0 s), player means (CBs >= 10 reps, all CB reps, context-residualized), correlated with the
    drill's speed at 0.0 .. 1.0 s after HIS turn. Diagonal = the drill's time course maps onto the game's.
(4) Which NFL window drives the outcome (rep level, mechanical): closing per window -> sep_at_throw, completion.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(61)
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
COMP = ["entry", "turn_min", "v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "step2_gain", "late_gain", "top", "t_top", "d_05", "d_10", "d0_15", "straight"]
for c in COMP:
    v = Z[c].reindex(cov.nfl_id_d.unique()); cov["c_" + c] = cov.nfl_id_d.map((Z[c] - v.mean()) / v.std())
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 240)


def fit(d, terms, y, ctx, n_perm=300, perm_term=None):
    d = d.dropna(subset=[y] + terms + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    f = f"{y} ~ {' + '.join(terms)} + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    out = {}
    for term in ([perm_term] if perm_term else terms):
        t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[term]; null = []
        for _ in range(n_perm):
            dd = d.copy(); dd[term] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index))
            null.append(smf.ols(f, data=dd).fit().tvalues[term])
        out[term] = (m.params[term], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1))
    return out, len(d), d.nfl_id_d.nunique()


TASKS = [("zone-matching", "db_closing", "wr_skill + C(route_ran)"), ("zone-matching", "sep_gain_break", "wr_skill + C(route_ran)"),
         ("zone-matching", "sep_at_throw", "wr_skill + C(route_ran)"), ("all", "sep_gain_break", "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
         ("targeted", "cpoe", "rec_target_prop + C(team_coverage_man_zone)")]
sub = lambda u: cov[cov.cell == u] if u == "zone-matching" else (cov[cov.targeted == 1] if u == "targeted" else cov)
print("=== (1) each drill phase alone (b per SD | player-permutation p) ===")
rows = []
for u, y, ctx in TASKS:
    for c in COMP:
        o, n, k = fit(sub(u), ["c_" + c], y, ctx, n_perm=200)
        rows.append({"reps": u, "outcome": y, "component": c, "b": o["c_" + c][0], "p_perm": o["c_" + c][1], "n": n, "CBs": k})
R = pd.DataFrame(rows); R["task"] = R.reps + ":" + R.outcome
R.to_csv(f"{ROOT}/results/72b_components.csv", index=False)
print(R.pivot_table(index="component", columns="task", values="b", sort=False).round(3).to_string())
print("perm p:\n" + R.pivot_table(index="component", columns="task", values="p_perm", sort=False).round(3).to_string())

print("\n=== (2) horse race: out-of-turn speed vs first two steps vs after (sum = top) ===")
for u, y, ctx in TASKS:
    o, n, k = fit(sub(u), ["c_v_out", "c_step2_gain", "c_late_gain"], y, ctx, n_perm=200)
    print(f"  {u:13s} {y:14s} " + " | ".join(f"{t[2:]} {b:+.3f} (p {p:.3f})" for t, (b, p) in o.items()) + f"  [reps {n}, CBs {k}]")
    o, n, k = fit(sub(u), ["c_v_04", "c_v_10"], y, ctx, n_perm=200)
    print(f"  {'':13s} {'':14s} speed after ~2 steps vs at 1 s: " + " | ".join(f"{t[2:]} {b:+.3f} (p {p:.3f})" for t, (b, p) in o.items()))

# ---------------- (3) time-matched translation
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
get = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = cov.idx_r.values, cov.idx_d.values
sr, lr, pr = get(A, "s"), get(A, "a_lat"), get(A, "post_pass")
xr, yr_, xd, yd, sd = get(A, "x_rel"), get(A, "y_field"), get(B, "x_rel"), get(B, "y_field"), get(B, "s")
vv = V[A] & V[B]; T = X.shape[1]; k_ = np.arange(T)[None, :] - PRE; r_ = np.arange(len(cov))
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k_ >= 5) & (k_ <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1); hasb = a.max(1) > 0
dist = np.hypot(xr - xd, yr_ - yd)
at = lambda arr, j: np.where(hasb & (j < T) & vv[r_, np.clip(j, 0, T - 1)], arr[r_, np.clip(j, 0, T - 1)], np.nan)
for i in range(0, 11, 2):
    cov[f"g_v{i:02d}"] = at(sd, jb + i)
for lo, hi in ((0, 3), (3, 6), (6, 10)):
    cov[f"g_close_{lo}_{hi}"] = (at(dist, jb + lo) - at(dist, jb + hi)) / ((hi - lo) / 10)
GV = [f"g_v{i:02d}" for i in range(0, 11, 2)] + ["g_close_0_3", "g_close_3_6", "g_close_6_10"]
for g in GV:
    dd = cov.dropna(subset=[g, "wr_skill"])
    cov.loc[dd.index, g + "_r"] = smf.ols(f"{g} ~ wr_skill + C(route_ran) + C(team_coverage_man_zone) + cushion", data=dd.dropna(subset=["cushion"])).fit().resid
n = cov.groupby("nfl_id_d").size()
PM = cov.groupby("nfl_id_d")[[g + "_r" for g in GV]].mean()[n >= 10]
PM.columns = GV
J = PM.join(Z[["v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "top", "d_10"]], how="inner")
print(f"\n=== (3) time-matched translation: drill speed after HIS turn (rows) vs NFL after the receiver's break (cols); r over {len(J)} CBs ===")
print(J.corr().loc[["v_out", "v_02", "v_04", "v_06", "v_08", "v_10", "top", "d_10"], GV].round(2).to_string())
J.to_csv(f"{ROOT}/results/72b_translation_player.csv")

print("\n=== (4) which NFL window matters for the outcome (rep level, per SD of the window's closing rate) ===")
for y, d in [("sep_at_throw", cov), ("completion", cov[cov.targeted == 1])]:
    dd = d.dropna(subset=["g_close_0_3", "g_close_3_6", "g_close_6_10", y, "wr_skill"]).copy()
    for c in ["g_close_0_3", "g_close_3_6", "g_close_6_10"]:
        dd[c] = (dd[c] - dd[c].mean()) / dd[c].std()
    m = (smf.logit if y == "completion" else smf.ols)(f"{y} ~ g_close_0_3 + g_close_3_6 + g_close_6_10 + wr_skill + C(team_coverage_man_zone)", data=dd)
    m = m.fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.nfl_id_d)[0]}) if y == "completion" else m.fit(cov_type="cluster", cov_kwds={"groups": G2(dd)})
    print(f"  {y:13s} " + " | ".join(f"{c} {m.params[c]:+.3f} (p {m.pvalues[c]:.3f})" for c in ["g_close_0_3", "g_close_3_6", "g_close_6_10"]) + f"  n {len(dd)}" + ("  [logit coef]" if y == "completion" else ""))
cov[["game_id", "play_id", "nfl_id_d"] + GV].to_parquet(f"{ROOT}/data/outcomes/cb_reps_time_windows.parquet", index=False)
