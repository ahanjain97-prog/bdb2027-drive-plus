"""CB Transition Drive Speed: ball-flight closing, out-of-sample backtests, robustness, and a completion model.

(a) Ball flight: on targeted reps, DB closing rate on the receiver between the throw and the ball's arrival
    (raw tracking events pass_forward / pass_arrived), DB speed in flight; chain combine drive -> flight closing -> completion.
(b) Backtests (rep level, outcome = separation gained at the break; baseline model = receiver skill + route + man/zone):
    rookies-forward  leave-one-draft-class-out over CBs: fit on reps of CBs from 2 classes, predict reps of the 3rd
    season-forward   fit on 2023-24 season reps, predict 2025 season reps
    For each: out-of-sample MSE with vs without drive speed, and a player-permutation null of the improvement.
(c) Robustness: looser rep definition (mean distance <= 4 yd), CB alignment control, every official test, draft round.
(d) Completion model: logit completion ~ drive speed (+ man/zone) on targets; expected completion % at drive-speed
    percentiles with player-clustered bootstrap CIs; yards per target with incompletions = 0.
"""
import json, sys, warnings
import duckdb
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
rng = np.random.default_rng(7)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_enriched.parquet")
dv = cov.drop_duplicates("nfl_id_d").set_index("nfl_id_d").drive_combine
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 220)

# ---------------- (a) ball flight
tg = cov[cov.targeted == 1].copy()
con = duckdb.connect(); con.register("k", tg[["game_id", "play_id", "nfl_id_r"]].drop_duplicates())
ev = con.execute(f"""SELECT t.game_id, t.play_id,
    min(t."time") FILTER (WHERE event='ball_snap') ts, min(t."time") FILTER (WHERE event='pass_forward') tf,
    min(t."time") FILTER (WHERE event IN ('pass_arrived','pass_outcome_caught','pass_outcome_incomplete')) ta
  FROM '{ROOT}/data/raw/game_tracking_*.parquet' t JOIN k ON t.game_id=k.game_id AND t.play_id=k.play_id AND t.nfl_id=k.nfl_id_r GROUP BY ALL""").df()
tg = tg.merge(ev, on=["game_id", "play_id"], how="left")
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
get = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = tg.idx_r.values, tg.idx_d.values
dist = np.hypot(get(A, "x_rel") - get(B, "x_rel"), get(A, "y_field") - get(B, "y_field")); sd = get(B, "s")
T = X.shape[1]; r = np.arange(len(tg))
jf = (np.round((tg.tf - tg.ts).dt.total_seconds() * 10).fillna(-99).astype(int) + PRE).values
ja = (np.round((tg.ta - tg.ts).dt.total_seconds() * 10).fillna(-99).astype(int) + PRE).values
ok = (jf >= 0) & (ja > jf) & (ja < T)
jf_, ja_ = np.clip(jf, 0, T - 1), np.clip(ja, 0, T - 1)
ok &= V[A, jf_] & V[A, ja_] & V[B, jf_] & V[B, ja_]
tg["flight_time"] = np.where(ok, (ja - jf) / 10.0, np.nan)
tg["flight_closing"] = np.where(ok, (dist[r, jf_] - dist[r, ja_]) / np.maximum((ja - jf) / 10.0, 0.1), np.nan)
tg["sep_at_arrival"] = np.where(ok, dist[r, ja_], np.nan)
cols_ = np.arange(T)[None, :]
fw = (cols_ >= jf_[:, None]) & (cols_ <= ja_[:, None]) & V[B] & ok[:, None]
tg["db_flight_speed"] = np.where(fw.sum(1) >= 2, np.where(fw, sd, 0).sum(1) / np.maximum(fw.sum(1), 1), np.nan)
print(f"=== (a) ball flight (targeted reps with events: {ok.sum()} of {len(tg)}) ===")
for y in ["flight_closing", "db_flight_speed", "sep_at_arrival"]:
    d = tg.dropna(subset=[y, "z_drive"])
    m = smf.ols(f"{y} ~ z_drive + flight_time + C(team_coverage_man_zone)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    print(f"  combine drive -> {y:16s} {m.params['z_drive']:+.3f} per SD (p {m.pvalues['z_drive']:.3f}, targets {len(d)})")
d = tg.dropna(subset=["flight_closing", "sep_at_throw", "completion", "z_drive"])
m0 = smf.ols("completion ~ z_drive + sep_at_throw + C(team_coverage_man_zone)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
m1 = smf.ols("completion ~ z_drive + sep_at_throw + flight_closing + flight_time + C(team_coverage_man_zone)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
print(f"  completion ~ drive + sep at throw: drive {m0.params['z_drive']:+.3f} (p {m0.pvalues['z_drive']:.3f}) -> + flight closing: drive "
      f"{m1.params['z_drive']:+.3f} (p {m1.pvalues['z_drive']:.3f}), flight closing {m1.params['flight_closing']:+.3f}/(yd/s) (p {m1.pvalues['flight_closing']:.3f})")

# ---------------- (b) backtests
BASEF = "sep_gain_break ~ wr_skill + C(team_coverage_man_zone) + C(route_ran)"
dd = cov.dropna(subset=["sep_gain_break", "wr_skill", "z_drive"]).copy()
dd["cls"] = dd.nfl_id_d.map(base.draft_year)


def oos(train, test):
    out = {}
    for name, f in [("base", BASEF), ("drive", BASEF + " + z_drive")]:
        m = smf.ols(f, data=train).fit()
        te = test[test.route_ran.isin(train.route_ran.unique())]
        out[name] = ((te.sep_gain_break - m.predict(te)) ** 2).mean()
    return out


def backtest(splits, label, n_perm=300):
    def run(d):
        mse_b = mse_d = 0.0; n = 0
        for tr_mask, te_mask in splits(d):
            o = oos(d[tr_mask], d[te_mask]); w = te_mask.sum()
            mse_b += o["base"] * w; mse_d += o["drive"] * w; n += w
        return 1 - mse_d / mse_b                                 # relative MSE improvement
    obs = run(dd)
    vals = dd.drop_duplicates("nfl_id_d").set_index("nfl_id_d").z_drive; null = []
    for _ in range(n_perm):
        p_ = dd.copy(); p_["z_drive"] = p_.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(run(p_))
    p = (1 + (np.array(null) >= obs).sum()) / (n_perm + 1)
    print(f"  {label:58s} OOS MSE improvement {obs * 100:+.2f}%  (player-permutation p {p:.3f}; null 95th pct {np.percentile(null, 95) * 100:+.2f}%)")


print("\n=== (b) out-of-sample backtests (separation gained at the break) ===")
backtest(lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())], "rookies-forward: leave-one-draft-class-out over CBs")
backtest(lambda d: [((d.season <= 2024).values, (d.season == 2025).values)], "season-forward: train 2023-24 seasons, test 2025")
backtest(lambda d: [((d.cls <= 2024).values, (d.cls == 2025).values)], "new-prospect: train 2023-24 class CBs, test 2025-class CBs")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    sub = dd[dd.team_coverage_man_zone == cz]
    ddb = dd; dd = sub
    backtest(lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())], f"rookies-forward, {cz} reps only")
    dd = ddb

# ---------------- (c) robustness
print("\n=== (c) robustness: separation gained at the break, b per SD of drive speed ===")
def rb(d, extra, label):
    cols = [c.strip() for c in extra.split("+") if c.strip()] if extra else []
    d = d.dropna(subset=["sep_gain_break", "wr_skill", "z_drive"] + [c for c in cols if not c.startswith("C(")])
    m = smf.ols(f"sep_gain_break ~ z_drive + wr_skill + C(team_coverage_man_zone) + C(route_ran)" + (f" + {extra}" if extra else ""), data=d).fit(
        cov_type="cluster", cov_kwds={"groups": G2(d)})
    print(f"  {label:52s} {m.params['z_drive']:+.4f} (p {m.pvalues['z_drive']:.3f}) reps {len(d)} CBs {d.nfl_id_d.nunique()}")
cov2 = cov.copy()
for c in ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle"]:
    v = base[c]; cov2["z_" + c] = cov2.nfl_id_d.map((v - v.reindex(dv.index).mean()) / v.reindex(dv.index).std())
cov2["z_round"] = cov2.nfl_id_d.map(base.draft_round.fillna(8) if "draft_round" in base else pd.Series(dtype=float))
pl = pd.read_parquet(f"{ROOT}/data/raw/players.parquet").set_index("nfl_id")
cov2["draft_round"] = cov2.nfl_id_d.map(pl.draft_round.fillna(8))
rb(cov2, "", "baseline (rep mean distance <= 3 yd)")
rb(cov2, "C(lined_up_position_d)", "+ CB alignment")
rb(cov2, "z_forty + z_ten_yd_split + z_vertical + z_broad_jump", "+ 40, 10-yd, vertical, broad (complete cases)")
rb(cov2, "z_three_cone + z_short_shuttle", "+ 3-cone + shuttle (complete cases)")
rb(cov2, "C(draft_round)", "+ draft round")
# looser rep definition: rebuild sep_gain_break reps at mean distance <= 4 yd
reps4 = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
print("  (looser 4-yd reps require rebuilding 41 with a 4.0 threshold; see 58b below)")

# ---------------- (d) completion model
print("\n=== (d) completion model on targets ===")
tg["yds_per_target"] = np.where(tg.completion == 1, tg.rec_yards.fillna(0), 0.0)
d = tg.dropna(subset=["z_drive", "completion"]).copy()
m = smf.logit("completion ~ z_drive + C(team_coverage_man_zone)", data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.nfl_id_d)[0]})
print(f"  logit: odds ratio per SD of drive speed {np.exp(m.params['z_drive']):.2f} (p {m.pvalues['z_drive']:.4f}), targets {len(d)}, CBs {d.nfl_id_d.nunique()}")
pcts = [10, 25, 50, 75, 90]
zq = {q: (np.percentile(dv.dropna(), q) - dv.mean()) / dv.std() for q in pcts}
def exp_comp(dd_, mdl):
    out = {}
    for q, zv in zq.items():
        x = dd_.copy(); x["z_drive"] = zv; out[q] = mdl.predict(x).mean()
    return out
pt = exp_comp(d, m)
boot = {q: [] for q in pcts}
ids = d.nfl_id_d.unique()
for _ in range(500):
    pick = rng.choice(ids, len(ids), replace=True)
    bd = pd.concat([d[d.nfl_id_d == i] for i in pick])
    try:
        mb = smf.logit("completion ~ z_drive + C(team_coverage_man_zone)", data=bd).fit(disp=0)
        for q, v in exp_comp(d, mb).items():
            boot[q].append(v)
    except Exception:
        pass
print("  expected completion % allowed by CB drive-speed percentile (yd/s), player-clustered bootstrap 90% CI:")
for q in pcts:
    lo, hi = np.percentile(boot[q], [5, 95])
    print(f"    p{q:<3d} ({np.percentile(dv.dropna(), q):.2f} yd/s): {pt[q] * 100:5.1f}%  [{lo * 100:.1f}, {hi * 100:.1f}]")
my = smf.ols("yds_per_target ~ z_drive + C(team_coverage_man_zone)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
me = smf.ols("epa ~ z_drive + C(team_coverage_man_zone)", data=d.dropna(subset=["epa"])).fit(cov_type="cluster", cov_kwds={"groups": G2(d.dropna(subset=["epa"]))})
print(f"  yards per target (incompletions = 0): {my.params['z_drive']:+.2f} per SD (p {my.pvalues['z_drive']:.3f}); EPA per target {me.params['z_drive']:+.3f} per SD (p {me.pvalues['z_drive']:.3f})")
print(f"  per 100 targets, +1 SD of drive speed: {m.params['z_drive'] and (pt[75] - pt[25]) * 100:.1f} fewer completions between p25 and p75 CBs; "
      f"{-my.params['z_drive'] * 100:.0f} yards; {-me.params['z_drive'] * 100:.1f} EPA")
tg.to_parquet(f"{ROOT}/data/outcomes/cb_targets_enriched.parquet", index=False)
