"""More checks and hypotheses around CB Transition Drive Speed.

(1) Completion confound: do fast-drive CBs simply face deeper / harder throws? completion ~ drive + air yards (pass_length)
    + route family + man/zone (+ separation at the throw).
(2) Looser rep definition: rebuild coverage reps with mean receiver-DB distance <= 4 yd (vs 3), refit the main model.
(3) Out-of-position replication (specified before looking): safeties (SS/FS) in coverage reps, same frozen drill metric.
(4) Man-coverage screen: every reliable DB catalog feature + the transition-drill phase metrics vs man-coverage reps
    (separation gained at the break, mean separation, separation at the throw); two-way clustered + player-permutation p,
    BH across the screen; any survivor gets the rookies-forward backtest.
"""
import json, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
rng = np.random.default_rng(11)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
drive = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 220)

# ---------------- (1) completion confound
tg = pd.read_parquet(f"{ROOT}/data/outcomes/cb_targets_enriched.parquet")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
tg["air_yards"] = pp.pass_length.reindex(pd.MultiIndex.from_arrays([tg.game_id, tg.play_id, tg.nfl_id_r])).values
print("=== (1) completion when targeted: is it just deeper throws? ===")
print(f"  r(drive speed, air yards of targets against him) = {tg[['drive_combine', 'air_yards']].dropna().corr().iloc[0, 1]:+.2f}")
for f, lab in [("completion ~ z_drive + C(team_coverage_man_zone)", "drive only"),
               ("completion ~ z_drive + air_yards + C(team_coverage_man_zone)", "+ air yards"),
               ("completion ~ z_drive + air_yards + C(fam) + C(team_coverage_man_zone)", "+ air yards + route family"),
               ("completion ~ z_drive + air_yards + C(fam) + sep_at_throw + C(team_coverage_man_zone)", "+ air yards + route + sep at throw")]:
    d = tg.dropna(subset=["z_drive", "air_yards", "fam", "sep_at_throw", "completion"])
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    print(f"  {lab:36s} drive {m.params['z_drive']:+.3f} per SD (p {m.pvalues['z_drive']:.3f}), targets {len(d)}")

# ---------------- (2) looser rep definition (4 yd)
print("\n=== (2) looser coverage-rep definition (mean distance <= 4 yd) ===")
src = open(f"{ROOT}/code/41_coverage_reps.py").read()
src = src.replace("keep = (n_live >= 10) & (sep_mean <= 3.0)", "keep = (n_live >= 10) & (sep_mean <= 4.0)")
src = src.replace('P.to_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet", index=False)', 'P.to_parquet(f"{ROOT}/data/outcomes/coverage_reps_4yd.parquet", index=False)')
src = src.split("print(f\"coverage reps")[0]
exec(src)
c4 = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps_4yd.parquet")
c4 = c4[c4.season_type.isin(["REG", "POST"]) & (c4.nullified != "Y")].copy()


def prep(c, positions):
    c = c[c.nfl_id_d.isin(base.index[base.nfl_position.isin(positions)])].copy()
    tot = c.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
    c["wr_skill"] = (c.nfl_id_r.map(tot["sum"]) - c.sep_gain_break.fillna(0)) / (c.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)
    v = drive.reindex(c.nfl_id_d.unique())
    c["z_drive"] = c.nfl_id_d.map((drive - v.mean()) / v.std())
    return c


def main_fit(c, label, extra=""):
    d = c.dropna(subset=["sep_gain_break", "wr_skill", "z_drive"])
    out = []
    for y in ["sep_gain_break", "sep_at_throw"]:
        dd = d.dropna(subset=[y])
        m = smf.ols(f"{y} ~ z_drive + wr_skill + C(team_coverage_man_zone) + C(route_ran){extra}", data=dd).fit(cov_type="cluster", cov_kwds={"groups": G2(dd)})
        out.append(f"{y} {m.params['z_drive']:+.4f} (p {m.pvalues['z_drive']:.3f})")
    print(f"  {label:44s} reps {len(d)}, DBs {d.nfl_id_d.nunique()} | " + " | ".join(out))


c4cb = prep(c4, ["CB"])
main_fit(c4cb, "CBs, 4-yd reps")
main_fit(c4cb, "CBs, 4-yd reps + CB alignment", " + C(lined_up_position_d)")

# ---------------- (3) safeties
print("\n=== (3) out-of-position replication: safeties (same drill metric) ===")
c3 = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
c3 = c3[c3.season_type.isin(["REG", "POST"]) & (c3.nullified != "Y")]
for lab, cdat in [("S, 3-yd reps", prep(c3, ["SS", "FS", "DB"])), ("S, 4-yd reps", prep(c4, ["SS", "FS", "DB"]))]:
    main_fit(cdat, lab)
    for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
        main_fit(cdat[cdat.team_coverage_man_zone == cz], f"   {lab}, {cz}")

# ---------------- (4) man-coverage screen
print("\n=== (4) man-coverage screen: every reliable DB feature vs CB man-coverage reps ===")
cat = pd.read_parquet(f"{ROOT}/data/features/catalog_CB.parquet")
meta = pd.read_csv(f"{ROOT}/data/features/catalog_CB_meta.csv", index_col=0)
feats = [c for c in cat.columns if meta.loc[c, "kind"] in ("family_composite", "drill_feature")] + [c for c in cat.columns if c.startswith("stopwatch:")]
ph = pd.read_csv(f"{ROOT}/results/56_cb_transition_player.csv", index_col=0)
FT = cat[feats].join(ph.add_prefix("transition:"), how="outer")
FT["transition:drive_speed_frozen"] = drive
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_enriched.parquet")
man = cov[cov.team_coverage_man_zone == "MAN_COVERAGE"].copy()
rows = []
for f in FT.columns:
    v = FT[f].reindex(man.nfl_id_d.unique())
    if v.notna().sum() < 20 or v.std() == 0:
        continue
    man["zf"] = man.nfl_id_d.map((FT[f] - v.mean()) / v.std())
    for y in ["sep_gain_break", "sep_mean", "sep_at_throw"]:
        d = man.dropna(subset=["zf", y, "wr_skill"])
        fml = f"{y} ~ zf + wr_skill + C(route_ran)"
        m = smf.ols(fml, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
        t0 = smf.ols(fml, data=d).fit().tvalues["zf"]
        vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").zf; null = []
        for _ in range(200):
            dd = d.copy(); dd["zf"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(fml, data=dd).fit().tvalues["zf"])
        rows.append({"feature": f, "outcome": y, "reps": len(d), "CBs": d.nfl_id_d.nunique(), "b_per_SD": m.params["zf"],
                     "p_clustered": m.pvalues["zf"], "p_perm": (1 + (np.abs(null) >= abs(t0)).sum()) / 201})
S = pd.DataFrame(rows)
o = np.argsort(S.p_perm.values); n = len(S); q = np.empty(n)
q[o] = np.minimum.accumulate((S.p_perm.values[o] * n / np.arange(1, n + 1))[::-1])[::-1]; S["q_bh"] = np.minimum(q, 1)
S.to_csv(f"{ROOT}/results/59_man_screen.csv", index=False)
print(f"  tests {n}; perm p < .05: {(S.p_perm < .05).sum()} (expected {.05 * n:.1f}); BH q < .10: {(S.q_bh < .10).sum()}")
print(S.sort_values("p_perm").head(12).round(4).to_string(index=False))
