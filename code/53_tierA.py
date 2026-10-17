"""Phase 3, tier A: pre-registered matched hypotheses (drill mechanic -> the same mechanic's NFL payoff).

The list below is written to results/53_tierA_prereg.md (with a timestamp) BEFORE any test is computed.
Player-level test per hypothesis: Pearson r in the expected direction (one-sided permutation p, 5000 perms), partial r
given weight, leave-one-draft-class-out OOS r (fit a 1-feature line on two classes, predict the third, pool), Holm across
the player-level family. Rep-level tests (coverage / tackle reps): OLS with two-way clustered SE + player-permutation p.
"""
import datetime, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
H = [  # (id, unit, feature, outcome, expected sign, rationale)
    ("A1", "WR", "fam:routes:accel_0_1s", "REC sep_oe vs man", +1, "acceleration into the route beats man coverage"),
    ("A2", "WR", "fam:routes:brk1_entry_speed", "REC sep_oe in-breaking", +1, "speed carried into the break"),
    ("A3", "WR", "drill:WR:SPEED_OUT_ROUTE_LEFT:turn_speed", "REC sep_oe out-breaking", +1, "speed through the out-cut"),
    ("A4", "WR", "drill:THREE_CONE_DRILL:brk1_exit_burst", "COV created sep_gain_break", +1, "burst out of a plant opens the receiver at his break"),
    ("A5", "WR", "fam:routes:catch_carry", "REC YAC over NGS expected", +1, "carrying speed through the catch"),
    ("A6", "WR", "drill:FORTY_YARD_DASH:top_speed", "REC sep_oe vertical", +1, "top speed on go routes"),
    ("A7", "WR", "drill:WR:CURL_ROUTE_RIGHT:turn_lean", "REC sep_oe stop", +1, "sharp curl turn on stop routes"),
    ("B1", "CB", "fam:backpedal:speed_at_1.5s", "COV allowed (man) sep_mean", -1, "faster backpedal stays on top in man"),
    ("B2", "CB", "drill:DB:W_DRILL:brk_mean_exit_burst", "COV allowed sep_gain_break", -1, "burst out of breaks limits what the receiver gains at his break"),
    ("B3", "CB", "drill:FORTY_YARD_DASH:top_speed", "COV allowed db_speed_deficit", -1, "top speed closes the speed gap after the break"),
    ("B4", "CB", "fam:all_db:speed_at_0.5s", "TACKLE closing_speed", +1, "first-step burst closes on the ball carrier"),
    ("B5", "CB", "drill:DB:W_DRILL:brk_mean_brake_time", "TACKLE yds_after_arrival", +1, "slow braking -> overrun -> more yards after arrival"),
    ("C1", "S", "fam:all_db:speed_at_0.5s", "TACKLE closing_speed", +1, "first-step burst closes on the ball carrier"),
    ("C2", "S", "drill:FORTY_YARD_DASH:top_speed", "S run tackles_oe", +1, "range in run support"),
    ("D1", "EDGE", "frozen:hoop_turn_speed", "EDGE penetration_oe", +1, "bend at speed (pre-registered bend metric)"),
    ("D2", "EDGE", "drill:DL:PASS_RUSH_DRILL:turn_speed", "H2H corner rate", +1, "turn speed around the bag -> turning the tackle's corner"),
    ("D3", "EDGE", "fam:rush:accel_sustained_peak", "EDGE get_off_oe (lower=better)", -1, "sustained acceleration -> faster get-off"),
    ("D4", "EDGE", "drill:FORTY_YARD_DASH:accel_sustained_peak", "EDGE depth2s_oe", +1, "acceleration -> depth at 2 s"),
    ("E1", "IDL", "frozen:hoop_turn_speed", "IDL penetration_oe", +1, "bend at speed"),
    ("E2", "IDL", "drill:THREE_CONE_DRILL:accel_0_1s", "IDL get_off_oe (lower=better)", -1, "short-area acceleration -> get-off"),
    ("E3", "IDL", "fam:all_dl:turn_speed", "H2H blocker ppa_oe vs him", +1, "turn speed -> wins vs guards / centers"),
    ("F1", "T", "drill:OL:PASS_PRO_MIRROR_DRILL:brk_mean_entry_speed", "T pressure prob allowed_oe (lower=better)", -1, "mirror speed -> fewer pressures"),
    ("F2", "T", "drill:OL:FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE:turn_lean", "H2H corner allowed rate", -1, "redirect in the wave drill -> fewer corners turned"),
    ("G1", "IOL", "drill:OL:PASS_PRO_MIRROR_DRILL:brk_mean_entry_speed", "IOL pressure prob allowed_oe (lower=better)", -1, "mirror speed -> fewer pressures"),
    ("G2", "IOL", "fam:lateral:top_speed", "H2H ppa_oe", -1, "lateral speed -> lower pressure probability in head-to-head reps"),
]
REP = [  # rep-level
    ("R1", "coverage", "sep_gain_break", "WR drill:THREE_CONE_DRILL:brk1_exit_burst (+) x DB drill:DB:W_DRILL:brk_mean_exit_burst (-); interaction < 0 (receiver burst pays most vs low-burst DBs)"),
    ("R2", "coverage (man)", "sep_mean", "WR fam:routes:accel_0_1s (+), DB fam:backpedal:speed_at_1.5s (-)"),
    ("R3", "tackle", "yds_after_arrival", "tackler fam:all_db:speed_at_0.5s (-) [DB tacklers]"),
]
with open(f"{ROOT}/results/53_tierA_prereg.md", "w") as fh:
    fh.write(f"# Tier A pre-registration\nwritten {datetime.datetime.now().isoformat(timespec='seconds')} before computing any tier A statistic\n\n")
    fh.write("| id | unit | combine feature | NFL outcome | expected | rationale |\n|---|---|---|---|---|---|\n")
    for h in H:
        fh.write(f"| {h[0]} | {h[1]} | `{h[2]}` | {h[3]} | {'+' if h[4] > 0 else '-'} | {h[5]} |\n")
    fh.write("\nRep-level:\n" + "".join(f"- {r[0]} ({r[1]}, outcome {r[2]}): {r[3]}\n" for r in REP))
print(open(f"{ROOT}/results/53_tierA_prereg.md").read())

base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
rng = np.random.default_rng(0)


def loco_r(x, y, cls):
    pred = pd.Series(np.nan, index=x.index)
    for c in np.unique(cls):
        tr, te = cls != c, cls == c
        if tr.sum() < 8 or te.sum() < 3:
            continue
        b = np.polyfit(x[tr], y[tr], 1)
        pred[te] = np.polyval(b, x[te])
    ok = pred.notna()
    return np.corrcoef(pred[ok], y[ok])[0, 1] if ok.sum() >= 10 else np.nan


rows = []
for hid, unit, feat, outc, sign, why in H:
    F = pd.read_parquet(f"{ROOT}/data/features/catalog_{unit}.parquet")
    O = pd.read_parquet(f"{ROOT}/data/outcomes/outcomes_{unit}.parquet")
    if feat not in F or outc not in O:
        rows.append({"id": hid, "unit": unit, "feature": feat, "outcome": outc, "note": "missing"}); continue
    d = pd.concat([F[feat].rename("x"), O[outc].rename("y"), base.combine_weight.rename("w"), base.draft_year.rename("cls")], axis=1).dropna(subset=["x", "y"])
    x, y = d.x.values, d.y.values
    r = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(5000)])
    p1 = (1 + (sign * null >= sign * r).sum()) / 5001
    dw = d.dropna(subset=["w"])
    Z = np.column_stack([np.ones(len(dw)), dw.w])
    rx = dw.x - Z @ np.linalg.lstsq(Z, dw.x, rcond=None)[0]; ry = dw.y - Z @ np.linalg.lstsq(Z, dw.y, rcond=None)[0]
    rows.append({"id": hid, "unit": unit, "feature": feat, "outcome": outc, "expected": "+" if sign > 0 else "-", "n": len(d),
                 "r": r, "p_one_sided": p1, "r_given_weight": np.corrcoef(rx, ry)[0, 1], "loco_oos_r": loco_r(d.x, d.y, d.cls.values),
                 "direction_ok": np.sign(r) == sign})
R = pd.DataFrame(rows)
ok = R.p_one_sided.notna()
o = np.argsort(R.loc[ok, "p_one_sided"].values); n = ok.sum()
adj = np.empty(n); adj[o] = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * R.loc[ok, "p_one_sided"].values[o]))
R.loc[ok, "p_holm"] = adj
R.to_csv(f"{ROOT}/results/53_tierA_player.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(R.round(3).to_string(index=False))

# ---------------- rep-level
print("\n=== rep-level ===")
def trait(unit, feat):
    return pd.read_parquet(f"{ROOT}/data/features/catalog_{unit}.parquet")[feat]
wr_burst = trait("WR", "drill:THREE_CONE_DRILL:brk1_exit_burst"); wr_acc = trait("WR", "fam:routes:accel_0_1s")
db_burst = pd.concat([trait("CB", "drill:DB:W_DRILL:brk_mean_exit_burst"), trait("S", "drill:DB:W_DRILL:brk_mean_exit_burst")]).groupby(level=0).first()
db_bp = pd.concat([trait("CB", "fam:backpedal:speed_at_1.5s"), trait("S", "fam:backpedal:speed_at_1.5s")]).groupby(level=0).first()
db_first = pd.concat([trait("CB", "fam:all_db:speed_at_0.5s"), trait("S", "fam:all_db:speed_at_0.5s")]).groupby(level=0).first()
cov = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
cov = cov[cov.season_type.isin(["REG", "POST"]) & (cov.nullified != "Y")].copy()
zs = lambda s_, ids: (s_ - s_.reindex(pd.unique(ids)).mean()) / s_.reindex(pd.unique(ids)).std()
cov["z_wr_burst"] = cov.nfl_id_r.map(zs(wr_burst, cov.nfl_id_r)); cov["z_db_burst"] = cov.nfl_id_d.map(zs(db_burst, cov.nfl_id_d))
cov["z_wr_acc"] = cov.nfl_id_r.map(zs(wr_acc, cov.nfl_id_r)); cov["z_db_bp"] = cov.nfl_id_d.map(zs(db_bp, cov.nfl_id_d))


def rep_test(d, f, terms, label, n_perm=500):
    d = d.dropna(subset=[c for c in d.columns if c in f])
    g = np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]]) if "nfl_id_r" in d else pd.factorize(d.nfl_id_tk)[0]
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
    out = {}
    for t in terms:
        null = []
        for _ in range(n_perm):
            dd = d.copy()
            for col, idc in [("z_wr_burst", "nfl_id_r"), ("z_wr_acc", "nfl_id_r"), ("z_db_burst", "nfl_id_d"), ("z_db_bp", "nfl_id_d"), ("z_tk", "nfl_id_tk")]:
                if col in dd and idc in dd:
                    vals = dd.drop_duplicates(idc).set_index(idc)[col]
                    dd[col] = dd[idc].map(pd.Series(rng.permutation(vals.values), index=vals.index))
            null.append(smf.ols(f, data=dd).fit().tvalues[t])
        out[t] = (m.params[t], m.pvalues[t], (1 + (np.abs(null) >= abs(m.tvalues[t])).sum()) / (n_perm + 1))
    print(f"{label}: reps {len(d)} | " + " | ".join(f"{t} {b:+.4f} (clustered p {p:.3f}, perm p {pp:.3f})" for t, (b, p, pp) in out.items()))


rep_test(cov, "sep_gain_break ~ z_wr_burst * z_db_burst + C(team_coverage_man_zone) + C(route_ran)", ["z_wr_burst", "z_db_burst", "z_wr_burst:z_db_burst"], "R1 coverage sep gained at break")
rep_test(cov[cov.team_coverage_man_zone == "MAN_COVERAGE"], "sep_mean ~ z_wr_acc + z_db_bp + C(route_ran)", ["z_wr_acc", "z_db_bp"], "R2 man coverage mean separation")
tk = pd.read_parquet(f"{ROOT}/data/outcomes/tackle_reps.parquet")
tk = tk[tk.season_type.isin(["REG", "POST"]) & tk.nfl_position.isin(["CB", "SS", "FS", "DB"])].copy()
tk["z_tk"] = tk.nfl_id_tk.map(zs(db_first, tk.nfl_id_tk))
rep_test(tk, "yds_after_arrival ~ z_tk + dist_at_catch", ["z_tk"], "R3 tackle yards after arrival (DB tacklers)")
