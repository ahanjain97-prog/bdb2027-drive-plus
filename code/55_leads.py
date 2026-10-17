"""Stress-test the leads from tier A / B (all post-hoc relative to the screen -> leads, not confirmed findings).

L1 CB  backpedal-45-transition top speed -> separation the receiver gains at his break (coverage reps, rep level:
       two-way clustered + player permutation; receiver's separation-creating skill in his OTHER reps as control)
L2 IDL tracked 40 sustained acceleration -> pass-rush pressure (play level logit, clustered by player)
L3 EDGE tracked 40 top speed -> blocker's peak pressure probability on head-to-head reps (blocker wave speed control)
L4 WR  curl-drill turn lean -> separation on stop (hitch) routes (play level, clustered by player)
For each: per draft class sign, and the same model with the matching OFFICIAL stopwatch result instead (tracking vs stopwatch).
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
rng = np.random.default_rng(1)
cat = lambda u: pd.read_parquet(f"{ROOT}/data/features/catalog_{u}.parquet")
def z(s_, ids):
    v = s_.reindex(pd.unique(ids)); return (s_ - v.mean()) / v.std()


def perm_t(d, f, term, id_col, col, n=500):
    t0 = smf.ols(f, data=d).fit().tvalues[term]
    vals = d.drop_duplicates(id_col).set_index(id_col)[col]
    null = []
    for _ in range(n):
        dd = d.copy(); dd[col] = dd[id_col].map(pd.Series(rng.permutation(vals.values), index=vals.index))
        null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return (1 + (np.abs(null) >= abs(t0)).sum()) / (n + 1)


def report(label, d, f, term, groups, id_col, col, cls_col="cls"):
    d = d.dropna(subset=[col])
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": groups(d)})
    p_perm = perm_t(d, f, term, id_col, col)
    by = {}
    for c, g in d.groupby(cls_col):
        try:
            mm = smf.ols(f, data=g).fit(); by[int(c)] = f"{mm.params[term]:+.3f}"
        except Exception:
            by[int(c)] = "n/a"
    print(f"{label}: b {m.params[term]:+.4f} (clustered p {m.pvalues[term]:.3f}, player-perm p {p_perm:.3f}) reps {len(d)} players {d[id_col].nunique()} | by class {by}")


# ---------------- L1 CB backpedal transition top speed -> separation gained at the break
cov = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
cov = cov[cov.season_type.isin(["REG", "POST"]) & (cov.nullified != "Y")].copy()
cb = pd.concat([cat("CB"), cat("S")])
cb = cb[~cb.index.duplicated()]
cov = cov[cov.nfl_id_d.isin(base.index[base.nfl_position == "CB"])].copy()
tot = cov.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
cov["wr_skill_loo"] = (cov.nfl_id_r.map(tot["sum"]) - cov.sep_gain_break.fillna(0)) / (cov.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)
cov["cls"] = cov.nfl_id_d.map(base.draft_year)
cov["z_bp"] = cov.nfl_id_d.map(z(cb["drill:DB:BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION:top_speed"], cov.nfl_id_d))
cov["z_forty"] = cov.nfl_id_d.map(z(-base.forty, cov.nfl_id_d))                 # faster = positive
cov["z_shuttle"] = cov.nfl_id_d.map(z(-base.short_shuttle, cov.nfl_id_d))
g2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
F1 = "sep_gain_break ~ {} + wr_skill_loo + C(team_coverage_man_zone) + C(route_ran)"
dd = cov.dropna(subset=["sep_gain_break", "wr_skill_loo"])
print("L1 CB: separation the receiver gains at his break (CB coverage reps)")
report("   backpedal-45 transition top speed (tracking)", dd, F1.format("z_bp"), "z_bp", g2, "nfl_id_d", "z_bp")
report("   official 40 (faster = +)", dd, F1.format("z_forty"), "z_forty", g2, "nfl_id_d", "z_forty")
report("   official short shuttle (faster = +)", dd, F1.format("z_shuttle"), "z_shuttle", g2, "nfl_id_d", "z_shuttle")
d2 = dd.dropna(subset=["z_bp", "z_forty"])
mm = smf.ols(F1.format("z_bp + z_forty"), data=d2).fit(cov_type="cluster", cov_kwds={"groups": g2(d2)})
print(f"   both in one model: tracking b {mm.params['z_bp']:+.4f} (p {mm.pvalues['z_bp']:.3f}) | official 40 b {mm.params['z_forty']:+.4f} (p {mm.pvalues['z_forty']:.3f})")

# ---------------- L2 IDL 40 sustained accel -> pressure (play level)
gp = pd.read_parquet(f"{ROOT}/data/bend/game_bend_play.parquet")
rush = gp[gp.nfl_position.isin(["DT", "NT"]) & (gp.lined_up_position == "INTERIOR_LINE") & (gp.play_type == "DROPBACK") &
          gp.player_get_off.notna() & gp.season_type.isin(["REG", "POST"]) & (gp.nullified != "Y")].copy()
rush["ytg_b"] = pd.cut(rush.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
idl = cat("IDL")
rush["cls"] = rush.nfl_id.map(base.draft_year)
rush["z_acc"] = rush.nfl_id.map(z(idl["drill:FORTY_YARD_DASH:accel_sustained_peak"], rush.nfl_id))
rush["z_ten"] = rush.nfl_id.map(z(-base.ten_yd_split, rush.nfl_id)); rush["z_forty"] = rush.nfl_id.map(z(-base.forty, rush.nfl_id))
rush["z_w"] = rush.nfl_id.map(z(base.combine_weight, rush.nfl_id))
g1 = lambda d: pd.factorize(d.nfl_id)[0]
F2 = "pressure ~ {} + z_w + C(blitzing) + C(down) + C(ytg_b)"
rr = rush.dropna(subset=["blitzing", "down", "ytg_b", "z_w"])
print("\nL2 IDL: pass-rush pressure per snap (linear probability, + weight)")
report("   40 sustained acceleration (tracking)", rr, F2.format("z_acc"), "z_acc", g1, "nfl_id", "z_acc")
report("   official 10-yd split (faster = +)", rr, F2.format("z_ten"), "z_ten", g1, "nfl_id", "z_ten")
report("   official 40 (faster = +)", rr, F2.format("z_forty"), "z_forty", g1, "nfl_id", "z_forty")

# ---------------- L3 EDGE 40 top speed -> blocker ppa in head-to-head reps
rep = pd.read_parquet(f"{ROOT}/data/bend/matchups.parquet")
rep = rep[rep.primary & rep.season_type.isin(["REG", "POST"]) & (rep.nullified != "Y") & (rep.dl_align == "EDGE")].copy()
rep["ytg_b"] = pd.cut(rep.yards_to_go.fillna(10), [-1, 3, 7, 12, 99], labels=False)
edge = cat("EDGE")
rep["cls"] = rep.dl_id.map(base.draft_year)
rep["z_vmax"] = rep.dl_id.map(z(edge["drill:FORTY_YARD_DASH:top_speed"], rep.dl_id))
rep["z_forty"] = rep.dl_id.map(z(-base.forty, rep.dl_id))
ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
w = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE")]
wave = w.groupby("nfl_id").s.mean()
rep["z_blk_wave"] = rep.ol_id.map(z(wave, rep.ol_id)).fillna(0)
rep["z_dlw"] = rep.dl_id.map(z(base.combine_weight, rep.dl_id))
gr = lambda d: np.column_stack([pd.factorize(d.dl_id)[0], pd.factorize(d.ol_id)[0]])
F3 = "ppa ~ {} + z_blk_wave + z_dlw + C(ol_align) + C(down) + C(ytg_b) + C(blitzing)"
re_ = rep.dropna(subset=["ppa", "down", "ytg_b", "z_dlw"])
print("\nL3 EDGE: blocker's peak pressure probability on head-to-head reps (+ blocker wave speed, rusher weight)")
report("   40 top speed (tracking)", re_, F3.format("z_vmax"), "z_vmax", gr, "dl_id", "z_vmax")
report("   official 40 (faster = +)", re_, F3.format("z_forty"), "z_forty", gr, "dl_id", "z_forty")

# ---------------- L4 WR curl lean -> separation on hitch routes (play level)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet"); g = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season_type"]]
r = pp.merge(g, on="game_id"); r = r[(r.season_type != "PRE") & (r.play_nullified_by_penalty != "Y") & (r.route_ran == "HITCH") & r.separation_at_pass_forward.notna()]
r = r[r.nfl_id.isin(base.index[base.nfl_position == "WR"])].copy()
wr = cat("WR")
r["cls"] = r.nfl_id.map(base.draft_year)
r["z_lean"] = r.nfl_id.map(z(wr["drill:WR:CURL_ROUTE_RIGHT:turn_lean"], r.nfl_id))
r["z_shuttle"] = r.nfl_id.map(z(-base.short_shuttle, r.nfl_id)); r["z_3c"] = r.nfl_id.map(z(-base.three_cone, r.nfl_id))
r["z_w"] = r.nfl_id.map(z(base.combine_weight, r.nfl_id))
r["press"] = (r.cushion <= 2).astype(int)
F4 = "separation_at_pass_forward ~ {} + z_w + C(lined_up_position) + C(team_coverage_man_zone) + press"
r4 = r.dropna(subset=["z_w", "team_coverage_man_zone", "lined_up_position"])
print("\nL4 WR: separation at the throw on hitch routes (+ weight, alignment, coverage, press)")
report("   curl-drill turn lean (tracking)", r4, F4.format("z_lean"), "z_lean", g1, "nfl_id", "z_lean")
report("   official short shuttle (faster = +)", r4, F4.format("z_shuttle"), "z_shuttle", g1, "nfl_id", "z_shuttle")
report("   official 3-cone (faster = +)", r4, F4.format("z_3c"), "z_3c", g1, "nfl_id", "z_3c")
