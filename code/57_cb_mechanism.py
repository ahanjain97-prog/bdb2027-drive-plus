"""CB Transition Drive Speed: in-game mechanism, moderators, and quantified outcomes.

In-game (CB coverage reps; the DB's body orientation IS tracked in games), after the receiver's break (frame jb):
  db_pre_backpedal  share of the 1.0 s before the break the DB moves backward relative to his facing (cos(o - dir) < -0.5)
  db_react_lag      time from the receiver's break to the DB's own peak |lateral accel| (his break), within 1.0 s
  db_drive_speed    DB's max speed in the 1.0 s after the receiver's break
  db_drive_gain     DB speed 0.7 s after the break minus at the break
  db_closing        rate the receiver-DB distance shrinks over the 0.7 s after the break (yd/s; + = closing)
Chain: combine drive speed -> in-game drive / closing (player level) -> separation gained (rep level) -> completion.
Moderators (interaction with the combine metric; player-permutation p): man vs zone; assignment family; cushion; route
family; CB alignment (outside vs slot); receiver top speed.
Outcomes per target: completion, receiving yards, EPA (offense), with sep_at_throw as the mediator.
"""
import json, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
sys.path.insert(0, f"{ROOT}/prereg")
from cb_drive_speed import cb_drive_speed
rng = np.random.default_rng(5)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
drive = cb_drive_speed(pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet"))
cbs = base.index[base.nfl_position == "CB"]
cov = pd.read_parquet(f"{ROOT}/data/outcomes/coverage_reps.parquet")
cov = cov[cov.season_type.isin(["REG", "POST"]) & (cov.nullified != "Y") & cov.nfl_id_d.isin(cbs)].reset_index(drop=True)

# ---------------- in-game DB mechanics around the receiver's break
get = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = cov.idx_r.values, cov.idx_d.values
sr, lr = get(A, "s"), get(A, "a_lat")
xr, yr = get(A, "x_rel"), get(A, "y_field"); xd, yd = get(B, "x_rel"), get(B, "y_field")
sd, ld, cod = get(B, "s"), get(B, "a_lat"), get(B, "cos_od")
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
lds = ld.copy(); lds[:, 1:-1] = (ld[:, :-2] + ld[:, 1:-1] + ld[:, 2:]) / 3
pr = get(A, "post_pass")
bw = vv & (k >= 5) & (k <= 30) & (sr > 2.0) & (pr < .5)
a = np.where(bw, np.abs(ls), -1); jb = a.argmax(1); hasb = a.max(1) > 0
dist = np.hypot(xr - xd, yr - yd)
r = np.arange(len(cov))
pre_w = vv & (cols >= (jb - 10)[:, None]) & (cols < jb[:, None]) & (sd[:, :] > 1.0)
cov["db_pre_backpedal"] = np.where(pre_w.sum(1) >= 3, (pre_w & (cod < -0.5)).sum(1) / np.maximum(pre_w.sum(1), 1), np.nan)
post_w = vv & (cols >= jb[:, None]) & (cols <= (jb + 10)[:, None])
pdl = np.where(post_w, np.abs(lds), -1); jdb = pdl.argmax(1)
cov["db_react_lag"] = np.where(hasb & (pdl.max(1) > 1.5), (jdb - jb) / 10.0, np.nan)
cov["db_drive_speed"] = np.where(hasb & (post_w.sum(1) >= 5), np.where(post_w, sd, 0).max(1), np.nan)
j7 = np.clip(jb + 7, 0, T - 1)
ok7 = hasb & vv[r, j7] & vv[r, jb]
cov["db_drive_gain"] = np.where(ok7, sd[r, j7] - sd[r, jb], np.nan)
cov["db_closing"] = np.where(ok7, (dist[r, jb] - dist[r, j7]) / 0.7, np.nan)
cov["drive_combine"] = cov.nfl_id_d.map(drive)
dv = cov.drop_duplicates("nfl_id_d").set_index("nfl_id_d").drive_combine
cov["z_drive"] = (cov.drive_combine - dv.mean()) / dv.std()
tot = cov.groupby("nfl_id_r").sep_gain_break.agg(["sum", "count"])
cov["wr_skill"] = (cov.nfl_id_r.map(tot["sum"]) - cov.sep_gain_break.fillna(0)) / (cov.nfl_id_r.map(tot["count"]) - 1).replace(0, np.nan)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet").set_index(["game_id", "play_id", "nfl_id"])
kr = pd.MultiIndex.from_arrays([cov.game_id, cov.play_id, cov.nfl_id_r])
cov["epa"] = pp.expected_points_added.reindex(kr).values
cov["press_bin"] = pd.cut(cov.cushion, [-1, 2, 6, 99], labels=["press (<=2)", "off 2-6", "deep off (>6)"])
FAM = {"SLANT": "in", "IN": "in", "CROSS": "in", "POST": "in", "ANGLE": "in", "OUT": "out", "CORNER": "out", "FLAT": "out",
       "WHEEL": "out", "HITCH": "stop", "GO": "vertical", "SCREEN": "screen"}
cov["fam"] = cov.route_ran.map(FAM)
def asg(a_):
    if a_ == "MAN": return "man"
    if isinstance(a_, str) and a_.startswith("DEEP"): return "deep zone (thirds/halves)"
    if isinstance(a_, str) and a_.startswith("QUARTERS"): return "quarters"
    if isinstance(a_, str) and ("FLAT" in a_ or "CURL" in a_ or "HOOK" in a_): return "underneath zone (flat/curl/hook)"
    return "other"
cov["asg"] = cov.coverage_assignment.map(asg)
cov["cb_slot"] = (cov.lined_up_position_d == "SLOT_CB").astype(int)
cat_wr = pd.read_parquet(f"{ROOT}/data/features/catalog_WR.parquet")
wv = cat_wr["drill:FORTY_YARD_DASH:top_speed"]
cov["wr_fast"] = cov.nfl_id_r.map(wv) > wv.reindex(cov.nfl_id_r.unique()).median()
cov.to_parquet(f"{ROOT}/data/outcomes/cb_reps_enriched.parquet", index=False)
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 220)

print("=== 1) does the combine drill show up in the CB's NFL movement? (player level, CBs with >= 10 reps) ===")
mech = ["db_pre_backpedal", "db_react_lag", "db_drive_speed", "db_drive_gain", "db_closing"]
P = cov.groupby("nfl_id_d")[mech].mean()[cov.groupby("nfl_id_d").size() >= 10]
P["drive_combine"] = drive
for m in mech:
    d = P[["drive_combine", m]].dropna()
    print(f"  combine drive speed vs NFL {m:17s} r {d.corr().iloc[0, 1]:+.2f}  (n {len(d)})")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    Pz = cov[cov.team_coverage_man_zone == cz].groupby("nfl_id_d")[["db_closing", "db_drive_speed"]].mean()
    Pz = Pz[cov[cov.team_coverage_man_zone == cz].groupby("nfl_id_d").size() >= 6].join(drive.rename("drive"))
    print(f"  {cz}: combine drive vs NFL closing r {Pz[['drive', 'db_closing']].dropna().corr().iloc[0, 1]:+.2f}, "
          f"vs NFL drive speed r {Pz[['drive', 'db_drive_speed']].dropna().corr().iloc[0, 1]:+.2f} (n {Pz.drive.notna().sum()})")

print("\n=== 2) do those NFL mechanics produce separation outcomes? (rep level, mechanical check) ===")
for m in mech:
    d = cov.dropna(subset=[m, "sep_gain_break", "wr_skill"])
    d["zm"] = (d[m] - d[m].mean()) / d[m].std()
    mm = smf.ols("sep_gain_break ~ zm + wr_skill + C(team_coverage_man_zone) + C(route_ran)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    print(f"  NFL {m:17s} -> separation gained: {mm.params['zm']:+.3f} yd/SD (p {mm.pvalues['zm']:.3f})")

print("\n=== 3) mediation: combine drive -> NFL closing -> separation gained ===")
d = cov.dropna(subset=["z_drive", "db_closing", "sep_gain_break", "wr_skill"])
f0 = "sep_gain_break ~ z_drive + wr_skill + C(team_coverage_man_zone) + C(route_ran)"
m0 = smf.ols(f0, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
m1 = smf.ols(f0 + " + db_closing", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
print(f"  total effect {m0.params['z_drive']:+.4f} (p {m0.pvalues['z_drive']:.3f}) -> with NFL closing in the model {m1.params['z_drive']:+.4f} "
      f"(p {m1.pvalues['z_drive']:.3f}); share mediated {1 - m1.params['z_drive'] / m0.params['z_drive']:.0%}  (reps {len(d)})")


def inter(d, mod, label, n_perm=300):
    d = d.dropna(subset=["z_drive", "sep_gain_break", "wr_skill", mod])
    f = f"sep_gain_break ~ z_drive * C({mod}) + wr_skill + C(route_ran)"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    levels = sorted(d[mod].unique(), key=str)
    out = []
    for lv in levels:
        sub = d[d[mod] == lv]
        if sub.nfl_id_d.nunique() < 8 or len(sub) < 60:
            out.append(f"{lv}: n/a (reps {len(sub)})"); continue
        ms = smf.ols("sep_gain_break ~ z_drive + wr_skill + C(route_ran)", data=sub).fit(cov_type="cluster", cov_kwds={"groups": G2(sub)})
        out.append(f"{lv}: {ms.params['z_drive']:+.3f} (p {ms.pvalues['z_drive']:.3f}, reps {len(sub)}, CBs {sub.nfl_id_d.nunique()})")
    # joint interaction test by player permutation (F on interaction terms)
    it = [t for t in m.params.index if t.startswith("z_drive:")]
    def stat(dd):
        mm = smf.ols(f, data=dd).fit(); return float(mm.f_test(" = 0, ".join(it) + " = 0").fvalue) if it else 0.0
    s0 = stat(d); vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").z_drive; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["z_drive"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(stat(dd))
    p = (1 + (np.array(null) >= s0).sum()) / (n_perm + 1)
    print(f"  {label:28s} interaction perm p {p:.3f} | " + " | ".join(out))


print("\n=== 4) where does it matter? slope of separation gained (yd per SD of combine drive speed) by moderator ===")
inter(cov, "team_coverage_man_zone", "man vs zone")
inter(cov, "asg", "CB assignment")
inter(cov, "press_bin", "cushion at the snap")
inter(cov[cov.fam.isin(["in", "out", "stop", "vertical"])], "fam", "route family")
inter(cov, "cb_slot", "slot CB (1) vs outside (0)")
inter(cov.dropna(subset=["wr_fast"]), "wr_fast", "receiver 40 top speed > median")

print("\n=== 5) quantified outcomes when targeted (per SD and per yd/s of combine drive speed) ===")
tg = cov[cov.targeted == 1].dropna(subset=["z_drive"]).copy()
sd_drive = dv.std()
for y, lab in [("completion", "completion rate"), ("rec_yards", "receiving yards per target"), ("epa", "EPA per target (offense)")]:
    d = tg.dropna(subset=[y])
    for cz in [None, "ZONE_COVERAGE", "MAN_COVERAGE"]:
        dd = d if cz is None else d[d.team_coverage_man_zone == cz]
        f = f"{y} ~ z_drive" + (" + C(team_coverage_man_zone)" if cz is None else "")
        mm = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": G2(dd)})
        print(f"  {lab:28s} {cz or 'all':14s}: {mm.params['z_drive']:+.3f} per SD = {mm.params['z_drive'] / sd_drive:+.3f} per yd/s "
              f"(p {mm.pvalues['z_drive']:.3f}, targets {len(dd)}, mean {dd[y].mean():.3f})")
d = tg.dropna(subset=["sep_at_throw", "completion"])
m_c = smf.logit("completion ~ sep_at_throw + C(team_coverage_man_zone)", data=d).fit(disp=0)
print(f"  completion vs separation at the throw (logit): +1 yd -> odds x{np.exp(m_c.params['sep_at_throw']):.2f}")
m_med = smf.ols("completion ~ z_drive + sep_at_throw + C(team_coverage_man_zone)", data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
print(f"  completion ~ drive + separation at throw: drive {m_med.params['z_drive']:+.3f} (p {m_med.pvalues['z_drive']:.3f}), separation {m_med.params['sep_at_throw']:+.3f}/yd")

print("\n=== 6) dose-response: combine drive speed bins (yd/s) ===")
cov["drive_bin"] = pd.cut(cov.drive_combine, [0, 8.6, 9.0, 9.4, 99], labels=["< 8.6", "8.6-9.0", "9.0-9.4", ">= 9.4"])
tab = cov.groupby(["drive_bin", "team_coverage_man_zone"]).agg(sep_gain=("sep_gain_break", "mean"), sep_throw=("sep_at_throw", "mean"),
                                                                reps=("sep_gain_break", "size"), cbs=("nfl_id_d", "nunique"))
tab2 = tg.assign(drive_bin=pd.cut(tg.drive_combine, [0, 8.6, 9.0, 9.4, 99], labels=["< 8.6", "8.6-9.0", "9.0-9.4", ">= 9.4"])).groupby("drive_bin").agg(
    completion=("completion", "mean"), yds_per_target=("rec_yards", "mean"), epa=("epa", "mean"), targets=("completion", "size"))
print(tab.round(3).to_string()); print(tab2.round(3).to_string())
