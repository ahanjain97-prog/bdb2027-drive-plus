"""Team-level validation: when a defense plays ZONE, is it better (lower offensive EPA) with a high-drive CB on the field?

Unit: defensive dropback with >= 1 pool CB on the field (player_play). Exposure = mean combine drive speed (frozen, within-
year z, standardized over CBs) of the pool CBs on the field. Outcomes (offense perspective; lower = better defense):
EPA, success (EPA > 0), completion (pass attempts).
  M1  y ~ drive * zone + context + defense team-season FE + offense team-season FE
      -> drive x zone = does the zone-minus-man gap improve with drive, net of each defense's overall quality.
  M2  + defense team-season x zone FE (each defense's own zone-man gap absorbed) -> identified ONLY by which CB is on the
      field within the same team-season.
  p: player permutation (shuffle drive across CBs, rebuild play exposure; 500), via FWL on a precomputed QR.
Natural experiment check: is drive related to draft slot / playing time? (teams cannot see it on a stopwatch)
Placebos: run plays (drive should not matter); CB 40 time and CB weight in place of drive; per-season replication.
Team-season view: zone EPA minus man EPA vs snap-weighted drive of the pool CBs (scatter + permutation).
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(109)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
games = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season", "season_type"]] if "season_type" in pd.read_parquet(f"{ROOT}/data/raw/games.parquet").columns else pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season"]]
cbs = base.index[base.nfl_position == "CB"]
DROP = ["C", "I", "IN", "S", "R"]
d = pp[pp.nfl_id.isin(cbs) & (pp.team_abbr != pp.possession_team)].merge(games, on="game_id")
# per-CB exposures: drive (frozen), 40, weight -- standardized over CBs who appear on defense
appear = d.nfl_id.unique()
def zser(s):
    s = s.reindex(appear); return (s - s.mean()) / s.std()
TR = {"drive": zser(Z.top), "forty_fast": zser(-base.forty), "weight": zser(base.combine_weight)}
print("natural-experiment check (CBs on defense): r(drive, log pick) "
      f"{pd.concat([TR['drive'], base.log_pick.reindex(appear)], axis=1).corr().iloc[0, 1]:+.2f}, r(drive, UDFA) "
      f"{pd.concat([TR['drive'], base.udfa.reindex(appear)], axis=1).corr().iloc[0, 1]:+.2f}, r(drive, dropback snaps) "
      f"{pd.concat([TR['drive'], d[d.pass_result.isin(DROP)].groupby('nfl_id').size().reindex(appear)], axis=1).corr().iloc[0, 1]:+.2f}, "
      f"r(drive, 40 speed) {pd.concat([TR['drive'], TR['forty_fast']], axis=1).corr().iloc[0, 1]:+.2f}")


def plays(kind):
    x = d[d.pass_result.isin(DROP)] if kind == "pass" else d[~d.pass_result.isin(DROP)]
    P = x.drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "season", "team_abbr", "possession_team", "down", "yards_to_go", "yardline_side",
                                                  "yardline_number", "quarter", "pre_snap_home_score", "pre_snap_visitor_score", "pass_result",
                                                  "expected_points_added", "team_coverage_man_zone"]].copy()
    P["ytg_goal"] = np.where(P.yardline_side == P.possession_team, 100 - P.yardline_number, P.yardline_number).astype(float)
    P["ytg_bin"] = pd.cut(P.ytg_goal, [-1, 10, 20, 40, 60, 80, 101]).astype(str)
    P["dist_bin"] = pd.cut(P.yards_to_go, [0, 2, 5, 8, 10, 15, 99]).astype(str)
    P["def_ts"] = P.team_abbr + P.season.astype(str); P["off_ts"] = P.possession_team + P.season.astype(str)
    P["epa"] = P.expected_points_added; P["success"] = (P.epa > 0).astype(float)
    P["completion"] = np.where(P.pass_result.isin(["C", "I", "IN"]), (P.pass_result == "C").astype(float), np.nan)
    P["zone"] = (P.team_coverage_man_zone == "ZONE_COVERAGE").astype(float)
    on = x[["game_id", "play_id", "nfl_id"]]
    return P, on


def exposure(on, ser):
    e = on.assign(v=on.nfl_id.map(ser)).dropna(subset=["v"]).groupby(["game_id", "play_id"]).v.mean()
    return e


FIX = "C(down) + C(dist_bin) + C(ytg_bin) + C(quarter) + C(def_ts) + C(off_ts)"


class FWL:
    """Residual-maker for a fixed design (FE + controls) so permutations only re-residualize the exposure columns."""
    def __init__(self, P, extra_fe=False):
        cols = [pd.get_dummies(P[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in ["down", "dist_bin", "ytg_bin", "quarter", "def_ts", "off_ts"]]
        M = [np.ones((len(P), 1)), P[["zone"]].values] + [c.values for c in cols]
        if extra_fe:
            M.append(pd.get_dummies(P.def_ts + "_" + P.zone.astype(int).astype(str), drop_first=True, dtype=float).values)
        M = np.hstack(M); M = M[:, np.linalg.matrix_rank(M[:2000]) * 0 + np.arange(M.shape[1])]
        q, r = np.linalg.qr(M); keep = np.abs(np.diag(r)) > 1e-8; self.Q = q[:, keep]
    def res(self, v):
        return v - self.Q @ (self.Q.T @ v)


def tstat(fw, y_r, cols):
    Xr = np.column_stack([fw.res(c) for c in cols]); b, *_ = np.linalg.lstsq(Xr, y_r, rcond=None)
    e = y_r - Xr @ b; s2 = e @ e / (len(e) - fw.Q.shape[1] - Xr.shape[1]); cov = s2 * np.linalg.inv(Xr.T @ Xr)
    return b, b / np.sqrt(np.diag(cov))


def run(kind, trait, outcomes, n_perm=500, models=("M1", "M2"), by_zone=True):
    P, on = plays(kind)
    ser = TR[trait]; P["x"] = P.set_index(["game_id", "play_id"]).index.map(exposure(on, ser))
    P = P.dropna(subset=["x"]).reset_index(drop=True)
    if by_zone:
        P = P[P.team_coverage_man_zone.notna()].reset_index(drop=True)
    res = []
    for mname in models:
        fw = FWL(P, extra_fe=(mname == "M2"))
        for y in outcomes:
            ok = P[y].notna().values
            if ok.mean() < 1:
                Pm = P[ok].reset_index(drop=True); fwm = FWL(Pm, extra_fe=(mname == "M2"))
            else:
                Pm, fwm = P, fw
            y_r = fwm.res(Pm[y].values.astype(float))
            cols = [Pm.x.values, Pm.x.values * Pm.zone.values] if by_zone else [Pm.x.values]
            b, t = tstat(fwm, y_r, cols)
            ids = Pm.set_index(["game_id", "play_id"]).index
            onm = on[on.set_index(["game_id", "play_id"]).index.isin(ids)]
            vals = ser.dropna(); null = []
            for _ in range(n_perm):
                sp = pd.Series(rng.permutation(vals.values), index=vals.index)
                xp = ids.map(exposure(onm, sp)).values.astype(float); xp = np.where(np.isnan(xp), np.nanmean(xp), xp)
                cp = [xp, xp * Pm.zone.values] if by_zone else [xp]
                null.append(tstat(fwm, y_r, cp)[1])
            null = np.array(null)
            r0 = {"plays": kind, "trait": trait, "model": mname, "outcome": y, "n_plays": len(Pm), "n_CBs": int(onm.nfl_id.isin(vals.index).groupby(onm.nfl_id).any().sum())}
            if by_zone:
                r0.update({"b_man": b[0], "b_zone": b[0] + b[1], "b_zone_minus_man": b[1],
                           "perm_p_interaction": (1 + (np.abs(null[:, 1]) >= abs(t[1])).sum()) / (n_perm + 1)})
                # zone-only slope permutation p (refit on zone plays as a separate check is done below)
            else:
                r0.update({"b": b[0], "perm_p": (1 + (np.abs(null[:, 0]) >= abs(t[0])).sum()) / (n_perm + 1)})
            res.append(r0); print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r0.items()}, flush=True)
    return pd.DataFrame(res), P


print("\n=== MAIN: drive x zone on defensive dropbacks (per SD of drive on the field; EPA = offense) ===")
R1, P = run("pass", "drive", ["epa", "success", "completion"])
print(f"  plays by coverage: {P.team_coverage_man_zone.value_counts().to_dict()}")
print("\n=== zone-only and man-only slopes (M1 structure, separate fits; player permutation) ===")
for cz in ["ZONE_COVERAGE", "MAN_COVERAGE"]:
    Pz = P[P.team_coverage_man_zone == cz].reset_index(drop=True); fw = FWL(Pz)
    _, on = plays("pass"); ids = Pz.set_index(["game_id", "play_id"]).index; onm = on[on.set_index(["game_id", "play_id"]).index.isin(ids)]
    for y in ["epa", "success"]:
        y_r = fw.res(Pz[y].values.astype(float)); b, t = tstat(fw, y_r, [Pz.x.values]); null = []
        vals = TR["drive"].dropna()
        for _ in range(500):
            sp = pd.Series(rng.permutation(vals.values), index=vals.index); xp = ids.map(exposure(onm, sp)).values.astype(float)
            null.append(tstat(fw, y_r, [np.where(np.isnan(xp), np.nanmean(xp), xp)])[1][0])
        print(f"  {cz:14s} {y:8s} {b[0]:+.4f} per SD (perm p {(1 + (np.abs(null) >= abs(t[0])).sum()) / 501:.3f}, plays {len(Pz)})")
print("\n=== PLACEBOS ===")
Rr, _ = run("run", "drive", ["epa", "success"], n_perm=300, models=("M1",), by_zone=False)
R40, _ = run("pass", "forty_fast", ["epa", "success"], n_perm=300, models=("M1",))
Rw, _ = run("pass", "weight", ["epa", "success"], n_perm=300, models=("M1",))
print("\n=== per-season replication of drive x zone (M1, EPA and success) ===")
for s in sorted(P.season.unique()):
    Ps = P[P.season == s].reset_index(drop=True); fw = FWL(Ps)
    out = []
    for y in ["epa", "success"]:
        b, t = tstat(fw, fw.res(Ps[y].values.astype(float)), [Ps.x.values, Ps.x.values * Ps.zone.values])
        out.append(f"{y}: man {b[0]:+.4f}, zone {b[0] + b[1]:+.4f}, zone-man {b[1]:+.4f} (t {t[1]:+.2f})")
    print(f"  {s} (plays {len(Ps)}): " + " | ".join(out))
pd.concat([R1, Rr, R40, Rw]).to_csv(f"{ROOT}/results/74_team_epa.csv", index=False)

# ---------------- team-season view
P["ctx_res"] = smf.ols("epa ~ C(down) + C(dist_bin) + C(ytg_bin) + C(quarter) + C(off_ts)", data=P).fit().resid
TS = P.groupby("def_ts").apply(lambda g: pd.Series({"zone_epa": g.ctx_res[g.zone == 1].mean(), "man_epa": g.ctx_res[g.zone == 0].mean(),
                                                    "n_zone": (g.zone == 1).sum(), "n_man": (g.zone == 0).sum(), "drive": g.x.mean()}), include_groups=False)
TS = TS[(TS.n_zone >= 60) & (TS.n_man >= 25)]; TS["gap"] = TS.zone_epa - TS.man_epa
w = 1 / (1 / TS.n_zone + 1 / TS.n_man)
wc = lambda a, b, w: np.cov(a, b, aweights=w)[0, 1] / np.sqrt(np.cov(a, a, aweights=w)[0, 1] * np.cov(b, b, aweights=w)[0, 1])
r_gap, r_zone = wc(TS.drive, TS.gap, w), wc(TS.drive, TS.zone_epa, TS.n_zone)
print(f"\n=== team-season view ({len(TS)} defense-seasons with >= 60 zone and >= 25 man dropbacks with a pool CB on the field) ===")
print(f"  weighted r(snap-weighted CB drive, zone EPA - man EPA) {r_gap:+.2f} | r(drive, zone EPA) {r_zone:+.2f}")
TS.round(3).to_csv(f"{ROOT}/results/74_team_season.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a_, ycol, lab in [(ax[0], "zone_epa", "zone dropbacks: EPA/play (context-adjusted)"), (ax[1], "gap", "zone minus man EPA/play")]:
    a_.scatter(TS.drive, TS[ycol], s=TS.n_zone / 4, alpha=.6, color="#2471a3")
    for i, r in TS.iterrows():
        a_.annotate(i, (r.drive, r[ycol]), fontsize=7, alpha=.7)
    cf = np.polyfit(TS.drive, TS[ycol], 1, w=np.sqrt(TS.n_zone)); xs = np.linspace(TS.drive.min(), TS.drive.max(), 10)
    a_.plot(xs, np.polyval(cf, xs), color="#c0392b"); a_.axhline(0, color="k", lw=.5)
    a_.set_xlabel("mean combine drive speed (SD) of the team's pool CBs on the field"); a_.set_ylabel(lab + "  (lower = better D)")
fig.suptitle("Defense-seasons: combine drive speed of the CBs on the field vs zone pass defense", fontsize=12)
fig.tight_layout(); fig.savefig(f"{ROOT}/results/74_team_season_scatter.png", dpi=140)
