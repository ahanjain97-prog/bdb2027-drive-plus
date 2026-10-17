"""Step 6 -- scheme fit: is drive speed worth more to a CB whose TEAM plays more match-type zone?

Team-season coverage mix from player_play (defensive pool-player rows, one row per team x play, dropbacks):
  match_share = share of Cover 3 + Cover 4 + Cover 6;  man_share = Cover 0/1/2 man.
Test on ALL of a CB's reps (not only zone-matching reps): y ~ drive * team_match_share + context; player-permutation p
for the interaction. Then a team value-of-drive ranking: predicted effect of +1 SD drive at each team's (pooled
2023-25) match share, for closing (yd/s) and completion over expected (pp).
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(89)
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
v = Z["top"].reindex(cov.nfl_id_d.unique()); cov["c_top"] = cov.nfl_id_d.map((Z["top"] - v.mean()) / v.std())
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
games = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season"]]
dfn = pp[(pp.team_abbr != pp.possession_team) & pp.pass_result.isin(["C", "I", "IN", "S", "R"]) & pp.team_coverage_type.notna()]
dfn = dfn.drop_duplicates(["game_id", "play_id", "team_abbr"]).merge(games, on="game_id")
MATCH = {"COVER_3_ZONE", "COVER_4_ZONE", "COVER_6_ZONE"}; MAN = {"COVER_0_MAN", "COVER_1_MAN", "COVER_2_MAN"}
ts = dfn.groupby(["team_abbr", "season"]).agg(plays=("play_id", "size"), match_share=("team_coverage_type", lambda s: s.isin(MATCH).mean()),
                                             man_share=("team_coverage_type", lambda s: s.isin(MAN).mean())).reset_index()
print(f"team-seasons {len(ts)}, plays per team-season median {ts.plays.median():.0f}; match share range {ts.match_share.min():.2f}-{ts.match_share.max():.2f} (sd {ts.match_share.std():.2f})")
own = pp[["game_id", "play_id", "nfl_id", "team_abbr"]].rename(columns={"nfl_id": "nfl_id_d", "team_abbr": "def_team"})
cov = cov.merge(own, on=["game_id", "play_id", "nfl_id_d"], how="left").merge(ts.rename(columns={"team_abbr": "def_team"}), on=["def_team", "season"], how="left")
cov["match_c"] = (cov.match_share - ts.match_share.mean()) / ts.match_share.std()
cov["man_c"] = (cov.man_share - ts.man_share.mean()) / ts.man_share.std()
print("CB reps with team mix:", cov.match_share.notna().sum(), "| CB-team-seasons:", cov.drop_duplicates(["nfl_id_d", "def_team", "season"]).shape[0])
print("tracking check -- share of a CB's reps that are zone-matching vs team match share (CB-team-season, >= 10 reps): r",
      round(cov.groupby(["nfl_id_d", "def_team", "season"]).filter(lambda g: len(g) >= 10).groupby(["nfl_id_d", "def_team", "season"]).agg(
          zm=("cell", lambda s: (s == "zone-matching").mean()), ms=("match_share", "first")).corr().iloc[0, 1], 2))
CL = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])


def inter(d, y, ctx, mod, n_perm=400):
    d = d.dropna(subset=[y, "c_top", mod] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    f = f"{y} ~ c_top * {mod} + {ctx}"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": CL(d)})
    t0 = smf.ols(f, data=d).fit().tvalues[f"c_top:{mod}"]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d").c_top; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd["c_top"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues[f"c_top:{mod}"])
    return m, (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d), d.nfl_id_d.nunique()


TASKS = [("db_closing", cov, "wr_skill + C(route_ran)"), ("sep_gain_break", cov, "wr_skill + C(route_ran)"),
         ("sep_at_throw", cov, "wr_skill + C(route_ran)"), ("cpoe", cov[cov.targeted == 1], "rec_target_prop")]
fits = {}
for mod in ["match_c", "man_c"]:
    print(f"\n=== drive x team {mod.replace('_c', ' share')} (per SD of the team share), ALL CB reps, no man/zone control ===")
    for y, d, ctx in TASKS:
        m, p, n, k = inter(d, y, ctx, mod)
        fits[(mod, y)] = m
        print(f"  {y:14s} drive {m.params['c_top']:+.3f} (p {m.pvalues['c_top']:.3f}) | team share {m.params[mod]:+.3f} | drive x share {m.params[f'c_top:{mod}']:+.3f} "
              f"(clustered p {m.pvalues[f'c_top:{mod}']:.3f}, perm p {p:.3f}) | reps {n}, CBs {k}")
# team value-of-drive ranking (pooled seasons)
tm = dfn.groupby("team_abbr").agg(match_share=("team_coverage_type", lambda s: s.isin(MATCH).mean()), man_share=("team_coverage_type", lambda s: s.isin(MAN).mean()))
tm["match_c"] = (tm.match_share - ts.match_share.mean()) / ts.match_share.std()
mc, mp_ = fits[("match_c", "db_closing")], fits[("match_c", "cpoe")]
tm["closing_per_SD"] = mc.params["c_top"] + mc.params["c_top:match_c"] * tm.match_c
tm["cpoe_pp_per_SD"] = 100 * (mp_.params["c_top"] + mp_.params["c_top:match_c"] * tm.match_c)
tm = tm.sort_values("closing_per_SD", ascending=False)
tm.round(3).to_csv(f"{ROOT}/results/73e_team_value_of_drive.csv")
print("\n=== value of +1 SD combine drive speed by team (pooled 2023-25 mix; model-implied) ===")
print(tm[["match_share", "man_share", "closing_per_SD", "cpoe_pp_per_SD"]].round(3).to_string())
