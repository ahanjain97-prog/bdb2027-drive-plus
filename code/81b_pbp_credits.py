"""Biggest outcome sample: play-by-play coverage credits for every pass snap a graded pool DB played (any receiver).

From player_play.play_description on pass attempts (C/I/IN) with the DB on defense:
  PBU         named in parentheses on an incomplete pass   "pass incomplete deep middle to T.Thornton (K.Lassiter)"
  INT         "INTERCEPTED by K.Joseph"
  DPI         "Penalty on CAR-C.Smith-Wade, Defensive Pass Interference"
  tackle_c    first name in parentheses on a completion = the defender who made the stop after the catch
  ball play   PBU or INT;  credited target = PBU + INT + tackle_c + DPI (a charting-style "target in his area")
Name match: first initial + "." + last name of the DB (suffixes dropped), against the text; brackets [..] = QB hit, ignored.
Tests (player-permutation p, position + alignment + man/zone + season controls; per 10 Drive+ points):
  per snap: ball-play rate, credited-target rate;  per credited target: completion, yards, ball-play share.
Also: alignment CB/SLOT_CB only vs all; zone vs man.
"""
import re, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(173)
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
GR = GR[GR.nfl_position.isin(["CB", "SS", "FS", "DB"])]
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")
games = pd.read_parquet(f"{ROOT}/data/raw/games.parquet")[["game_id", "season"]]
d = pp[pp.nfl_id.isin(GR.index) & (pp.team_abbr != pp.possession_team) & pp.pass_result.isin(["C", "I", "IN"])].merge(games, on="game_id")
d = d[d.play_nullified_by_penalty != "Y"].copy() if "play_nullified_by_penalty" in d else d


def abbr(name):
    parts = [p for p in re.sub(r"\b(Jr\.?|Sr\.?|II|III|IV|V)\b", "", name).split() if p]
    return f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else name


d["abbr"] = d.nfl_id.map(GR.display_name.map(abbr))
desc = d.play_description.fillna("")
no_br = desc.str.replace(r"\[[^\]]*\]", "", regex=True)
first_paren = no_br.str.extract(r"\(([^()]*)\)\s*\.?\s*(?:$|Penalty|PENALTY|\n|[A-Z])", expand=False).fillna("")
# defender list inside the first parenthesis after the receiver ("to X (A; B)"); skip formation parentheses like (Shotgun)
after_to = no_br.str.extract(r"pass[^()]*? to [^()]*?\(([^()]*)\)", expand=False).fillna("")
names_in = after_to.str.split(";").apply(lambda xs: [x.strip() for x in xs if x.strip()])
d["named_first"] = [bool(n) and n[0] == a for n, a in zip(names_in, d.abbr)]
d["named_any"] = [a in n for n, a in zip(names_in, d.abbr)]
d["pbu"] = ((d.pass_result == "I") & d.named_any).astype(float)
d["int_"] = [(r == "IN") and bool(re.search(rf"INTERCEPTED by {re.escape(a)}\b", s)) for r, a, s in zip(d.pass_result, d.abbr, desc)]
d["int_"] = d.int_.astype(float)
d["dpi"] = [bool(re.search(rf"Penalty on [A-Z]{{2,3}}-{re.escape(a)}, Defensive Pass Interference", s, re.I)) for a, s in zip(d.abbr, desc)]
d["dpi"] = d.dpi.astype(float)
d["tackle_c"] = ((d.pass_result == "C") & d.named_first).astype(float)
d["ball_play"] = ((d.pbu + d.int_) > 0).astype(float)
d["credited"] = ((d.pbu + d.int_ + d.tackle_c + d.dpi) > 0).astype(float)
d["comp_c"] = np.where(d.credited == 1, (d.pass_result == "C").astype(float), np.nan)
d["yds_c"] = np.where(d.credited == 1, d.yards_gained, np.nan)
d["bp_c"] = np.where(d.credited == 1, d.ball_play, np.nan)
d["dp"] = d.nfl_id.map((GR.drive_plus - 100) / 10); d["pos"] = d.nfl_id.map(GR.nfl_position); d["is_s"] = (d.pos != "CB").astype(int)
d["aln"] = d.lined_up_position.where(d.lined_up_position.isin(["CB", "SLOT_CB"]), "other")
d["zone"] = (d.team_coverage_man_zone == "ZONE_COVERAGE").astype(int)
print(f"pass snaps {len(d)}, DBs {d.nfl_id.nunique()} | named anywhere: PBU {int(d.pbu.sum())}, INT {int(d.int_.sum())}, DPI {int(d.dpi.sum())}, "
      f"stop after catch {int(d.tackle_c.sum())} -> credited targets {int(d.credited.sum())} (vs 327 pool-receiver coverage targets)")
print("credited targets by alignment:", d.groupby("aln").credited.sum().astype(int).to_dict(), "| per DB median:", d.groupby("nfl_id").credited.sum().median())
chk = d[d.credited == 1].sample(6, random_state=1)
for _, r in chk.iterrows():
    print(f"   [{r.abbr}] pbu {int(r.pbu)} int {int(r.int_)} stop {int(r.tackle_c)} dpi {int(r.dpi)} | {r.play_description[:150]}")
d.to_parquet(f"{ROOT}/data/outcomes/81b_pbp_credits.parquet", index=False)
CL = lambda x: pd.factorize(x.nfl_id)[0]


def test(x, y, ctx, n_perm=500):
    x = x.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"
    m = smf.ols(f, data=x).fit(cov_type="cluster", cov_kwds={"groups": CL(x)})
    t0 = smf.ols(f, data=x).fit().tvalues["dp"]; vals = x.drop_duplicates("nfl_id").set_index("nfl_id").dp; null = []
    for _ in range(n_perm):
        xx = x.copy(); xx["dp"] = xx.nfl_id.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=xx).fit().tvalues["dp"])
    return m.params["dp"], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(x), x.nfl_id.nunique(), x[y].mean()


CTX = "is_s + C(aln) + zone + C(season)"
rows = []
for lab, x in [("all alignments", d), ("at CB / slot", d[d["aln"] != "other"]), ("at CB / slot, zone", d[(d["aln"] != "other") & (d.zone == 1)]),
               ("at CB / slot, man", d[(d["aln"] != "other") & (d.zone == 0)])]:
    for y, unit, sc in [("ball_play", "per snap", 100), ("credited", "per snap", 100), ("pbu", "per snap", 100),
                        ("comp_c", "per credited target", 100), ("bp_c", "per credited target", 100), ("yds_c", "per credited target", 1)]:
        ctx = CTX.replace(" + zone", "") if "zone" in lab or "man" in lab else CTX
        b, p, n, k, mu = test(x, y, ctx)
        rows.append({"sample": lab, "outcome": y, "unit": unit, "per 10 Drive+": b * sc, "mean": mu * sc, "perm p": p, "n": n, "DBs": k})
        print(f"  {lab:20s} {y:9s} ({unit:19s}) {b * sc:+.3f} vs mean {mu * sc:.2f} (perm p {p:.3f}; n {n}, DBs {k})", flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/81b_pbp_credits.csv", index=False)
