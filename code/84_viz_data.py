"""Export compact JSON for the interactive Drive+ story page (artifact). Output results/viz/drive_plus_viz.json

drill    : every CB attempt of the 45-degree drill -- rotated path (x, y), speed by frame, time from turn end, markers
board    : every graded DB -- Drive+, tier, position, class, 40, raw drive speed, NFL coverage reps/targets/completion/open
clock    : separation change around the break by Drive+ third (zone-matching / man), mean + 90% player bootstrap band
chase    : DB position relative to the receiver 1.0 s after the break, per rep, by Drive+ third (zone-matching / man)
throwmap : zone throws by Drive+ third (low/high) x side (his / other) x depth band: n, completion, share
curves   : per-DB open rate, xComp, completion vs Drive+ + rep-level fits with 90% player-bootstrap bands
decomp   : completion gap (110 vs 90) split into before / after the throw, with 90% CIs
spec     : specification-curve estimates (81c)
stats    : headline numbers already computed in earlier scripts (sources noted)
"""
import json, sys, warnings, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(197)
os.makedirs(f"{ROOT}/results/viz", exist_ok=True)
sys.argv = ["x", "none"]
ns = {}; exec(open(f"{ROOT}/code/80_visuals.py").read(), ns)
GR, base = ns["GR"], ns["base"]
r2 = lambda a: [None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else round(float(v), 2) for v in a]
OUT = {}

# ---------------- drill
att = ns["drill_attempts"](); cbs = set(GR[GR.nfl_position == "CB"].index)
drill = []
for r in att:
    if r["nfl_id"] not in cbs:
        continue
    m = (r["t"] >= -1.2) & (r["t"] <= 1.6); sl = slice(r["on"], min(r["jt"] + 8, len(r["x"])))
    drill.append({"id": int(r["nfl_id"]), "t": r2(r["t"][m]), "s": r2(r["s"][m]), "px": r2(r["x"][sl]), "py": r2(r["y"][sl]), "ps": r2(r["s"][sl]),
                  "j0": int(r["j0"] - r["on"]), "j1": int(r["j1"] - r["on"]), "jt": int(r["jt"] - r["on"]), "top": round(float(r["top"]), 2)})
OUT["drill"] = drill

# ---------------- board
S3 = pd.read_parquet(f"{ROOT}/data/outcomes/79_scored_coverage_reps.parquet")
G = GR[GR.nfl_position.isin(["CB", "SS", "FS", "DB"])].join(base[["forty"]])
agg = S3.groupby("nfl_id_d").agg(reps=("dp", "size"), targets=("targeted", "sum"), comp=("completion", "mean"), open=("open", "mean"), xcomp=("xcomp_rep", "mean"))
G = G.join(agg)
OUT["board"] = [{"id": int(i), "name": r.display_name, "pos": r.nfl_position, "cls": int(r.draft_year), "dp": round(r.drive_plus, 1), "tier": r.tier,
                 "forty": None if pd.isna(r.forty) else float(r.forty), "speed": round(r.cb_drive_speed, 2),
                 "reps": None if pd.isna(r.reps) else int(r.reps), "targets": None if pd.isna(r.targets) else int(r.targets),
                 "comp": None if pd.isna(r.comp) else round(r.comp, 3), "open": None if pd.isna(r.open) else round(r.open, 3),
                 "xcomp": None if pd.isna(r.xcomp) else round(r.xcomp, 3)} for i, r in G.iterrows()]

# ---------------- clock + chase (3-yd reps, all graded DBs)
R, arr, vv, jb, T, dpl = ns["nfl_reps"](); terciles = ns["terciles"]
OFF = np.arange(-5, 16); n = len(R); r_ = np.arange(n)
cols = np.clip(jb[:, None] + OFF[None, :], 0, T - 1); okw = vv[r_[:, None], cols] & ((jb[:, None] + OFF[None, :]) < T)
dist = np.hypot(arr["x_rel"] - arr["dx"], arr["y_field"] - arr["dy"])
Sm = np.where(okw, dist[r_[:, None], cols], np.nan); Sm = Sm - Sm[:, [2]]
clock = {"t": r2(OFF / 10)}
for cell in ["zone-matching", "man-matching"]:
    m = (R.cell == cell).values; terc, q = terciles(R, R.cell == cell, dpl); tv = terc.values; rec = {"gap": round(float((R.t_throw - R.t_break)[m].median()), 2), "reps": int(m.sum())}
    for tr in ["low", "mid", "high"]:
        mm = m & (tv == tr); ids = R[mm].nfl_id_d.unique(); mu = np.nanmean(Sm[mm], 0); bs = []
        for _ in range(300):
            pk = rng.choice(ids, len(ids)); idx = np.concatenate([np.where(mm & (R.nfl_id_d.values == i))[0] for i in pk]); bs.append(np.nanmean(Sm[idx], 0))
        lo, hi = np.nanpercentile(bs, [5, 95], 0)
        rec[tr] = {"mean": r2(mu), "lo": r2(lo), "hi": r2(hi), "dbs": int(len(ids)), "reps": int(mm.sum()),
                   "cut": [round(float(100 + 10 * dpl.reindex(q.index[q == tr]).min()), 1), round(float(100 + 10 * dpl.reindex(q.index[q == tr]).max()), 1)]}
    clock[cell] = rec
OUT["clock"] = clock
j0, j1 = jb, jb + 10; ok = (j1 < T) & vv[r_, np.clip(j1, 0, T - 1)] & vv[r_, j0]
def rel(j):
    j = np.clip(j, 0, T - 1); rx, ry, dx, dy = arr["x_rel"][r_, j], arr["y_field"][r_, j], arr["dx"][r_, j], arr["dy"][r_, j]
    vx, vy = arr["vx"][r_, j], arr["vy"][r_, j]; nv = np.hypot(vx, vy) + 1e-6; ux, uy = vx / nv, vy / nv; px, py = dx - rx, dy - ry
    return px * ux + py * uy, -px * uy + py * ux
f0, l0 = rel(j0); f1, l1 = rel(j1); l1 = l1 * np.where(l0 < 0, -1, 1); sep1 = np.hypot(f1, l1)
chase = {}
for cell in ["zone-matching", "man-matching"]:
    m = (R.cell == cell).values & ok; terc, q = terciles(R, R.cell == cell, dpl); tv = terc.values; rec = {}
    for tr in ["low", "high"]:
        sel = m & (tv == tr)
        rec[tr] = {"f": r2(f1[sel]), "l": r2(l1[sel]), "open": round(float(np.mean(sep1[sel] >= 2)), 3), "median": round(float(np.median(sep1[sel])), 2),
                   "reps": int(sel.sum()), "dbs": int(R[sel].nfl_id_d.nunique())}
    chase[cell] = rec
OUT["chase"] = chase

# ---------------- throw map (zone)
src = open(f"{ROOT}/code/77b_depth_check.py").read(); ns2 = {}; exec(src.split("class FWL2")[0], ns2); d = ns2["d"]
cbt = d.drop_duplicates("nfl_id").set_index("nfl_id").dp; q = pd.qcut(cbt, 3, labels=["low", "mid", "high"]); d["terc"] = d.nfl_id.map(q)
z = d[(d.team_coverage_man_zone == "ZONE_COVERAGE") & d.rel.isin(["toward", "away"])].copy()
z["band"] = pd.cut(z.air, [-99, 10, 20, 99], labels=["short", "intermediate", "deep"], right=False)
t = z.groupby(["terc", "rel", "band"]).agg(n=("completion", "size"), comp=("completion", "mean")).reset_index()
t["share"] = t.n / t.groupby(["terc", "rel"]).n.transform("sum")
OUT["throwmap"] = {"rows": [{"terc": r.terc, "rel": r.rel, "band": r.band, "n": int(r.n), "comp": round(r.comp, 3), "share": round(r.share, 3)} for r in t.itertuples() if r.terc != "mid"],
                   "cbs": {k: int((q == k).sum()) for k in ["low", "high"]}, "throws": int(len(z[z.terc.isin(["low", "high"])])),
                   "cut": {"low": round(float(100 + 10 * cbt[q == "low"].max()), 1), "high": round(float(100 + 10 * cbt[q == "high"].min()), 1)}}

# ---------------- curves (3-yd scored reps) + decomposition
S3["drive_plus"] = 100 + 10 * S3.dp
grid = np.linspace(80, 135, 34)
def fit_curve(dd, y, logit, n_boot=300):
    dd = dd.dropna(subset=[y, "dp"]); f = f"{y} ~ dp"
    mk = (lambda x: smf.logit(f, data=x).fit(disp=0)) if logit else (lambda x: smf.ols(f, data=x).fit())
    g = pd.DataFrame({"dp": (grid - 100) / 10}); mu = mk(dd).predict(g).values; ids = dd.nfl_id_d.unique(); by = {i: dd[dd.nfl_id_d == i] for i in ids}; bs = []
    for _ in range(n_boot):
        try:
            bs.append(mk(pd.concat([by[i] for i in rng.choice(ids, len(ids))])).predict(g).values)
        except Exception:
            pass
    lo, hi = np.percentile(bs, [5, 95], 0); return r2(mu), r2(lo), r2(hi)
M = pd.read_csv(f"{ROOT}/results/79_metrics_coverage_reps.csv", index_col=0)
curves = {"grid": r2(grid)}
for key, dd, y, logit, mk_, min_n in [("open", S3.dropna(subset=["open"]), "open", True, "open rate (sep >= 2 yd)", 10),
                                      ("xcomp", S3.dropna(subset=["xcomp_rep"]), "xcomp_rep", False, "xComp allowed per rep", 10),
                                      ("comp", S3[S3.targeted == 1].dropna(subset=["completion"]), "completion", True, "completion allowed (targets)", 5)]:
    mu, lo, hi = fit_curve(dd, y, logit)
    pdb = dd.groupby("nfl_id_d").agg(v=(y, "mean"), n=(y, "size"), dpl=("drive_plus", "first"), s=("is_s", "first"))
    pdb = pdb[pdb.n >= min_n]
    curves[key] = {"mu": mu, "lo": lo, "hi": hi, "slope": round(float(M.loc[mk_, "Drive+ slope per 10 pts"]), 4), "p": round(float(M.loc[mk_, "perm p"]), 3),
                   "n": int(len(dd)), "dbs": int(dd.nfl_id_d.nunique()), "min_n": min_n,
                   "dots": [{"id": int(i), "dp": round(r.dpl, 1), "v": round(r.v, 3), "n": int(r.n)} for i, r in pdb.iterrows()]}
OUT["curves"] = curves
tt = S3[S3.targeted == 1].dropna(subset=["completion", "xcomp_rep", "cpoe_t"]).copy()
f = "{} ~ dp + C(team_coverage_man_zone) + C(route_ran) + is_s"
slopes = lambda x: {y: smf.ols(f.format(y), data=x).fit().params["dp"] for y in ["completion", "xcomp_rep", "cpoe_t"]}
est = slopes(tt); ids = tt.nfl_id_d.unique(); by = {i: tt[tt.nfl_id_d == i] for i in ids}
B = pd.DataFrame([slopes(pd.concat([by[i] for i in rng.choice(ids, len(ids))])) for _ in range(1000)])
OUT["decomp"] = {k: {"est": round(200 * est[k], 1), "lo": round(200 * float(np.percentile(B[k], 5)), 1), "hi": round(200 * float(np.percentile(B[k], 95)), 1)}
                 for k in ["completion", "xcomp_rep", "cpoe_t"]}
OUT["decomp"]["targets"] = int(len(tt)); OUT["decomp"]["dbs"] = int(len(ids))

# ---------------- spec curve
S = pd.read_csv(f"{ROOT}/results/81c_spec_curve.csv")
OUT["spec"] = [{"claim": r.claim, "b": round(r.b * (100 if r.claim.startswith("ball plays") else 1), 4), "se": round(r.se * (100 if r.claim.startswith("ball plays") else 1), 4),
                "p": round(r.p, 4), "sig": bool(r.sig), "right": bool(r.right_sign)} for r in S.itertuples()]
OUT["stats"] = {"throwside_zone_completion_pp": -2.05, "throwside_zone_p": 0.026, "throwside_rows": 22312, "throwside_cbs": 57,
                "short_share_pp": -2.2, "short_share_p": 0.012, "air_yds": 0.48, "air_p": 0.014, "r_forty": -0.33,
                "seasons": {"2024": {"comp": -2.5, "comp_p": 0.18, "open": -1.7, "open_p": 0.20, "close": 0.05, "close_p": 0.42, "dbs": 39},
                            "2025": {"comp": -12.6, "comp_p": 0.001, "open": -2.9, "open_p": 0.004, "close": 0.22, "close_p": 0.02, "dbs": 59},
                            "rookie": {"comp": -9.0, "comp_p": 0.001, "open": -2.1, "open_p": 0.08, "close": 0.17, "close_p": 0.03, "dbs": 57}},
                "credits_zone_side": {"pbu": 0.17, "pbu_se": 0.28, "pbu_expected": 0.93, "yds": -0.18, "yds_p": 0.13, "throws": 5935},
                "prereg_sha": "f1a2847b248efce018aaeb884cf0e8ffdde2aadee2dcb098b48a2ced66f55bc6"}
js = json.dumps(OUT, separators=(",", ":"))
open(f"{ROOT}/results/viz/drive_plus_viz.json", "w").write(js)
print("bytes", len(js), {k: len(json.dumps(v)) for k, v in OUT.items()})
