"""Page data on the registered wide sample (addenda e/f): composite criterion, zone-match framing, splits table.
Merges into the existing viz JSON (drill, board, throwmap, decomp, spec, film unchanged) -> results/viz/drive_plus_viz_wide.json
"""
import json, re, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; rng = np.random.default_rng(239)
D = json.load(open(f"{ROOT}/results/viz/drive_plus_viz.json"))
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet").reset_index(drop=True)
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other"); W["drive_plus"] = 100 + 10 * W.dp
for y, sgn in {"sep_gain_break": -1, "db_closing": +1, "sep_at_throw": -1, "open": -1}.items():
    d = W.dropna(subset=[y, "wr_skill"]); W.loc[d.index, y + "_r"] = smf.ols(f"{y} ~ wr_skill + C(zone) + C(route_ran) + is_s", data=d).fit().resid
    W[y + "_z"] = sgn * (W[y + "_r"] - W[y + "_r"].mean()) / W[y + "_r"].std()
W["composite"] = W[[c for c in W.columns if c.endswith("_z")]].mean(axis=1, skipna=True)
r2 = lambda a: [None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else round(float(v), 3) for v in a]
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = W.idx_r.values, W.idx_d.values
arr = {k: g(A, k) for k in ["x_rel", "y_field", "vx", "vy", "s", "a_lat", "post_pass"]}; arr.update({"dx": g(B, "x_rel"), "dy": g(B, "y_field")})
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE
ls = arr["a_lat"].copy(); ls[:, 1:-1] = (arr["a_lat"][:, :-2] + arr["a_lat"][:, 1:-1] + arr["a_lat"][:, 2:]) / 3
jb = np.where(vv & (k >= 5) & (k <= 30) & (arr["s"] > 2) & (arr["post_pass"] < .5), np.abs(ls), -1).argmax(1)
dpl = W.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp


def terciles(mask):
    ids = W[mask].nfl_id_d.unique(); q = pd.qcut(dpl.reindex(ids), 3, labels=["low", "mid", "high"]); return W.nfl_id_d.map(q), q


# ---------------- clock
OFF = np.arange(-5, 16); n = len(W); r_ = np.arange(n)
cols = np.clip(jb[:, None] + OFF[None, :], 0, T - 1); okw = vv[r_[:, None], cols] & ((jb[:, None] + OFF[None, :]) < T)
dist = np.hypot(arr["x_rel"] - arr["dx"], arr["y_field"] - arr["dy"])
Sm = np.where(okw, dist[r_[:, None], cols], np.nan); Sm = Sm - Sm[:, [2]]
clock = {"t": r2(OFF / 10)}
for cell in ["zone-matching", "man-matching"]:
    m = (W.cell == cell).values; terc, q = terciles(W.cell == cell); tv = terc.values; rec = {"gap": 0.6, "reps": int(m.sum())}
    for tr in ["low", "mid", "high"]:
        mm = m & (tv == tr); ids = W[mm].nfl_id_d.unique(); mu = np.nanmean(Sm[mm], 0); bs = []
        for _ in range(300):
            pk = rng.choice(ids, len(ids)); idx = np.concatenate([np.where(mm & (W.nfl_id_d.values == i))[0] for i in pk]); bs.append(np.nanmean(Sm[idx], 0))
        lo, hi = np.nanpercentile(bs, [5, 95], 0)
        rec[tr] = {"mean": r2(mu), "lo": r2(lo), "hi": r2(hi), "dbs": int(len(ids)), "reps": int(mm.sum()),
                   "cut": [round(float(100 + 10 * dpl.reindex(q.index[q == tr]).min()), 1), round(float(100 + 10 * dpl.reindex(q.index[q == tr]).max()), 1)]}
    clock[cell] = rec
D["clock"] = clock
# ---------------- chase
j0, j1 = jb, jb + 10; ok = (j1 < T) & vv[r_, np.clip(j1, 0, T - 1)] & vv[r_, j0]
def rel(j):
    j = np.clip(j, 0, T - 1); rx, ry, dx, dy = arr["x_rel"][r_, j], arr["y_field"][r_, j], arr["dx"][r_, j], arr["dy"][r_, j]
    vx, vy = arr["vx"][r_, j], arr["vy"][r_, j]; nv = np.hypot(vx, vy) + 1e-6; ux, uy = vx / nv, vy / nv; px, py = dx - rx, dy - ry
    return px * ux + py * uy, -px * uy + py * ux
f0, l0 = rel(j0); f1, l1 = rel(j1); l1 = l1 * np.where(l0 < 0, -1, 1); sep1 = np.hypot(f1, l1)
chase = {}
for cell in ["zone-matching", "man-matching"]:
    m = (W.cell == cell).values & ok; terc, q = terciles(W.cell == cell); tv = terc.values; rec = {}
    for tr in ["low", "high"]:
        sel = m & (tv == tr)
        rec[tr] = {"f": r2(f1[sel]), "l": r2(l1[sel]), "open": round(float(np.mean(sep1[sel] >= 2)), 3), "median": round(float(np.median(sep1[sel])), 2), "reps": int(sel.sum()), "dbs": int(W[sel].nfl_id_d.nunique())}
    chase[cell] = rec
D["chase"] = chase
# ---------------- curves on wide reps: open, composite, completion
grid = np.linspace(80, 135, 34)
def fit_curve(dd, y, logit, n_boot=300):
    dd = dd.dropna(subset=[y, "dp"]); f = f"{y} ~ dp"; mk = (lambda x: smf.logit(f, data=x).fit(disp=0)) if logit else (lambda x: smf.ols(f, data=x).fit())
    gg = pd.DataFrame({"dp": (grid - 100) / 10}); mu = mk(dd).predict(gg).values; ids = dd.nfl_id_d.unique(); by = {i: dd[dd.nfl_id_d == i] for i in ids}; bs = []
    for _ in range(n_boot):
        try: bs.append(mk(pd.concat([by[i] for i in rng.choice(ids, len(ids))])).predict(gg).values)
        except Exception: pass
    lo, hi = np.percentile(bs, [5, 95], 0); return r2(mu), r2(lo), r2(hi)
def slope_p(d, y, ctx, sign, n_perm=300):
    d = d.dropna(subset=[y, "dp"]).copy(); f = f"{y} ~ dp + {ctx}"; mod = smf.ols(f, data=d); used = d.loc[mod.data.row_labels]
    m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(used.nfl_id_d)[0]}); t0 = mod.fit().tvalues["dp"] * sign
    vals = used.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp; null = []
    for _ in range(n_perm):
        dd = used.copy(); dd["dp"] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index)); null.append(smf.ols(f, data=dd).fit().tvalues["dp"] * sign)
    return float(m.params["dp"]), float(m.bse["dp"]), (1 + (np.array(null) >= t0).sum()) / (n_perm + 1), len(used), used.nfl_id_d.nunique()
curves = {"grid": r2(grid)}
for key, dd, y, logit, ctx, sign, min_n in [("open", W.dropna(subset=["open"]), "open", True, "C(zone) + C(route_ran) + is_s", -1, 10),
                                          ("composite", W.dropna(subset=["composite"]), "composite", False, "is_s", +1, 10),
                                          ("comp", W[W.targeted == 1].dropna(subset=["completion"]), "completion", True, "C(zone) + C(route_ran) + is_s", -1, 5)]:
    mu, lo, hi = fit_curve(dd, y, logit); b, se, p, nn, k_ = slope_p(dd, y, ctx, sign)
    pdb = dd.groupby("nfl_id_d").agg(v=(y, "mean"), n=(y, "size"), dpl=("drive_plus", "first"), s=("is_s", "first")); pdb = pdb[pdb.n >= min_n]
    curves[key] = {"mu": mu, "lo": lo, "hi": hi, "slope": round(b, 4), "se": round(se, 4), "p": round(p, 3), "n": int(nn), "dbs": int(k_), "min_n": min_n,
                   "dots": [{"id": int(i), "dp": round(r.dpl, 1), "v": round(r.v, 3), "n": int(r.n)} for i, r in pdb.iterrows()]}
    print(key, round(b, 4), round(p, 3), nn, k_, flush=True)
D["curves"] = curves
# ---------------- KPIs: 110 vs 90 completion on the wide targets, reliability / disattenuation
W = W.sort_values(["nfl_id_d", "game_id", "play_id"]); W["half"] = W.groupby("nfl_id_d").cumcount() % 2
h = W.dropna(subset=["composite"]).groupby(["nfl_id_d", "half"]).composite.agg(["mean", "size"]).unstack(); h = h[h["size"].min(1) >= 8]
rr = np.corrcoef(h["mean"][0], h["mean"][1])[0, 1]; rel_c = 2 * rr / (1 + rr)
pm = W.groupby("nfl_id_d").agg(c=("composite", "mean"), n=("dp", "size"), dp=("dp", "first")); pm = pm[pm.n >= 10]
r_obs = pm[["c", "dp"]].corr().iloc[0, 1]
D["wide"] = {"reps": int(len(W)), "dbs": int(W.nfl_id_d.nunique()), "targets": int((W.targeted == 1).sum()), "cbs": int(W[W.is_s == 0].nfl_id_d.nunique()),
             "comp110v90": round(2 * curves["comp"]["slope"] * 100, 1), "comp110v90_lo": round(2 * 100 * (curves["comp"]["slope"] - 1.2816 * curves["comp"]["se"]), 1),
             "comp110v90_hi": round(2 * 100 * (curves["comp"]["slope"] + 1.2816 * curves["comp"]["se"]), 1),
             "rel_composite": round(rel_c, 2), "r_obs": round(r_obs, 2), "r_true": round(r_obs / np.sqrt(.66 * rel_c), 2), "players_r": int(len(pm))}
# ---------------- splits table parsed from 89
txt = open(f"{ROOT}/results/89_wide_validation.txt").read(); splits = []; sec = None
for line in txt.splitlines():
    m = re.match(r"\s+\[(.+?)\] reps (\d+), DBs (\d+), targets (\d+)", line)
    if m: sec = {"label": m.group(1), "reps": int(m.group(2)), "dbs": int(m.group(3)), "targets": int(m.group(4))}; continue
    m = re.match(r"\s+(.+?)\s{2,}([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\] p1 (\d\.\d+) · (\w+(?: \w+)?)\s+· n (\d+), DBs (\d+)", line)
    if m and sec: splits.append({"split": sec["label"], "claim": m.group(1).strip(), "b": float(m.group(2)), "lo": float(m.group(3)), "hi": float(m.group(4)),
                                 "p1": float(m.group(5)), "verdict": m.group(6).strip(), "n": int(m.group(7)), "dbs": int(m.group(8))})
D["splits"] = splits; print("splits parsed:", len(splits), "supported:", sum(s["verdict"] == "supported" for s in splits))
S = {s["split"]: s for s in []}
def pick(split, claim):
    for s in splits:
        if s["split"] == split and s["claim"].startswith(claim): return s
D["stats"]["seasons"] = {yr: {"comp": pick(f"4 yd, {yr} season", "completion")["b"], "comp_p": pick(f"4 yd, {yr} season", "completion")["p1"],
                               "open": pick(f"4 yd, {yr} season", "open")["b"], "open_p": pick(f"4 yd, {yr} season", "open")["p1"],
                               "close": pick(f"4 yd, {yr} season", "closing, zone-matching")["b"], "close_p": pick(f"4 yd, {yr} season", "closing, zone-matching")["p1"],
                               "dbs": pick(f"4 yd, {yr} season", "completion")["dbs"]} for yr in ["2024", "2025"]}
D["stats"]["ngs_side"] = {"open_pp": -2.9, "p": 0.036, "sep": -0.14, "sep_p": 0.16, "rows": 2611}
import math
def clean(o):
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, dict): return {k_: clean(v) for k_, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
D = clean(D); js = json.dumps(D, separators=(",", ":"), allow_nan=False)
open(f"{ROOT}/results/viz/drive_plus_viz_wide.json", "w").write(js); print("bytes", len(js), D["wide"])
