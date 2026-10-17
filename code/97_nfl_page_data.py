"""Data for the NFL-audience page: the wide viz JSON plus a route tree (wide reps), a mph conversion and staff-use numbers.
Output: results/viz/drive_plus_viz_nfl.json
"""
import json, numpy as np, pandas as pd
ROOT = "/Users/ahanjain/bdb_ssl"
D = json.load(open(f"{ROOT}/results/viz/drive_plus_viz_wide.json"))

# route tree on the wide reps: median separation change, bottom third minus top third, by coverage (same rule as code/96)
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other")
dpl = W.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp
tree = {}
for cell in ["zone-matching", "man-matching"]:
    d = W[W.cell == cell]; ids = d.nfl_id_d.unique(); q = pd.qcut(dpl.reindex(ids), 3, labels=["low", "mid", "high"]); d = d.assign(terc=d.nfl_id_d.map(q))
    rows = []
    for rt in ["GO", "POST", "CORNER", "IN", "OUT", "CROSS", "SLANT", "HITCH"]:
        g = d[d.route_ran == rt]; lo, hi = g[g.terc == "low"].sep_gain_break.dropna(), g[g.terc == "high"].sep_gain_break.dropna()
        if len(lo) < 8 or len(hi) < 8: continue
        rows.append({"route": rt, "gap": round(float(lo.median() - hi.median()), 3), "low": round(float(lo.median()), 3), "high": round(float(hi.median()), 3),
                     "n_low": int(len(lo)), "n_high": int(len(hi))})
    tree[cell] = rows
D["routeTree"] = tree

# yd/s -> mph and an approximate Drive+ -> raw speed line among corners (grades are scaled within class, so this is a guide)
B = pd.DataFrame(D["board"]); cb = B[(B.pos == "CB") & B.speed.notna()]
k, c0 = np.polyfit(cb.dp, cb.speed, 1)
D["mph"] = {"per_yds": 2.04545, "slope": float(k), "icept": float(c0), "n": int(len(cb)), "raw_min": float(cb.speed.min()), "raw_max": float(cb.speed.max())}

# staff-use numbers (from code/90 and the reliability analysis; Spearman-Brown from one attempt)
r1 = .49
D["staff"] = {"rel": [{"att": n, "r": round(n * r1 / (1 + (n - 1) * r1), 2)} for n in (1, 2, 3, 4)],
              "composite_by_cov": {"zone-match": .094, "all reps": .084, "man": .017}}
# which moment of the drive predicts Sundays: per-moment speed -> zone-match closing (code/72b, registered 3-yd reps), with reliability (72a)
comp = pd.read_csv(f"{ROOT}/results/72b_components.csv"); rel = pd.read_csv(f"{ROOT}/results/72a_reliability.csv", index_col=0)
zc = comp[(comp.reps == "zone-matching") & (comp.outcome == "db_closing")].set_index("component")
D["staff"]["steps"] = [{"t": t, "b": round(float(zc.loc[k, "b"]), 3), "p": round(float(zc.loc[k, "p_perm"]), 3), "rel": float(rel.loc[k, "reliability_of_mean"]),
                        "n": int(zc.loc[k, "n"]), "cbs": int(zc.loc[k, "CBs"])} for t, k in [(0, "v_out"), (.2, "v_02"), (.4, "v_04"), (.6, "v_06"), (.8, "v_08"), (1.0, "v_10")]]
print("steps:", D["staff"]["steps"])
json.dump(D, open(f"{ROOT}/results/viz/drive_plus_viz_nfl.json", "w"), allow_nan=True)
print("route tree:", {c: [(r["route"], r["gap"]) for r in v] for c, v in tree.items()})
print(f"mph line: speed = {c0:.3f} + {k:.4f} * Drive+  (n {len(cb)} CBs); at 90/100/110/120: " + ", ".join(f"{(c0 + k * x) * 2.04545:.1f}" for x in (90, 100, 110, 120)) + " mph")
print("reliability:", D["staff"]["rel"])
