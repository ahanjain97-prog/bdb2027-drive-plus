"""CB drive speed: (1) mechanistic prediction, (2) controls ladder, (3) out-of-sample prediction.

(1) 72b: the drill speed that matters is reached 0.8-1.0 s after the turn, and it maps onto NFL closing 0.3-1.0 s after
    the receiver's break, not the first 0.3 s. Prediction: drive speed should matter only when the ball comes out late
    enough after the break for that phase to happen. gap = t_throw - t_break. Slope of sep_at_throw / completion /
    CPOE on drive speed by gap bin (< 0.5, 0.5-1.0, > 1.0 s) and the continuous drive x gap interaction (player perm p).
(2) Controls ladder for the frozen metric (zone-matching closing; all-rep separation gained; CPOE):
    L0 receiver skill + route (+ man/zone)    L1 + situation: cushion, press, slot, down, distance, break time, sep at break
    L2 + receiver athleticism: forty, weight, height (pool receivers; mean-imputed + flag)
    L3 + CB stopwatch + size battery: forty, 10-yd split, 3-cone, shuttle, vertical, broad, weight, height, arm (within-year
        mean-imputed + count of missing)
    L4 + CB pre-break behaviour in the rep: backpedal share, speed at the break, pre-break alignment with the receiver
    L5 + defense team + season fixed effects     L6 + draft capital (log pick, UDFA) -- sanity only
(3) Out-of-sample (rep-level MSE reduction vs the same model without the CB block): leave-one-draft-class-out and
    season-forward (2023-24 -> 2025). Blocks: frozen top; speed at 1.0 s; late-drive mean (v_06, v_08, v_10); ridge on
    the whole post-turn curve (v_out .. v_10, top, d_10) with in-fold alpha; top + CB stopwatch/size battery.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(67)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
tw = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_time_windows.parquet")
cov = cov.merge(tw[["game_id", "play_id", "nfl_id_d", "g_v00"]].drop_duplicates(["game_id", "play_id", "nfl_id_d"]), on=["game_id", "play_id", "nfl_id_d"], how="left")
Z["late"] = Z[["v_06", "v_08", "v_10"]].mean(1)
for c in ["top", "v_10", "late", "v_out", "v_02", "v_04", "v_06", "v_08", "d_10"]:
    v = Z[c].reindex(cov.nfl_id_d.unique()); cov["c_" + c] = cov.nfl_id_d.map((Z[c] - v.mean()) / v.std())
cov["gap"] = cov.t_throw - cov.t_break
cov["gap_bin"] = pd.cut(cov.gap, [-9, 0.5, 1.0, 99], labels=["<0.5 s", "0.5-1.0 s", ">1.0 s"])
# CB stopwatch/size battery (within-year mean impute + missing count)
SW = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_weight", "combine_height", "arm_length"]
cb = base.reindex(cov.nfl_id_d.unique())[SW + ["draft_year", "log_pick", "udfa"]].copy()
cb["sw_missing"] = cb[SW].isna().sum(1)
for c in SW:
    cb[c] = cb[c].fillna(cb.groupby("draft_year")[c].transform("mean")).fillna(cb[c].mean())
    cb[c] = (cb[c] - cb[c].mean()) / cb[c].std()
for c in SW + ["sw_missing", "log_pick", "udfa"]:
    cov["cb_" + c] = cov.nfl_id_d.map(cb[c])
wr = base.reindex(cov.nfl_id_r.unique())[["forty", "combine_weight", "combine_height"]].copy()
for c in wr.columns:
    cov["wr_" + c + "_miss"] = cov.nfl_id_r.map(wr[c].isna().astype(float))
    wr[c] = wr[c].fillna(wr[c].mean()); cov["wr_" + c] = cov.nfl_id_r.map((wr[c] - wr[c].mean()) / wr[c].std())
# pre-break alignment (70) and the CB's team
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = cov.idx_r.values, cov.idx_d.values
vxr, vyr, sr, lr, pr = g(A, "vx"), g(A, "vy"), g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
vxd, vyd, sd = g(B, "vx"), g(B, "vy"), g(B, "s"); vv = V[A] & V[B]
T = X.shape[1]; k = np.arange(T)[None, :] - PRE; cols = np.arange(T)[None, :]
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
jb = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1).argmax(1)
w = vv & (cols >= (jb - 5)[:, None]) & (cols < jb[:, None]) & (sd > 1.0) & (sr > 1.0)
cosv = (vxr * vxd + vyr * vyd) / (np.hypot(vxr, vyr) * np.hypot(vxd, vyd) + 1e-6)
cov["pre_align"] = np.where(w.sum(1) >= 3, np.where(w, cosv, 0).sum(1) / np.maximum(w.sum(1), 1), np.nan)
pp = pd.read_parquet(f"{ROOT}/data/raw/player_play.parquet")[["game_id", "play_id", "nfl_id", "team_abbr"]]
cov = cov.merge(pp.rename(columns={"nfl_id": "nfl_id_d", "team_abbr": "def_team"}), on=["game_id", "play_id", "nfl_id_d"], how="left")
G2 = lambda d: np.column_stack([pd.factorize(d.nfl_id_r)[0], pd.factorize(d.nfl_id_d)[0]])
pd.set_option("display.width", 240)


def est(d, term, f, n_perm=200):
    vars_ = [v for v in set(f.replace("C(", " ").replace(")", " ").replace("~", " ").replace("+", " ").replace("*", " ").split()) if v in d.columns]
    d = d.dropna(subset=vars_).copy()
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": G2(d)})
    base_t = term.split(":")[0]
    t0 = smf.ols(f, data=d).fit().tvalues[term]; vals = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[base_t]; null = []
    for _ in range(n_perm):
        dd = d.copy(); dd[base_t] = dd.nfl_id_d.map(pd.Series(rng.permutation(vals.values), index=vals.index))
        null.append(smf.ols(f, data=dd).fit().tvalues[term])
    return m.params[term], m.pvalues[term], (1 + (np.abs(null) >= abs(t0)).sum()) / (n_perm + 1), len(d), d.nfl_id_d.nunique()


# ---------------- (1)
print("=== (1) does drive speed matter only when the ball comes out late enough after the break? ===")
print("  reps with a throw by gap:", cov.gap_bin.value_counts().sort_index().to_dict())
for y, d, ctx in [("sep_at_throw", cov, "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
                  ("sep_at_throw", cov[cov.cell == "zone-matching"], "wr_skill + C(route_ran)"),
                  ("cpoe", cov[cov.targeted == 1], "rec_target_prop + C(team_coverage_man_zone)")]:
    lab = y + (" (zone-matching)" if len(d) < 1000 and y != "cpoe" else "")
    out = []
    for gb in ["<0.5 s", "0.5-1.0 s", ">1.0 s"]:
        dd = d[d.gap_bin == gb]
        try:
            b, p, pp_, n, k = est(dd, "c_top", f"{y} ~ c_top + {ctx}", n_perm=150)
            out.append(f"{gb}: {b:+.3f} (perm p {pp_:.3f}, n {n}, CBs {k})")
        except Exception as e:
            out.append(f"{gb}: n/a")
    b, p, pp_, n, k = est(d, "c_top:gap", f"{y} ~ c_top * gap + {ctx}", n_perm=300)
    print(f"  {lab:28s} " + " | ".join(out) + f" || drive x gap {b:+.3f} per s (perm p {pp_:.3f})")
tg = cov[(cov.targeted == 1)].dropna(subset=["completion", "c_top", "gap"]).copy()
m = smf.logit("completion ~ c_top * gap + C(team_coverage_man_zone)", data=tg).fit(disp=0)
print("  completion when targeted (logit): " + " | ".join(f"{t} {m.params[t]:+.3f} (p {m.pvalues[t]:.3f})" for t in ["c_top", "gap", "c_top:gap"]) + f"  n {len(tg)}")
for gq in [0.3, 0.8, 1.5]:
    pr_ = [m.predict(tg.assign(c_top=zv, gap=gq)).mean() for zv in (-1.28, 1.28)]
    print(f"    ball out {gq:.1f} s after the break: completion p10 drive {pr_[0] * 100:.0f}% -> p90 drive {pr_[1] * 100:.0f}%")

# ---------------- (2)
print("\n=== (2) controls ladder for the frozen drive metric (b per SD | clustered p | player-permutation p | reps, CBs) ===")
L = [("L0 receiver skill + route", ""),
     ("L1 + situation", " + cushion + C(press_bin) + cb_slot + C(down) + yards_to_go + t_break + sep_at_break"),
     ("L2 + receiver athleticism", " + wr_forty + wr_combine_weight + wr_combine_height + wr_forty_miss"),
     ("L3 + CB stopwatch/size battery", " + " + " + ".join("cb_" + c for c in SW) + " + cb_sw_missing"),
     ("L4 + CB pre-break behaviour", " + db_pre_backpedal + g_v00 + pre_align"),
     ("L5 + defense team + season FE", " + C(def_team) + C(season)"),
     ("L6 + draft capital (sanity)", " + cb_log_pick + cb_udfa")]
rows = []
for y, d, ctx in [("db_closing", cov[cov.cell == "zone-matching"], "wr_skill + C(route_ran)"),
                  ("sep_gain_break", cov, "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
                  ("cpoe", cov[cov.targeted == 1], "rec_target_prop + C(team_coverage_man_zone)")]:
    acc = ""
    for name, add in L:
        acc += add
        if y == "cpoe" and name.startswith("L5"):
            add_ = " + C(season)"; acc = acc.replace(" + C(def_team) + C(season)", add_)    # 284 targets: team FE too thin
        b, p, pp_, n, k = est(d, "c_top", f"{y} ~ c_top + {ctx}{acc}")
        rows.append({"outcome": y, "rung": name, "b": b, "p_clustered": p, "p_perm": pp_, "reps": n, "CBs": k})
        print(f"  {y:14s} {name:32s} {b:+.4f} | p {p:.3f} | perm {pp_:.3f} | {n}, {k}", flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/72c_controls_ladder.csv", index=False)

# ---------------- (3)
print("\n=== (3) out-of-sample: rep-level MSE reduction (%) vs the context-only model ===")


def oos(d, y, ctx, block, splits):
    d = d.dropna(subset=[y] + (["wr_skill"] if "wr_skill" in ctx else [])).copy()
    gain = tot = 0.0
    for tr, te in splits(d):
        a, b = d[tr].copy(), d[te].copy()
        if "route_ran" in ctx:
            b = b[b.route_ran.isin(a.route_ran.unique())]
        m0 = smf.ols(f"{y} ~ {ctx}", data=a).fit(); a["res"] = m0.resid; b["res"] = b[y] - m0.predict(b)
        P_ = cb_block(block, a, b)
        if P_ is None:
            continue
        pa, pb = P_
        a["xb"], b["xb"] = pa, pb
        a, b = a.dropna(subset=["xb"]), b.dropna(subset=["xb"])
        s = smf.ols("res ~ xb", data=a).fit()
        gain += ((b.res - b.res.mean() * 0) ** 2).sum() - ((b.res - s.predict(b)) ** 2).sum(); tot += (b.res ** 2).sum()
    return 100 * gain / tot


def cb_block(block, a, b):
    if block in ("c_top", "c_v_10", "c_late"):
        return a[block].values, b[block].values
    cols = {"curve ridge": ["c_v_out", "c_v_02", "c_v_04", "c_v_06", "c_v_08", "c_v_10", "c_top", "c_d_10"],
            "top + stopwatch/size": ["c_top"] + ["cb_" + c for c in SW]}[block]
    Pa = a.groupby("nfl_id_d").agg(res=("res", "mean"), n=("res", "size")).join(a.groupby("nfl_id_d")[cols].first()).dropna()
    if len(Pa) < 10:
        return None
    r = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(Pa[cols].values, Pa.res.values, sample_weight=Pa.n.values)
    fa = pd.Series(r.predict(a[cols].fillna(0).values), index=a.index).where(a[cols].notna().all(1))
    fb = pd.Series(r.predict(b[cols].fillna(0).values), index=b.index).where(b[cols].notna().all(1))
    return fa.values, fb.values


LOCO = lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())]
SFWD = lambda d: [((d.season <= 2024).values, (d.season == 2025).values)]
res = []
for y, d, ctx in [("db_closing", cov[cov.cell == "zone-matching"], "wr_skill + C(route_ran)"),
                  ("sep_gain_break", cov, "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
                  ("sep_at_throw", cov, "wr_skill + C(team_coverage_man_zone) + C(route_ran)"),
                  ("cpoe", cov[cov.targeted == 1], "rec_target_prop + C(team_coverage_man_zone)")]:
    d = d.dropna(subset=["c_top"])
    for blk in ["c_top", "c_v_10", "c_late", "curve ridge", "top + stopwatch/size"]:
        r0 = {"outcome": y + (" (zone-matching)" if y == "db_closing" else ""), "block": blk.replace("c_", "")}
        for sn, sp in (("class-out", LOCO), ("season-fwd", SFWD)):
            r0[sn] = oos(d, y, ctx, blk, sp)
        # permutation null for class-out (shuffle CB identity on the combine side)
        null = []
        ids = d.nfl_id_d.unique()
        cols_all = [c for c in d.columns if c.startswith("c_") or c.startswith("cb_")]
        per = d.drop_duplicates("nfl_id_d").set_index("nfl_id_d")[cols_all]
        for _ in range(60):
            mp = dict(zip(ids, rng.permutation(ids)))
            dd = d.drop(columns=cols_all).join(per.rename(index=mp), on="nfl_id_d")
            null.append(oos(dd, y, ctx, blk, LOCO))
        r0["class-out perm p"] = (1 + (np.array(null) >= r0["class-out"]).sum()) / 61
        res.append(r0); print("  ", {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r0.items()}, flush=True)
pd.DataFrame(res).to_csv(f"{ROOT}/results/72c_oos.csv", index=False)
