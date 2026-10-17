"""Football-specific visuals for Drive+ (figures 1-5; 6-7 in 80b). Output results/figs/*.png

1 combine drill trace: best vs lowest Drive+ CB, overhead on a field, colored by speed + speed curves of every CB attempt
2 the chase: average receiver and DB paths around the receiver's break (receiver frame), high vs low Drive+ third
3 separation clock: separation change around the break by Drive+ third, with football annotations
4 throw map: where QBs throw at low vs high Drive+ CBs (zone) and how often it is caught, by depth and side
5 Drive+ is not the 40 + the Drive+ big board
"""
import json, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; OUTD = f"{ROOT}/results/figs"
import os; os.makedirs(OUTD, exist_ok=True)
rng = np.random.default_rng(163)
HI, LO, MID, GOLD = "#1565c0", "#c62828", "#9e9e9e", "#f9a825"
TURF, TURF2, CHALK = "#2f6b33", "#357a39", "#f4f4f0"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False})
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
GR = pd.read_csv(f"{ROOT}/prereg/drive_plus_frozen_2026-10-12.csv", index_col=0)
FIG = sys.argv[1:] or ["1", "2", "3", "4", "5"]


def turf(ax, x0, x1, y0, y1, step=5, label_every=None, hash_marks=True):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=TURF, zorder=0))
    k = 0
    for y in np.arange(np.floor(y0 / step) * step, y1 + .01, step):
        if y0 <= y <= y1:
            if k % 2 == 0 and y + step <= y1:
                ax.add_patch(Rectangle((x0, y), x1 - x0, step, color=TURF2, zorder=0))
            ax.axhline(y, color=CHALK, lw=1.0, alpha=.55, zorder=1)
            k += 1
    if hash_marks:
        for y in np.arange(np.ceil(y0), y1, 1):
            for xx in (x0 + (x1 - x0) * .33, x0 + (x1 - x0) * .67):
                ax.plot([xx - .25, xx + .25], [y, y], color=CHALK, lw=.8, alpha=.45, zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def footer(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color="#666", ha="left", va="bottom")


# ============================================================ FIGURE 1: combine drill trace
def drill_attempts():
    ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
    d = ct[(ct.entity_type == "PLAYER") & (ct.drill_name == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")].copy()
    d["time"] = pd.to_datetime(d.time); d = d.sort_values(["event_id", "time"]); out = []
    for eid, g in d.groupby("event_id"):
        t = (g.time - g.time.iat[0]).dt.total_seconds().values; x, y = g.x.values.astype(float), g.y.values.astype(float)
        if len(t) < 12:
            continue
        dt = np.median(np.diff(t)); vx = savgol_filter(x, 7, 2, deriv=1, delta=dt); vy = savgol_filter(y, 7, 2, deriv=1, delta=dt)
        s = np.hypot(vx, vy); hd = np.degrees(np.unwrap(np.arctan2(vx, vy)))
        on = int(np.argmax(s >= 1.0)); run = np.where((s >= 2.0) & (np.arange(len(s)) > on))[0]
        if len(run) < 4:
            continue
        line = np.median(hd[run[:4]]); dev = np.abs(hd - line)
        st = np.where((np.arange(len(s)) > run[3]) & (s >= 3.0) & (dev > 10))[0]
        if not len(st):
            continue
        j0 = st[0]; win = slice(j0, min(j0 + 15, len(s))); dmax = dev[win].max()
        if dmax < 25 or dmax > 90:
            continue
        j1 = j0 + int(np.argmax(dev[win] >= 0.9 * dmax))
        k, jend, top, kt = j1, min(j1 + 20, len(s) - 1), s[j1], j1
        while k < jend:
            k += 1
            if s[k] > top:
                top, kt = s[k], k
            elif s[k] < top - 0.5:
                break
        th = np.radians(line); px, py = x - x[on], y - y[on]           # rotate: backpedal line -> +y
        rx = px * np.cos(th) - py * np.sin(th); ry = px * np.sin(th) + py * np.cos(th)
        if rx[min(j1 + 5, len(rx) - 1)] < 0:
            rx = -rx
        out.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "t": t - t[j1], "x": rx, "y": ry, "s": s, "on": on, "j0": j0, "j1": j1, "jt": kt, "top": top})
    return out


def fig1():
    att = drill_attempts()
    best = GR[GR.nfl_position == "CB"].index[0]; worst = GR[GR.nfl_position == "CB"].index[-1]
    pick = {}
    for pid in (best, worst):
        a = [r for r in att if r["nfl_id"] == pid]; mean_top = np.mean([r["top"] for r in a])
        pick[pid] = min(a, key=lambda r: abs(r["top"] - mean_top))
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1, 1, 1.25], hspace=.32, wspace=.12)
    norm = plt.Normalize(0, 10.5); cmap = plt.get_cmap("inferno")
    for c, (pid, col) in enumerate([(best, HI), (worst, LO)]):
        r = pick[pid]; ax = fig.add_subplot(gs[0, c])
        sl = slice(r["on"], min(r["jt"] + 8, len(r["x"])))
        xs, ys, ss = r["x"][sl], r["y"][sl], r["s"][sl]
        turf(ax, -7.5, 10.5, -1.5, max(ys.max() + 2, 16), step=5)
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2); seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=norm, linewidth=6, capstyle="round", zorder=3); lc.set_array(ss[:-1]); ax.add_collection(lc)
        for j, mk, lab, side in [(r["on"], "o", "start", -1), (r["j0"], "^", "turn starts", 1), (r["j1"], "D", "turn ends", -1)]:
            ax.scatter(r["x"][j], r["y"][j], s=70, marker=mk, color="white", edgecolor="k", zorder=5)
            ax.annotate(lab, (r["x"][j], r["y"][j]), xytext=(9 * side, 0), textcoords="offset points", ha="left" if side > 0 else "right", va="center", fontsize=9, color="white", fontweight="bold")
        ax.scatter(r["x"][r["jt"]], r["y"][r["jt"]], s=260, marker="*", color=GOLD, edgecolor="k", zorder=6)
        ax.annotate(f"drive top\n{r['top']:.1f} yd/s", (r["x"][r["jt"]], r["y"][r["jt"]]), xytext=(-12, 2), textcoords="offset points", ha="right", va="center", fontsize=9.5, color=GOLD, fontweight="bold")
        g = GR.loc[pid]
        ax.set_title(f"{g.display_name} ({int(g.draft_year)})\nDrive+ {g.drive_plus:.0f}", color=col, fontsize=12.5)
    cax = fig.add_subplot(gs[0, 2]); cax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cbx = cax.inset_axes([0.12, 0.10, 0.8, 0.05]); cb = fig.colorbar(sm, cax=cbx, orientation="horizontal")
    cb.set_label("speed on the path (yd/s)")
    cax.text(.12, .78, "The drill", fontsize=13, fontweight="bold", transform=cax.transAxes)
    cax.text(.12, .30, "Backpedal-and-transition (45°).\nThe DB backpedals, breaks on\nthe coach's cue and drives.\n\nDrive+ = how fast he is moving\nonce he comes out of the turn,\nscaled within his combine class:\n100 = average 2023-25 CB,\n10 points = 1 SD.",
             fontsize=9.5, transform=cax.transAxes, color="#333", linespacing=1.35)
    ax = fig.add_subplot(gs[1, :])
    cbs = set(GR[GR.nfl_position == "CB"].index)
    for r in att:
        if r["nfl_id"] in cbs and r["nfl_id"] not in (best, worst):
            m = (r["t"] >= -1.0) & (r["t"] <= 1.6); ax.plot(r["t"][m], r["s"][m], color="#bbb", lw=.7, alpha=.5, zorder=1)
    for pid, col in [(best, HI), (worst, LO)]:
        r = pick[pid]; m = (r["t"] >= -1.0) & (r["t"] <= 1.6)
        ax.plot(r["t"][m], r["s"][m], color=col, lw=3, zorder=3, label=f"{GR.loc[pid].display_name} (Drive+ {GR.loc[pid].drive_plus:.0f})")
    ax.axvspan(0, .4, color="#e0e0e0", alpha=.6, zorder=0); ax.axvspan(.8, 1.0, color=GOLD, alpha=.35, zorder=0)
    ax.axvline(0, color="k", lw=.8, ls=":")
    ax.text(.2, 2.6, "first 2 steps\n(no signal)", ha="center", fontsize=9, color="#555")
    ax.text(.9, 2.6, "steps 3-5\nDrive+ lives here", ha="center", fontsize=9, color="#7a5c00", fontweight="bold")
    ax.text(-.5, 9.4, "backpedal", ha="center", fontsize=9, color="#555"); ax.text(-.12, 9.4, "turn", ha="center", fontsize=9, color="#555")
    ax.set_xlim(-1.0, 1.6); ax.set_ylim(2, 10.8); ax.set_xlabel("seconds from the end of the turn"); ax.set_ylabel("speed (yd/s)")
    ax.set_title("Every CB attempt (gray): the speed that separates corners is reached 0.8-1.0 s after the turn, not in the first steps", fontsize=11, loc="left")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("Drive+: how fast a corner gets going out of the break", fontsize=16, fontweight="bold", x=.01, ha="left", y=.995)
    footer(fig, "NFL Combine tracking (10 Hz), BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION. Paths rotated so the backpedal runs up the page; shown: the attempt closest to each player's average.")
    fig.savefig(f"{OUTD}/fig1_combine_drill.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============================================================ shared NFL rep arrays (3-yd reps, all graded DBs)
def nfl_reps():
    src = open(f"{ROOT}/code/78_grade_card_all_db.py").read(); ns = {}
    exec(src.split('pd.set_option("display.width", 250)')[0], ns)
    R = ns["build"]("coverage_reps"); X, V, F, PRE = ns["X"], ns["V"], ns["F"], ns["PRE"]
    g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
    A, B = R.idx_r.values, R.idx_d.values
    arr = {k: g(A, k) for k in ["x_rel", "y_field", "vx", "vy", "s", "a_lat", "post_pass"]}
    arr.update({"dx": g(B, "x_rel"), "dy": g(B, "y_field")})
    vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE
    ls = arr["a_lat"].copy(); ls[:, 1:-1] = (arr["a_lat"][:, :-2] + arr["a_lat"][:, 1:-1] + arr["a_lat"][:, 2:]) / 3
    jb = np.where(vv & (k >= 5) & (k <= 30) & (arr["s"] > 2) & (arr["post_pass"] < .5), np.abs(ls), -1).argmax(1)
    dpl = R.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp
    return R, arr, vv, jb, T, dpl


def terciles(R, mask, dpl):
    ids = R[mask].nfl_id_d.unique(); q = pd.qcut(dpl.reindex(ids), 3, labels=["low", "mid", "high"])
    return R.nfl_id_d.map(q), q


# ============================================================ FIGURE 2: the chase
def fig2(R, arr, vv, jb, T, dpl):
    from scipy.stats import gaussian_kde
    n = len(R); r_ = np.arange(n); j0, j1 = jb, jb + 10
    ok = (j1 < T) & vv[r_, np.clip(j1, 0, T - 1)] & vv[r_, j0]
    def rel(j):
        j = np.clip(j, 0, T - 1); rx, ry, dx, dy = arr["x_rel"][r_, j], arr["y_field"][r_, j], arr["dx"][r_, j], arr["dy"][r_, j]
        vx, vy = arr["vx"][r_, j], arr["vy"][r_, j]; nv = np.hypot(vx, vy) + 1e-6; ux, uy = vx / nv, vy / nv
        px, py = dx - rx, dy - ry
        return px * ux + py * uy, -px * uy + py * ux
    f0, l0 = rel(j0); f1, l1 = rel(j1)
    l1 = l1 * np.where(l0 < 0, -1, 1)                                             # mirror: DB starts on the receiver's left
    sep1 = np.hypot(f1, l1)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    m = (R.cell == "zone-matching").values & ok; terc, q = terciles(R, R.cell == "zone-matching", dpl); tv = terc.values
    mm_ = (R.cell == "man-matching").values & ok; terc_m, _ = terciles(R, R.cell == "man-matching", dpl)
    man_open = {t_: np.mean(sep1[mm_ & (terc_m.values == t_)] >= 2) for t_ in ["low", "high"]}
    for ax, tr, col in [(axes[0], "low", LO), (axes[1], "high", HI)]:
        sel = m & (tv == tr); fx, ly, sp = f1[sel], l1[sel], sep1[sel]
        turf(ax, -4.6, 2.6, -2.2, 4.2, step=1, hash_marks=False)
        try:
            kde = gaussian_kde(np.vstack([fx, ly])); gx, gy = np.mgrid[-4.6:2.6:120j, -2.2:4.2:110j]
            zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contourf(gx, gy, zz, levels=[zz.max() * .45, zz.max() * 1.01], colors=[col], alpha=.28, zorder=2)
            ax.contour(gx, gy, zz, levels=[zz.max() * .2, zz.max() * .45], colors=["white"], linewidths=[.6, 1.0], alpha=.7, zorder=2)
        except Exception:
            pass
        ax.scatter(fx, ly, s=16, color=col, alpha=.55, edgecolor="white", lw=.3, zorder=3)
        for rad, lab, ang in [(1, "1 yd", 50), (2, "2 yd: OPEN beyond", -48), (3, "3 yd", 60)]:
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, color="white", lw=2.2 if rad == 2 else .8, ls="-" if rad == 2 else "--", alpha=.95 if rad == 2 else .6, zorder=4))
            ax.text(rad * np.cos(np.radians(ang)) + .05, rad * np.sin(np.radians(ang)), lab, color="white", fontsize=9.5 if rad == 2 else 8.5, fontweight="bold" if rad == 2 else None, zorder=5)
        ax.scatter([0], [0], s=520, color="white", edgecolor="k", zorder=6); ax.text(0, 0, "WR", ha="center", va="center", fontsize=10, fontweight="bold", zorder=7)
        ax.annotate("", xy=(2.45, 0), xytext=(.38, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=2.4), zorder=5)
        ax.text(1.45, -.32, "receiver's direction", color="white", fontsize=9, ha="center")
        ax.text(-2.6, -.32, "← trailing", color="white", fontsize=9, ha="center")
        op = np.mean(sp >= 2)
        ax.text(-4.4, 3.85, f"receiver open (2+ yd): {100 * op:.0f}% of reps", color="white", fontsize=13, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=.35", fc=col, ec="white", lw=1.2))
        ax.text(-4.4, 3.05, f"median separation {np.median(sp):.1f} yd  ·  {sel.sum()} reps", color="white", fontsize=10, va="top")
        ax.set_title(f"{'Bottom' if tr == 'low' else 'Top'} third Drive+ ({(q == tr).sum()} DBs)", color=col, fontsize=13.5)
    fig.suptitle("One second after the break: where the DB is", fontsize=16.5, fontweight="bold", x=.01, ha="left")
    fig.text(.01, .905, "Zone coverage, DB matching the receiver. The receiver is fixed at the centre running right; each dot is the DB on one rep, 1.0 s after the receiver's break "
             f"(reps mirrored so the DB started on the receiver's left). In man coverage the gap is small: open on {100 * man_open['low']:.0f}% vs {100 * man_open['high']:.0f}% of reps.",
             fontsize=9.8, color="#444", wrap=True)
    footer(fig, "2023-25 NFL tracking, coverage reps with both players in the BDB pool (≤ 3 yd for ≥ 1 s), all graded DBs; Drive+ thirds among DBs with zone-matching reps.")
    fig.savefig(f"{OUTD}/fig2_the_chase.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============================================================ FIGURE 3: separation clock
def fig3(R, arr, vv, jb, T, dpl):
    OFF = np.arange(-5, 16); n = len(R); r_ = np.arange(n)
    cols = np.clip(jb[:, None] + OFF[None, :], 0, T - 1); okw = vv[r_[:, None], cols] & ((jb[:, None] + OFF[None, :]) < T)
    dist = np.hypot(arr["x_rel"] - arr["dx"], arr["y_field"] - arr["dy"])
    S = np.where(okw, dist[r_[:, None], cols], np.nan); S = S - S[:, [2]]; tt = OFF / 10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for ax, cell in zip(axes, ["zone-matching", "man-matching"]):
        m = (R.cell == cell).values; terc, q = terciles(R, R.cell == cell, dpl); tv = terc.values
        gap = (R.t_throw - R.t_break)[m].median()
        ax.axvspan(.3, 1.0, color=GOLD, alpha=.18, zorder=0)
        for tr, col, lab in [("low", LO, "bottom third"), ("mid", MID, "middle third"), ("high", HI, "top third")]:
            mm = m & (tv == tr); ids = R[mm].nfl_id_d.unique(); mu = np.nanmean(S[mm], 0)
            bs = []
            for _ in range(300):
                pick = rng.choice(ids, len(ids)); idx = np.concatenate([np.where(mm & (R.nfl_id_d.values == i))[0] for i in pick]); bs.append(np.nanmean(S[idx], 0))
            lo, hi = np.nanpercentile(bs, [5, 95], 0)
            ax.fill_between(tt, lo, hi, color=col, alpha=.13 if tr != "mid" else .0); ax.plot(tt, mu, color=col, lw=2.8 if tr != "mid" else 1.6, label=f"{lab} Drive+ ({len(ids)} DBs)")
        ax.axvline(0, color="k", lw=1); ax.axhline(0, color="k", lw=.5)
        ax.axvline(gap, color="#555", lw=1, ls="--")
        ymax = ax.get_ylim()[1]
        ax.text(0.02, ymax * .97, "receiver\nbreaks", fontsize=9, va="top"); ax.text(gap + .02, ymax * .97, f"typical throw\n(+{gap:.1f} s)", fontsize=9, va="top", color="#555")
        ax.text(.65, ymax * .55, "steps 3-5 of\nthe DB's drive", ha="center", fontsize=9, color="#7a5c00", fontweight="bold")
        ax.set_xlabel("seconds from the receiver's break"); ax.set_ylabel("separation gained since 0.3 s before the break (yd)")
        ax.set_title({"zone-matching": "Zone, matching the receiver", "man-matching": "Man coverage"}[cell], loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0, .82), fontsize=8.8, frameon=False)
    fig.suptitle("The separation clock: high-Drive+ DBs stop the receiver's break from turning into space", fontsize=15, fontweight="bold", x=.01, ha="left", y=1.02)
    footer(fig, "Bands: 90% player bootstrap. 2023-25 coverage reps (≤ 3 yd for ≥ 1 s), both players in the BDB pool, all graded DBs; Drive+ thirds within each panel.")
    fig.savefig(f"{OUTD}/fig3_separation_clock.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============================================================ FIGURE 4: throw map
def fig4():
    src = open(f"{ROOT}/code/77b_depth_check.py").read(); ns = {}; exec(src.split("class FWL2")[0], ns); d = ns["d"]
    cbt = d.drop_duplicates("nfl_id").set_index("nfl_id").dp; q = pd.qcut(cbt, 3, labels=["low", "mid", "high"]); d["terc"] = d.nfl_id.map(q)
    z = d[(d.team_coverage_man_zone == "ZONE_COVERAGE") & d.rel.isin(["toward", "away"])].copy()
    z["band"] = pd.cut(z.air, [-99, 10, 20, 99], labels=["short", "intermediate", "deep"], right=False)
    t = z.groupby(["terc", "rel", "band"]).agg(n=("completion", "size"), comp=("completion", "mean")).reset_index()
    t["share"] = t.n / t.groupby(["terc", "rel"]).n.transform("sum")
    fig = plt.figure(figsize=(14, 7.6))
    axes = [fig.add_axes([0.03, 0.13, 0.45, 0.72]), fig.add_axes([0.52, 0.13, 0.45, 0.72])]
    cmap = plt.get_cmap("RdYlBu_r"); norm = plt.Normalize(.25, .80)
    Y = {"short": (0, 10), "intermediate": (10, 20), "deep": (20, 32)}
    for ax, tr, col in [(axes[0], "low", LO), (axes[1], "high", HI)]:
        turf(ax, -30, 30, -4, 36, step=5, hash_marks=False)
        ax.add_patch(Rectangle((-26.65, -4), 53.3, 40, fill=False, ec="white", lw=2, zorder=3))
        ax.axhline(0, color="#1e3a8a", lw=3, zorder=4, xmin=(30 - 26.65) / 60, xmax=(30 + 26.65) / 60)
        ax.text(0, -2.4, "line of scrimmage", color="white", ha="center", fontsize=9.5)
        for side, (x0, x1) in [("away", (-26.65, -.6)), ("toward", (.6, 26.65))]:
            for b, (y0, y1) in Y.items():
                rr = t[(t.terc == tr) & (t.rel == side) & (t.band == b)].iloc[0]
                other = t[(t.terc == tr) & (t.rel == ("away" if side == "toward" else "toward")) & (t.band == b)].iloc[0]
                c = cmap(norm(rr.comp))
                ax.add_patch(Rectangle((x0, y0 + .3), x1 - x0, y1 - y0 - .6, color="white", zorder=2))
                ax.add_patch(Rectangle((x0, y0 + .3), x1 - x0, y1 - y0 - .6, color=c, alpha=1.0 if side == "toward" else .55, zorder=2))
                cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
                ax.text(cx, cy + 1.6, f"{100 * rr.comp:.0f}% caught", ha="center", fontsize=13 if side == "toward" else 11, fontweight="bold", color="#111", zorder=5)
                ax.text(cx, cy - 1.2, f"{100 * rr.share:.0f}% of throws  (n {rr.n})", ha="center", fontsize=8.8, color="#222", zorder=5)
                if side == "toward":
                    dlt = 100 * (rr.comp - other.comp)
                    ax.text(cx, cy - 3.9, f"{dlt:+.0f} {'pt' if abs(round(dlt)) == 1 else 'pts'} vs other side", ha="center", fontsize=9.5, fontweight="bold", zorder=5, color="#111",
                            bbox=dict(boxstyle="round,pad=.25", fc="white", ec=LO if dlt > 3 else "#999", lw=1.4 if dlt > 3 else .6, alpha=.95))
        for b, (y0, y1) in Y.items():
            ax.text(-29.4, (y0 + y1) / 2, b.upper(), color="white", fontsize=9, fontweight="bold", rotation=90, va="center", ha="center")
        ax.text(13.8, 33.6, "HIS SIDE", ha="center", color="white", fontsize=12, fontweight="bold")
        ax.text(-13.8, 33.6, "other side", ha="center", color="white", fontsize=11)
        ax.scatter([24.2], [1.8], s=380, marker="o", color=col, edgecolor="white", lw=2, zorder=6)
        ax.text(24.2, 1.8, "CB", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=7)
        ax.set_title(f"Throwing at a {'LOW' if tr == 'low' else 'HIGH'} Drive+ corner  ({'bottom' if tr == 'low' else 'top'} third, {(q == tr).sum()} CBs)",
                     color=col, fontsize=13)
    cax = fig.add_axes([0.25, 0.105, 0.5, 0.02]); sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label("completion % (zone coverage)"); cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    fig.text(.01, .975, "QBs don't avoid high-Drive+ corners. They throw deeper at them, and it doesn't work", fontsize=16, fontweight="bold")
    fig.text(.01, .915, "Every 2023-25 zone-coverage pass with a graded pool CB on a clear side (14.6K throw-CB pairs). His side vs the other side of the field, by depth of the throw. "
             "Deep balls at a low-Drive+ corner: 45% caught (34% to the other side). At a high-Drive+ corner: 32% (32%), and he sees more of them.", fontsize=9.8, color="#444", wrap=True)
    footer(fig, "Raw rates. Model with defense and offense team-season fixed effects, per 10 Drive+ points, his side minus other side: completion -2.05 pp (p .026); "
           "share of short throws -2.2 pp (p .012); air yards +0.48 (p .014).")
    fig.savefig(f"{OUTD}/fig4_throw_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ============================================================ FIGURE 5: Drive+ is not the 40 + big board
def fig5():
    G = GR[GR.nfl_position.isin(["CB", "SS", "FS", "DB"])].join(base[["forty"]])
    fig = plt.figure(figsize=(16.5, 8.5)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=.62)
    ax = fig.add_subplot(gs[0])
    for lo_, hi_, c in [(110, 145, "#dbe8f7"), (100, 110, "#eef4fb"), (90, 100, "#fbeeee"), (70, 90, "#f7dbdb")]:
        ax.axhspan(lo_, hi_, color=c, zorder=0)
    for lab, y_ in [("110+", 112), ("100-110", 104.5), ("90-100", 94.5), ("<90", 86)]:
        ax.text(4.613, y_, "tier " + lab, fontsize=8.5, color="#555", va="center", ha="left", style="italic")
    cb_ = G[G.nfl_position == "CB"].dropna(subset=["forty"]); s_ = G[G.nfl_position != "CB"].dropna(subset=["forty"])
    ax.scatter(s_.forty, s_.drive_plus, s=30, color="#9e9e9e", alpha=.6, label=f"safeties / DBs ({len(s_)})", zorder=2)
    ax.scatter(cb_.forty, cb_.drive_plus, s=55, c=cb_.drive_plus, cmap="coolwarm_r" if False else "RdBu", vmin=80, vmax=125, edgecolor="k", lw=.5, label=f"cornerbacks ({len(cb_)})", zorder=3)
    r = np.corrcoef(cb_.forty, cb_.drive_plus)[0, 1]
    top7 = list(cb_.sort_values("drive_plus").index[-7:][::-1]); low4 = list(cb_.sort_values("drive_plus").index[:4]); fast = list(cb_.sort_values("forty").index[:2])
    cluster = [i for i in top7 if 112 < cb_.loc[i].drive_plus < 116]
    for k_, pid in enumerate(cluster):
        rr = cb_.loc[pid]; ax.annotate(f"{rr.display_name} {rr.drive_plus:.0f}", (rr.forty, rr.drive_plus), xytext=(4.605, 131 - 4 * k_), textcoords="data",
                                       fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#555", lw=.6), va="center")
    for pid in dict.fromkeys([i for i in top7 if i not in cluster] + low4 + fast):
        rr = cb_.loc[pid]; ax.annotate(f"{rr.display_name} {rr.drive_plus:.0f}", (rr.forty, rr.drive_plus), xytext=(5, 3), textcoords="offset points", fontsize=8.5)
    ax.axhline(100, color="k", lw=.7, ls=":")
    ax.set_xlim(4.62, 4.20); ax.set_ylim(76, 140)
    ax.set_xlabel("official 40-yard dash (s)  →  faster"); ax.set_ylabel("Drive+  (100 = avg 2023-25 CB, 10 = 1 SD)")
    ax.set_title(f"Drive+ isn't the stopwatch (r = {r:+.2f} with the 40 among CBs)", loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax2 = fig.add_subplot(gs[1]); top = G.sort_values("drive_plus", ascending=False).head(20).iloc[::-1]
    tcol = {"110+": HI, "100-110": "#64b5f6"}
    ax2.barh(range(len(top)), top.drive_plus - 100, left=100, color=[HI if p_ == "CB" else "#78909c" for p_ in top.nfl_position], height=.72)
    for i, (pid, rr) in enumerate(top.iterrows()):
        ax2.text(rr.drive_plus + .5, i, f"{rr.drive_plus:.0f}", va="center", fontsize=9, fontweight="bold")
    ax2.set_yticks(range(len(top))); ax2.set_yticklabels([f"{rr.display_name}  ({rr.nfl_position}, {int(rr.draft_year)})" for _, rr in top.iterrows()], fontsize=9)
    ax2.set_xlim(100, 140); ax2.set_xlabel("Drive+"); ax2.tick_params(axis="y", length=0)
    ax2.bar([0], [0], color=HI, label="CB"); ax2.bar([0], [0], color="#78909c", label="safety / DB"); ax2.legend(loc="lower right", fontsize=9, frameon=False)
    ax2.set_title("The Drive+ big board: top 20 DBs, 2023-25 classes", loc="left")
    fig.suptitle("Who drives out of the break", fontsize=16, fontweight="bold", x=.01, ha="left", y=.99)
    footer(fig, "Drive+ = frozen formula (prereg/drive_plus.py): drive top speed out of the 45° transition, z-scored within combine class, rescaled to the 2023-25 pool CBs.")
    fig.savefig(f"{OUTD}/fig5_drive_plus_board.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if "1" in FIG: fig1(); print("fig1 done", flush=True)
if "2" in FIG or "3" in FIG:
    R, arr, vv, jb, T, dpl = nfl_reps()
    if "2" in FIG: fig2(R, arr, vv, jb, T, dpl); print("fig2 done", flush=True)
    if "3" in FIG: fig3(R, arr, vv, jb, T, dpl); print("fig3 done", flush=True)
if "4" in FIG: fig4(); print("fig4 done", flush=True)
if "5" in FIG: fig5(); print("fig5 done", flush=True)
