"""Visuals 6-7 for Drive+ (registered 3-yd reps, all graded DBs; per-rep expected completion from 79).

6 Drive+ curves (Stuff+ style): open rate at the throw (every rep), catch probability if thrown (every rep), completion %
  allowed when targeted -- per-DB dots + rep-level fit with a 90% player-bootstrap band.
7 Before vs after the throw: on the same targets, completion = expected completion (separation, depth, speed at the throw)
  + completion over expected, so the Drive+ slopes add exactly. Shown for a 110 vs a 90 Drive+ DB on a play timeline.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"; OUTD = f"{ROOT}/results/figs"
rng = np.random.default_rng(167)
HI, LO, GOLD = "#1565c0", "#c62828", "#f9a825"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.spines.top": False, "axes.spines.right": False})
R = pd.read_parquet(f"{ROOT}/data/outcomes/79_scored_coverage_reps.parquet")
R["drive_plus"] = 100 + 10 * R.dp
M = pd.read_csv(f"{ROOT}/results/79_metrics_coverage_reps.csv", index_col=0)
footer = lambda fig, t: fig.text(.01, .005, t, fontsize=7.5, color="#666")


def fit_curve(d, y, grid, logit=True, n_boot=300):
    d = d.dropna(subset=[y, "dp"]); f = f"{y} ~ dp"
    mk = (lambda dd: smf.logit(f, data=dd).fit(disp=0)) if logit else (lambda dd: smf.ols(f, data=dd).fit())
    g = pd.DataFrame({"dp": (grid - 100) / 10}); mu = mk(d).predict(g).values; ids = d.nfl_id_d.unique(); bs = []
    by = {i: d[d.nfl_id_d == i] for i in ids}
    for _ in range(n_boot):
        dd = pd.concat([by[i] for i in rng.choice(ids, len(ids))])
        try:
            bs.append(mk(dd).predict(g).values)
        except Exception:
            pass
    lo, hi = np.percentile(bs, [5, 95], 0)
    return mu, lo, hi


def fig6():
    grid = np.linspace(80, 135, 60)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    panels = [("open", R.dropna(subset=["open"]), "Every rep: how often his man is open\n(2+ yd at the throw)", "open rate (sep >= 2 yd)", 10, True, 100, "%"),
              ("xcomp_rep", R.dropna(subset=["xcomp_rep"]), "Every rep: catch probability\nif the ball had gone to his man", "xComp allowed per rep", 10, False, 100, "%"),
              ("completion", R[R.targeted == 1].dropna(subset=["completion"]), "When targeted: completion %\nallowed", "completion allowed (targets)", 5, True, 100, "%")]
    for ax, (y, d, title, mkey, min_n, logit, sc, unit) in zip(axes, panels):
        pdb = d.groupby("nfl_id_d").agg(v=(y, "mean"), n=(y, "size"), dpl=("drive_plus", "first"), s=("is_s", "first"))
        pdb = pdb[pdb.n >= min_n]
        for is_s, mk_ in [(0, "o"), (1, "s")]:
            q = pdb[pdb.s == is_s]
            ax.scatter(q.dpl, sc * q.v, s=12 + 3 * q.n, marker=mk_, c=q.dpl, cmap="RdBu", vmin=80, vmax=125, edgecolor="k", lw=.4, alpha=.85, zorder=3,
                       label=("CB" if is_s == 0 else "safety / DB") + f" ({len(q)})")
        mu, lo, hi = fit_curve(d, y, grid, logit=logit)
        ax.fill_between(grid, sc * lo, sc * hi, color="#555", alpha=.15, zorder=1); ax.plot(grid, sc * mu, color="k", lw=2.4, zorder=2)
        b, p = M.loc[mkey, "Drive+ slope per 10 pts"], M.loc[mkey, "perm p"]
        ax.text(.97, .96, f"{sc * b:+.1f} pts per 10 Drive+\n(p {p:.3f}; {len(d):,} {'targets' if y == 'completion' else 'reps'}, {d.nfl_id_d.nunique()} DBs)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, fontweight="bold", bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#999"))
        ax.axvline(100, color="#999", lw=.8, ls=":")
        ax.set_title(title, loc="left"); ax.set_xlabel("Drive+  (100 = avg CB, 10 = 1 SD)"); ax.set_ylabel(f"{unit}")
        ax.legend(loc="lower left", fontsize=8.5, frameon=False, title=f"DBs with ≥ {min_n} {'targets' if y == 'completion' else 'reps'}", title_fontsize=8.5)
    fig.suptitle("Drive+ on Sundays: higher grade, fewer open receivers, fewer catches", fontsize=16, fontweight="bold", x=.01, ha="left", y=1.03)
    footer(fig, "Lines: rep-level fit with 90% player-bootstrap band; dots: each DB's raw average, sized by sample (small samples are noisy, which is why the line, not the dots, is the evidence). "
                "Slopes: model with route, man/zone and position controls, player-permutation p. 2023-25, 3-yd coverage reps, all graded DBs.")
    fig.savefig(f"{OUTD}/fig6_drive_plus_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig7():
    t = R[(R.targeted == 1)].dropna(subset=["completion", "xcomp_rep", "cpoe_t"]).copy()
    f = "{} ~ dp + C(team_coverage_man_zone) + C(route_ran) + is_s"
    slopes = lambda d: {y: smf.ols(f.format(y), data=d).fit().params["dp"] for y in ["completion", "xcomp_rep", "cpoe_t"]}
    est = slopes(t); ids = t.nfl_id_d.unique(); by = {i: t[t.nfl_id_d == i] for i in ids}; bs = []
    for _ in range(1000):
        bs.append(slopes(pd.concat([by[i] for i in rng.choice(ids, len(ids))])))
    B = pd.DataFrame(bs); ci = {k: np.percentile(B[k], [5, 95]) for k in B}
    k2 = 2 * 100                                                     # 110 vs 90 = 2 SD, in percentage points
    before, after, total = k2 * est["xcomp_rep"], k2 * est["cpoe_t"], k2 * est["completion"]
    print(f"targets {len(t)}, DBs {len(ids)} | per 10 pts: total {100 * est['completion']:+.2f}, before {100 * est['xcomp_rep']:+.2f}, after {100 * est['cpoe_t']:+.2f}")
    fig = plt.figure(figsize=(14, 7.2)); ax = fig.add_axes([.03, .1, .94, .74]); ax.axis("off")
    U = .22                                                            # plot units per percentage point
    ax.set_xlim(-.6, 10.4); ax.set_ylim(-11.6 * U * 10 / 2.2 if False else -26 * U, 2.6)
    stages = [(0, "SNAP"), (3.3, "RECEIVER\nBREAKS"), (6.4, "BALL\nTHROWN"), (9.6, "CATCH\nPOINT")]
    ax.add_patch(plt.Rectangle((-.3, 1.0), 10.2, .55, color="#2f6b33"))
    for xx, lab in stages:
        ax.scatter([xx], [1.27], s=260, color="white", edgecolor="k", zorder=3); ax.text(xx, 1.85, lab, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(9.95, 1.27), xytext=(-.3, 1.27), arrowprops=dict(arrowstyle="-|>", color="white", lw=2), zorder=2)
    ax.axhline(.6, xmin=0, xmax=1, color="#999", lw=.8)
    def bar(x0, x1, val, col, title, sub, lo_, hi_):
        h = val * U
        ax.add_patch(plt.Rectangle((x0, .6), x1 - x0, h, color=col, alpha=.92))
        ax.errorbar([x1 + .18], [.6 + h], yerr=[[(val - lo_) * U], [(hi_ - val) * U]], fmt="o", color="#333", ms=4, capsize=5, lw=1.3)
        ax.text(x1 + .3, .6 + lo_ * U, "90% CI", fontsize=8, color="#555", va="center")
        ax.text((x0 + x1) / 2, .6 + h / 2, f"{val:+.1f}\npts", ha="center", va="center", color="white", fontsize=16, fontweight="bold")
        ax.text((x0 + x1) / 2, .6 + lo_ * U - .35, f"{title}\n{sub}\n(90% CI {lo_:+.1f} to {hi_:+.1f})", ha="center", va="top", fontsize=9.6, color="#222")
    bar(3.75, 5.95, before, GOLD, "BEFORE THE THROW", "tighter coverage: separation,\ndepth, speed at release", k2 * ci["xcomp_rep"][0], k2 * ci["xcomp_rep"][1])
    bar(6.85, 9.15, after, HI, "AFTER THE THROW", "ball in the air / catch point:\ncompletion over expected", k2 * ci["cpoe_t"][0], k2 * ci["cpoe_t"][1])
    ax.text(-.3, .1, "Completion % allowed,\n110 vs 90 Drive+ DB", fontsize=13, fontweight="bold", va="top", color="#333")
    ax.text(-.3, -1.0, f"{total:+.1f} pts", fontsize=28, fontweight="bold", va="top", color=LO)
    ax.text(-.3, -2.3, f"90% CI {k2 * ci['completion'][0]:+.1f} to {k2 * ci['completion'][1]:+.1f}\n{len(t)} targets, {len(ids)} DBs\n\n{before:+.1f} before the throw\n{after:+.1f} after the throw",
            fontsize=10.5, va="top", color="#444")
    fig.text(.01, .955, "Where the Drive+ advantage shows up on a pass play", fontsize=16, fontweight="bold")
    fig.text(.01, .905, "On the same targets, completion % = what separation, depth and receiver speed at the throw predict + what happens after the ball is released, "
             "so the two pieces add up exactly. About two-thirds of the gap appears after the throw; the before-the-throw piece is smaller and its interval touches zero.", fontsize=10, color="#444", wrap=True)
    footer(fig, "Expected completion: gradient-boosting model on 9,787 pool-receiver targets (NGS separation at release, depth, speed, sideline, route, situation), cross-fitted by game, "
                "applied with the DB's own separation. Slopes with route, man/zone and position controls; CIs = 90% player bootstrap. 2023-25, 3-yd reps.")
    fig.savefig(f"{OUTD}/fig7_before_after_throw.png", dpi=200, bbox_inches="tight"); plt.close(fig)


import sys
if "6" in sys.argv[1:] or len(sys.argv) == 1: fig6(); print("fig6 done", flush=True)
fig7(); print("fig7 done", flush=True)
