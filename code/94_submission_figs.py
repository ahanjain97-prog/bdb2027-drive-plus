"""Submission figures on the registered wide sample, built from results/viz/drive_plus_viz_wide.json (same numbers as the page).
F2 sunday (clock zone/man + one-second clouds)   F3 Drive+ curves (open / composite / completion)   F5 supported-in-every-split
Reused as-is: fig1 (drill), fig4 (throw map), fig7 (before/after), fig5 (board). All copied, downscaled to <= 1600 px, to submission/figs/.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT = "/Users/ahanjain/bdb_ssl"; OUT = f"{ROOT}/submission/figs"
D = json.load(open(f"{ROOT}/results/viz/drive_plus_viz_wide.json"))
HI, LO, MID, GOLD = "#1565c0", "#c62828", "#9e9e9e", "#f9a825"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})
foot = lambda fig, t: fig.text(.01, .004, t, fontsize=7.5, color="#666")
nn = lambda a: np.array([np.nan if v is None else v for v in a], dtype=float)

# ---------------- F2
C, CH = D["clock"], D["chase"]; T = nn(C["t"])
fig = plt.figure(figsize=(13, 9.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.42, wspace=.22)
for j, cell in enumerate(["zone-matching", "man-matching"]):
    ax = fig.add_subplot(gs[0, j]); c = C[cell]
    ax.axvspan(.3, 1.0, color=GOLD, alpha=.16)
    for tr, col, lab in [("low", LO, "bottom third"), ("mid", MID, "middle third"), ("high", HI, "top third")]:
        mu, lo, hi = nn(c[tr]["mean"]), nn(c[tr]["lo"]), nn(c[tr]["hi"])
        if tr != "mid": ax.fill_between(T, lo, hi, color=col, alpha=.13)
        ax.plot(T, mu, color=col, lw=2.6 if tr != "mid" else 1.5, label=f"{lab} Drive+ ({c[tr]['cut'][0]:.0f}–{c[tr]['cut'][1]:.0f}, {c[tr]['dbs']} DBs)")
    ax.axvline(0, color="k", lw=1); ax.axhline(0, color="k", lw=.5); ax.axvline(c["gap"], color="#555", lw=1, ls="--")
    ym = ax.get_ylim()[1]; ax.text(.02, ym * .97, "receiver's\nkey move", fontsize=9, va="top"); ax.text(c["gap"] + .02, ym * .97, "typical throw", fontsize=9, va="top", color="#555")
    ax.text(.65, ym * .08, "DB's drive (steps 3–5)", ha="center", fontsize=9, color="#7a5c00", fontweight="bold")
    ax.set_title(("Zone, matching the receiver" if j == 0 else "Man coverage") + f"  ({c['reps']} reps)", loc="left")
    ax.set_xlabel("seconds from the receiver's key move"); ax.set_ylabel("separation gained since 0.3 s before (yd)"); ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0, .86))
for j, (tr, col, lab) in enumerate([("low", LO, "Bottom third Drive+"), ("high", HI, "Top third Drive+")]):
    ax = fig.add_subplot(gs[1, j]); z = CH["zone-matching"][tr]; f, l = nn(z["f"]), nn(z["l"])
    ax.set_facecolor("#2f6b33"); ax.set_aspect("equal"); ax.set_xlim(-4.6, 2.6); ax.set_ylim(-2.2, 4.2)
    for v in range(-2, 5): ax.axhline(v, color="#f4f4f0", lw=.6, alpha=.35)
    for r, lab2, w in [(1, "1 yd", .8), (2, "2 yd: open", 2.0), (3, "3 yd", .8)]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="white", lw=w, ls="-" if r == 2 else "--", alpha=.95 if r == 2 else .6))
    ax.text(-3.55, 1.55, "2 yd: open →", color="white", fontsize=9, fontweight="bold"); ax.scatter(f, l, s=14, color=col, alpha=.6, edgecolor="white", lw=.3)
    ax.scatter([0], [0], s=380, color="white", edgecolor="k", zorder=5); ax.text(0, 0, "WR", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
    ax.annotate("", xy=(2.3, 0), xytext=(.35, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=2)); ax.text(1.3, -.3, "receiver's direction", color="white", fontsize=8.5, ha="center")
    ax.text(-4.4, 3.85, f"receiver open (2+ yd): {int(100 * z['open'] + .5)}% of reps", color="white", fontsize=12, fontweight="bold", va="top", bbox=dict(boxstyle="round,pad=.3", fc=col, ec="white"))
    ax.text(-4.4, 3.05, f"median separation {z['median']:.1f} yd · {z['reps']} reps, {z['dbs']} DBs", color="white", fontsize=9.5, va="top")
    ax.set_title(f"{lab}: 1.0 s after the move, zone-match", loc="left", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Sundays: in zone-match coverage the gap opens during the DB's drive; in man it barely moves", fontsize=14, fontweight="bold", x=.01, ha="left", y=.98)
foot(fig, "Wide coverage reps: pool receiver and graded DB within 4 yd from the snap to the receiver's key move (pre-move selection only), 2023–25, all graded DBs. Bands: 90% player bootstrap. "
          "Bottom: each ▽ is the DB on one rep, mirrored so he started on the receiver's left.")
fig.savefig(f"{OUT}/F2_sundays.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------- F3
cv = D["curves"]; G = nn(cv["grid"])
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
f2 = lambda v: f"{v:+.2f}"
PAN = [("open", "Every rep: his man open at the throw (2+ yd)", 100, lambda v: f"{v:.0f}%", lambda v: f"{v:+.1f} pts"),
       ("composite", "Every rep: composite coverage score", 1, lambda v: f"{v:+.1f}", lambda v: f"{v:+.2f} SD"),
       ("comp", "Targets: completion % allowed", 100, lambda v: f"{v:.0f}%", lambda v: f"{v:+.1f} pts")]
for ax, (k, title, sc, tick, fe) in zip(axes, PAN):
    c = cv[k]; mu, lo, hi = nn(c["mu"]) * sc, nn(c["lo"]) * sc, nn(c["hi"]) * sc
    dots = c["dots"]; ax.scatter([d["dp"] for d in dots], [d["v"] * sc for d in dots], s=[8 + d["n"] * (0.5 if k != "comp" else 2) for d in dots], color=MID, alpha=.45, edgecolor="white", lw=.6)
    ax.fill_between(G, lo, hi, color=HI, alpha=.14); ax.plot(G, mu, color=HI, lw=2.4); ax.axvline(100, color="#bbb", lw=.8, ls=":")
    if k == "composite": ax.axhline(0, color="#bbb", lw=.8)
    ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel("Drive+  (100 = average CB, 10 = 1 SD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, t=tick: t(v)))
    lo80, hi80 = (c["slope"] - 1.2816 * c["se"]) * sc, (c["slope"] + 1.2816 * c["se"]) * sc
    ax.text(.03, .04, f"{fe(c['slope'] * sc)} per 10 Drive+\n80% interval {fe(lo80)} to {fe(hi80)}\none-sided permutation p {c['p']:.3f}\n{c['n']:,} {'targets' if k == 'comp' else 'reps'}, {c['dbs']} DBs",
            transform=ax.transAxes, fontsize=9.5, va="bottom", bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#bbb"))
fig.suptitle("Higher Drive+, tighter coverage, fewer catches (line = rep-level fit with 90% player-bootstrap band; dots = each DB's average, sized by sample)", fontsize=12.5, fontweight="bold", x=.01, ha="left", y=1.02)
foot(fig, "Wide reps, all graded DBs, 2023–25. Composite = mean of four context-adjusted measures per rep (separation gained, closing, separation at throw, open), SD units, higher = tighter; player-mean reliability .70.")
fig.savefig(f"{OUT}/F3_curves.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------- F5 splits
S = D["splits"]; splits = list(dict.fromkeys(s["split"] for s in S))
CL = [("sep gained (all)", "separation gained (yd)"), ("closing, zone-matching", "closing, zone-match (yd/s)"), ("closing, all zone", "closing, all zone (yd/s)"),
      ("open at throw (all)", "receiver open at throw (pts)"), ("sep at throw (all)", "separation at throw (yd)"), ("completion (targets)", "completion when targeted (pts)")]
fig, axes = plt.subplots(1, 6, figsize=(17, 5.4), sharey=True)
LAB = {"radius 3 yd": "radius 3 yd", "radius 4 yd": "radius 4 yd (registered)", "radius 5 yd": "radius 5 yd", "4 yd, CBs only": "corners only", "4 yd, align ≥ .7": "stricter matching (.7)",
       "4 yd, 2024 season": "2024 season only", "4 yd, 2025 season": "2025 season only", "4 yd, no GO": "verticals removed"}
for ax, (key, lab) in zip(axes, CL):
    rows = []
    for sp in splits:
        s = next((x for x in S if x["split"] == sp and x["claim"].startswith(key)), None); rows.append(s)
    for i, s in enumerate(rows):
        if s is None: continue
        col = HI if s["verdict"] == "supported" else (MID if s["verdict"] == "right sign" else LO)
        yy = len(splits) - 1 - i; ax.plot([s["lo"], s["hi"]], [yy, yy], color=col, lw=2, alpha=.8); ax.scatter([s["b"]], [yy], color=col, s=42, zorder=3, edgecolor="white", lw=1)
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(len(splits))); ax.set_yticklabels([LAB.get(sp, sp) for sp in splits][::-1], fontsize=9)
    ax.set_title(lab, fontsize=10, loc="left"); ax.tick_params(axis="x", labelsize=8.5); ax.grid(axis="x", color="#eee")
sup = sum(s["verdict"] == "supported" for s in S)
fig.suptitle(f"Supported in {sup} of {len(S)} claim × split cells under a standard fixed in advance (dot = estimate per 10 Drive+, bar = 80% interval; blue = expected sign, one-sided permutation p < .10 and interval excluding zero; gray = right sign only)",
             fontsize=11.5, fontweight="bold", x=.01, ha="left", y=1.03)
foot(fig, "Wide reps, all graded DBs unless noted. The one gray cell is completion when targeted in the 2024 season alone (155 targets).")
fig.savefig(f"{OUT}/F5_splits.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------- copy / downscale reused figures
for src, dst in [("fig1_combine_drill.png", "F1_drill.png"), ("fig4_throw_map.png", "F4_throw_map.png"), ("fig7_before_after_throw.png", "F6_after_throw.png"), ("fig5_drive_plus_board.png", "F7_board.png"),
                 ("../../submission/figs/F2_sundays.png", "F2_sundays.png"), ("../../submission/figs/F3_curves.png", "F3_curves.png"), ("../../submission/figs/F5_splits.png", "F5_splits.png")]:
    im = Image.open(f"{ROOT}/results/figs/{src}").convert("RGB"); w, h = im.size
    if w > 1600: im = im.resize((1600, int(h * 1600 / w)), Image.LANCZOS)
    im.save(f"{OUT}/{dst}", optimize=True, quality=88)
import os; print({f: f"{os.path.getsize(f'{OUT}/{f}') // 1024} KB" for f in sorted(os.listdir(OUT))})
