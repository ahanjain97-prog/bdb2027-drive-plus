"""Football-native submission figures.
F3 film still: typical zone-match IN-route rep, top-third vs bottom-third Drive+ corner, drawn as a coach's diagram (rep_film.json).
F6 route tree: median separation gap (bottom-third minus top-third Drive+) by route, zone-match vs man, on the wide reps.
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
ROOT = "/Users/ahanjain/bdb_ssl"; OUT = f"{ROOT}/submission/figs"
HI, LO, MID = "#1565c0", "#c62828", "#9e9e9e"; TURF, TURF2, CHALK = "#2f6b33", "#357a39", "#f4f4f0"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
foot = lambda fig, t: fig.text(.01, .004, t, fontsize=7.5, color="#666")

# ============================================================ F3 film still
FILM = json.load(open(f"{ROOT}/results/viz/rep_film.json"))["reps"]
ROUTE = "CORNER"; RNAME = {"CORNER": "corner route", "POST": "post route", "IN": "in/dig route"}[ROUTE]
def typical(grp, route=ROUTE, cell="zone-matching"):
    reps = [r for r in FILM if r["cell"] == cell and r["route"] == route and r["grp"] == grp and r["sep"][10] is not None and r["sep"][20] is not None
            and (r["jt"] is None or r["jt"] >= r["jb"] + 3) and r["wr"]["y"][r["jb"]] >= 7 and len(r["wr"]["x"]) > r["jb"] + 10]
    print(grp, route, "eligible reps", len(reps))
    sb = np.median([r["sep"][10] for r in reps]); s10 = np.median([r["sep"][20] for r in reps])
    return sorted(reps, key=lambda r: (r["sep"][10] - sb) ** 2 + (r["sep"][20] - s10) ** 2)[0], len(reps), sb, s10
def field(ax, x0, x1, y0, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=TURF, zorder=0))
    for y in range(int(np.floor(y0 / 5) * 5), int(y1) + 1, 5):
        if y0 <= y <= y1:
            ax.axhline(y, color=CHALK, lw=1.2 if y % 10 == 0 else .8, alpha=.7, zorder=1)
            if y % 10 == 0 and y > 0: ax.text(x0 + .6, y + .3, f"{y}", color=CHALK, fontsize=8, alpha=.9)
    for y in np.arange(np.ceil(y0), y1, 1):
        for hx in (-3.08, 3.08): ax.plot([hx - .35, hx + .35], [y, y], color=CHALK, lw=.8, alpha=.55, zorder=1)
    for sx in (-26.65, 26.65):
        if x0 < sx < x1: ax.axvline(sx, color=CHALK, lw=3, zorder=2)
    ax.axhline(0, color="#1e3a8a", lw=3, zorder=2); ax.text(x1 - .6, .4, "LOS", color="white", fontsize=8.5, ha="right", fontweight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
A, nA, sbA, s10A = typical("top"); B, nB, sbB, s10B = typical("bottom")
fig, axes = plt.subplots(1, 2, figsize=(12, 9.4))
def ext(r): return np.concatenate([r["wr"]["x"][:r["jb"] + 11], r["db"]["x"][:r["jb"] + 11]]), np.concatenate([r["wr"]["y"][:r["jb"] + 11], r["db"]["y"][:r["jb"] + 11]])
for ax, r, col, lab, n_, sb, s10 in [(axes[0], A, HI, "Top-third Drive+", nA, sbA, s10A), (axes[1], B, LO, "Bottom-third Drive+", nB, sbB, s10B)]:
    xs_, ys_ = ext(r); cx = np.median(xs_); field(ax, cx - 9, cx + 9, min(-3, ys_.min() - 1.5), ys_.max() + 4)
    wx, wy, dx, dy, do = map(np.array, (r["wr"]["x"], r["wr"]["y"], r["db"]["x"], r["db"]["y"], r["db"]["o"]))
    jb, jt = r["jb"], r["jt"]; last = min(len(wx) - 1, jb + 10)
    ax.plot(dx[:last + 1], dy[:last + 1], color=col, lw=3.0, solid_capstyle="round", zorder=4)
    ax.plot(wx[:last + 1], wy[:last + 1], color="white", lw=2.6, solid_capstyle="round", zorder=5)
    marks = [(0, "1", "snap"), (jb, "2", "key move"), (min(jb + 5, last), "3", "+0.5 s"), (last, "4", "+1.0 s")]; lines = []
    for j, num, tag in marks:
        ax.plot([wx[j], dx[j]], [wy[j], dy[j]], color="white", lw=1.0, ls=(0, (2, 2)), zorder=3)
        tri = np.array([[0, 1.0], [-.65, -.6], [.65, -.6]]) * .85; th = np.radians(do[j]); R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        ax.add_patch(Polygon(tri @ R.T + [dx[j], dy[j]], closed=True, color=col, ec="white", lw=1.2, zorder=7))
        ax.add_patch(plt.Circle((wx[j], wy[j]), .5, color="white", ec="k", lw=1.2, zorder=6))
        mx, my = (wx[j] + dx[j]) / 2, (wy[j] + dy[j]) / 2
        ax.text(mx, my, num, ha="center", va="center", fontsize=8.5, fontweight="bold", color="k", bbox=dict(boxstyle="circle,pad=.18", fc="#f9a825", ec="k", lw=.8), zorder=9)
        lines.append(f"{num}  {tag:9s} {float(np.hypot(wx[j] - dx[j], wy[j] - dy[j])):.1f} yd")
    if jt is not None and jt <= last:
        ax.add_patch(plt.Circle((wx[jt], wy[jt]), 1.1, fill=False, color="#f9a825", lw=2.2, zorder=6)); lines.append(f"ball thrown at +{(jt - jb) / 10:.1f} s")
    ax.text(.02, .02, "\n".join(lines), transform=ax.transAxes, fontsize=9.2, family="DejaVu Sans Mono", color="white", va="bottom", bbox=dict(boxstyle="round,pad=.4", fc="#00000099", ec="none"), zorder=10)
    ax.set_title(f"{lab}: {r['cbName']} ({r['dp']:.0f})\nvs {r['wrName']} · {r['season']} wk {r['week']} · {r['shell']} · {r['team']} D", color=col, fontsize=11, loc="left")
    ax.text(.02, .975, f"typical of {n_} {'top' if col == HI else 'bottom'}-third reps on this route\n(group medians: {sb:.1f} yd at the move, {s10:.1f} yd at +1.0 s)", transform=ax.transAxes, color="white", fontsize=8.5, va="top", zorder=10)
fig.suptitle(f"Film still: {RNAME}, zone-match, two corners. ○ receiver · ▲ corner, pointing where his shoulders face · dashed = separation", fontsize=12, fontweight="bold", x=.01, ha="left", y=.995)
foot(fig, f"Real 2023–25 reps from the registered 3-yd film set; each panel is the rep closest to its group's median separation at the key move and at +1.0 s on {RNAME}s (break ≥ 7 yd downfield) in zone-match coverage. Field to scale (yard lines, hashes, LOS).")
fig.savefig(f"{OUT}/F3_film_still.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ============================================================ F6 route tree (wide reps)
base = pd.read_parquet(f"{ROOT}/data/baseline.parquet").set_index("nfl_id")
W = pd.read_parquet(f"{ROOT}/data/outcomes/89_wide_reps.parquet")
W["cell"] = np.where(W.pa >= .5, np.where(W.zone == 1, "zone-matching", "man-matching"), "other")
dpl = W.drop_duplicates("nfl_id_d").set_index("nfl_id_d").dp
TREE = {"GO": [(0, 0), (0, 22)], "POST": [(0, 0), (0, 12), (-7.5, 19.5)], "CORNER": [(0, 0), (0, 12), (7.5, 19.5)], "IN": [(0, 0), (0, 10), (-10, 10)],
        "OUT": [(0, 0), (0, 10), (10, 10)], "CROSS": [(0, 0), (0, 4), (-14, 6.8)], "SLANT": [(0, 0), (0, 3), (-6.8, 8.3)], "HITCH": [(0, 0), (0, 6.5), (1.2, 5.3)]}
NAME = {"GO": "go", "POST": "post", "CORNER": "corner", "IN": "in / dig", "OUT": "out", "CROSS": "crosser", "SLANT": "slant", "HITCH": "hitch / curl"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6.9)); cmap = plt.get_cmap("RdBu"); norm = plt.Normalize(-1.5, 1.5)
LABPOS = {"GO": (0, 22.9, "center", "bottom"), "POST": (-8.1, 20.0, "right", "bottom"), "CORNER": (8.1, 20.0, "left", "bottom"), "IN": (-10.6, 11.3, "right", "bottom"), "OUT": (10.6, 10.6, "left", "bottom"),
          "CROSS": (-14.4, 6.6, "right", "center"), "SLANT": (-7.1, 8.7, "right", "bottom"), "HITCH": (1.9, 5.4, "left", "center")}
rows = []
for ax, cell, title in [(axes[0], "zone-matching", "Zone, matching the receiver"), (axes[1], "man-matching", "Man coverage")]:
    d = W[W.cell == cell]; ids = d.nfl_id_d.unique(); q = pd.qcut(dpl.reindex(ids), 3, labels=["low", "mid", "high"]); d = d.assign(terc=d.nfl_id_d.map(q))
    ax.add_patch(Rectangle((-20, -2), 40, 26.5, color=TURF, zorder=0))
    for y in range(0, 25, 5): ax.axhline(y, color=CHALK, lw=1.0 if y % 10 == 0 else .7, alpha=.6, zorder=1); ax.text(-19.6, y + .3, f"{y}", color=CHALK, fontsize=8, alpha=.9)
    for y in range(-1, 24):
        for hx in (-3.08, 3.08): ax.plot([hx - .3, hx + .3], [y, y], color=CHALK, lw=.7, alpha=.5, zorder=1)
    ax.axhline(0, color="#1e3a8a", lw=3, zorder=2)
    for rt, pts in TREE.items():
        g = d[d.route_ran == rt]; lo_, hi_ = g[g.terc == "low"].sep_gain_break.dropna(), g[g.terc == "high"].sep_gain_break.dropna()
        if len(lo_) < 8 or len(hi_) < 8: continue
        gap = float(lo_.median() - hi_.median()); n = len(lo_) + len(hi_); rows.append({"cell": cell, "route": rt, "gap": gap, "n_low": len(lo_), "n_high": len(hi_)})
        col = cmap(norm(gap)); lw = 2.5 + 7 * min(n, 300) / 300
        xs, ys = zip(*pts); ax.plot(xs, ys, color="white", lw=lw + 2.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=4)
        ex, ey = pts[-1]; px, py = pts[-2]; ang = np.degrees(np.arctan2(ey - py, ex - px))
        ax.add_patch(FancyArrowPatch((px + (ex - px) * .92, py + (ey - py) * .92), (ex, ey), arrowstyle="-|>", mutation_scale=16, color=col, lw=0, zorder=5))
        lx, ly, ha, va = LABPOS[rt]
        ax.text(lx, ly, f"{NAME[rt]}  {gap:+.1f} yd\n(n {n})", ha=ha, va=va, fontsize=9.2, color="white", fontweight="bold", linespacing=1.15,
                bbox=dict(boxstyle="round,pad=.2", fc="#00000070", ec="none"), zorder=6)
    ax.add_patch(plt.Circle((0, 0), .8, color="white", ec="k", lw=1.5, zorder=7)); ax.text(0, -1.9, "receiver", color="white", ha="center", fontsize=9)
    ax.set_xlim(-20, 20); ax.set_ylim(-2.5, 24.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=.035, pad=.04, aspect=45)
cb.set_label("extra separation given up by a bottom-third Drive+ corner vs a top-third corner, 1.0 s after the receiver's key move (yd; blue = Drive+ helps)")
fig.suptitle("Where Drive+ shows up: routes that keep going after the move. Little on stop routes, little in man.", fontsize=13, fontweight="bold", x=.01, ha="left", y=.99)
foot(fig, "Wide reps, 2023–25, Drive+ thirds among DBs with reps in each coverage; median separation change from 0.3 s before the key move to 1.0 s after, bottom third minus top third; line width = reps. Routes with fewer than 8 reps per group omitted. Raw medians, exploratory.")
fig.savefig(f"{OUT}/F6_route_tree.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print(pd.DataFrame(rows).pivot(index="route", columns="cell", values="gap").round(2).to_string())
