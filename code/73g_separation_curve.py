"""Step 8 -- the headline chart: receiver-CB separation around the receiver's break, by combine drive-speed tercile.

Separation relative to 0.3 s before the break (so every curve starts at 0 where sep_gain_break starts), from -0.5 to
+1.5 s, for zone-matching and man-matching reps; mean per tercile with CB-cluster bootstrap 95% bands (300 draws);
bottom panel = top-minus-bottom tercile difference. Also reports where the zone-matching difference first excludes 0.
Output results/73g_separation_curve.png (+ csv).
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
rng = np.random.default_rng(97)
voc = json.load(open(f"{ROOT}/data/seq/vocab.json")); F, PRE = voc["feats"], voc["pre"]
X = np.load(f"{ROOT}/data/seq/X.npy", mmap_mode="r"); V = np.load(f"{ROOT}/data/seq/valid.npy")
Z = pd.read_parquet(f"{ROOT}/data/features/cb_drive_parts.parquet")
cov = pd.read_parquet(f"{ROOT}/data/outcomes/cb_reps_sink_bend.parquet")
cov = cov[cov.nfl_id_d.isin(Z.index) & cov.cell.isin(["zone-matching", "man-matching"])].reset_index(drop=True)
g = lambda idx, n: np.asarray(X[idx][:, :, F.index(n)], dtype=np.float32)
A, B = cov.idx_r.values, cov.idx_d.values
sr, lr, pr = g(A, "s"), g(A, "a_lat"), g(A, "post_pass")
dist = np.hypot(g(A, "x_rel") - g(B, "x_rel"), g(A, "y_field") - g(B, "y_field"))
vv = V[A] & V[B]; T = X.shape[1]; k = np.arange(T)[None, :] - PRE
ls = lr.copy(); ls[:, 1:-1] = (lr[:, :-2] + lr[:, 1:-1] + lr[:, 2:]) / 3
a = np.where(vv & (k >= 5) & (k <= 30) & (sr > 2) & (pr < .5), np.abs(ls), -1); jb = a.argmax(1)
OFF = np.arange(-5, 16); r_ = np.arange(len(cov))
cols = np.clip(jb[:, None] + OFF[None, :], 0, T - 1)
S = np.where(vv[r_[:, None], cols] & ((jb[:, None] + OFF[None, :]) < T), dist[r_[:, None], cols], np.nan)
S = S - S[:, [2]]                                                       # relative to -0.3 s
cbz = Z.top.reindex(cov.nfl_id_d.unique())
terc = pd.qcut(cbz, 3, labels=["slowest third", "middle third", "fastest third"])
cov["terc"] = cov.nfl_id_d.map(terc)
print("CBs per tercile:", terc.value_counts().to_dict(), "| reps per cell x tercile:\n", pd.crosstab(cov.cell, cov.terc).to_string())


def curve(mask, ids_sample=None):
    d = cov[mask]
    if ids_sample is not None:
        idx = np.concatenate([d.index[d.nfl_id_d == i].values for i in ids_sample])
    else:
        idx = d.index.values
    return np.nanmean(S[idx], 0)


tt = OFF / 10
out, fig = [], plt.figure(figsize=(12, 8))
axs = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)]
COL = {"slowest third": "#c0392b", "middle third": "#95a5a6", "fastest third": "#2471a3"}
for j, cell in enumerate(["zone-matching", "man-matching"]):
    ax = axs[j]; diffs = []
    for t_ in COL:
        m = (cov.cell == cell) & (cov.terc == t_)
        ids = cov[m].nfl_id_d.unique(); mu = curve(m)
        bs = np.array([curve(m, rng.choice(ids, len(ids))) for _ in range(300)])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], 0)
        ax.plot(tt, mu, color=COL[t_], lw=2.2, label=f"{t_} drive speed ({len(ids)} CBs)"); ax.fill_between(tt, lo, hi, color=COL[t_], alpha=.12)
        for i, x_ in enumerate(tt):
            out.append({"cell": cell, "tercile": t_, "t": x_, "mean": mu[i], "lo": lo[i], "hi": hi[i]})
    ax.axvline(0, color="k", lw=.8, ls=":"); ax.axhline(0, color="k", lw=.5)
    ax.set_title(f"{cell} coverage reps", fontsize=12); ax.set_xlabel("seconds from the receiver's break"); ax.set_ylabel("separation change (yd) vs 0.3 s before the break")
    ax.legend(fontsize=8, loc="upper left")
    mt, mb = (cov.cell == cell) & (cov.terc == "fastest third"), (cov.cell == cell) & (cov.terc == "slowest third")
    it, ib = cov[mt].nfl_id_d.unique(), cov[mb].nfl_id_d.unique()
    dmu = curve(mt) - curve(mb)
    dbs = np.array([curve(mt, rng.choice(it, len(it))) - curve(mb, rng.choice(ib, len(ib))) for _ in range(500)])
    lo, hi = np.nanpercentile(dbs, [2.5, 97.5], 0)
    c_ = "#1e8449" if cell == "zone-matching" else "#7d3c98"
    axs[2].plot(tt, dmu, color=c_, lw=2.2, label=cell); axs[2].fill_between(tt, lo, hi, color=c_, alpha=.15)
    excl = tt[(hi < 0) & (tt > 0)]
    print(f"{cell}: fastest-minus-slowest separation change at +0.5 s {dmu[OFF == 5][0]:+.2f} yd [{lo[OFF == 5][0]:+.2f}, {hi[OFF == 5][0]:+.2f}], "
          f"at +1.0 s {dmu[OFF == 10][0]:+.2f} [{lo[OFF == 10][0]:+.2f}, {hi[OFF == 10][0]:+.2f}], at +1.5 s {dmu[-1]:+.2f} [{lo[-1]:+.2f}, {hi[-1]:+.2f}]; "
          f"band below 0 from: {excl.min() if len(excl) else 'never'} s")
    for i, x_ in enumerate(tt):
        out.append({"cell": cell, "tercile": "fastest - slowest", "t": x_, "mean": dmu[i], "lo": lo[i], "hi": hi[i]})
axs[2].axhline(0, color="k", lw=.6); axs[2].axvline(0, color="k", lw=.8, ls=":")
axs[2].axvspan(0.3, 1.0, color="#f4d03f", alpha=.15, label="phase the drill translates to (0.3-1.0 s)")
axs[2].set_title("fastest-third minus slowest-third drive speed (negative = fast-drive CBs allow less separation)", fontsize=11)
axs[2].set_xlabel("seconds from the receiver's break"); axs[2].set_ylabel("difference (yd)"); axs[2].legend(fontsize=9)
fig.suptitle("Combine transition drive speed vs NFL separation around the receiver's break", fontsize=14, y=.995)
fig.tight_layout(); fig.savefig(f"{ROOT}/results/73g_separation_curve.png", dpi=150)
pd.DataFrame(out).to_csv(f"{ROOT}/results/73g_separation_curve.csv", index=False)
