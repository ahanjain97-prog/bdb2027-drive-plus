"""Robustness of rusher-bend x blocker-quickness interaction: player-level permutation null (shuffle each player's
combine trait across players of the same role, refit), subsets, weight controls, seasons."""
import warnings, contextlib, io
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/Users/ahanjain/bdb_ssl/code/31_matchup_model.py").read().split("res = []")[0])
CTXF = "C(dl_align) + C(ol_align) + C(down) + C(ytg_b) + C(blitzing)"


def tstat(d, y, extra=""):
    mdl = smf.ols(f"{y} ~ z_dl_hoop * z_ol_wave {extra} + {CTXF}", data=d).fit()
    return mdl.params["z_dl_hoop:z_ol_wave"], mdl.tvalues["z_dl_hoop:z_ol_wave"]


rng = np.random.default_rng(0)
base_d = rep.dropna(subset=["z_dl_hoop", "z_ol_wave", "down", "ytg_b", "blitzing"]).copy()
dl_vals = base_d.drop_duplicates("dl_id").set_index("dl_id").z_dl_hoop
ol_vals = base_d.drop_duplicates("ol_id").set_index("ol_id").z_ol_wave
for y in ["dl_pressure", "ppa", "ol_pressure_allowed", "corner_turned"]:
    d = base_d.dropna(subset=[y])
    b, t = tstat(d, y)
    null = []
    for _ in range(500):
        dd = d.copy()
        dd["z_dl_hoop"] = dd.dl_id.map(pd.Series(rng.permutation(dl_vals.values), index=dl_vals.index))
        dd["z_ol_wave"] = dd.ol_id.map(pd.Series(rng.permutation(ol_vals.values), index=ol_vals.index))
        null.append(tstat(dd, y)[1])
    null = np.array(null)
    p = (1 + (np.abs(null) >= abs(t)).sum()) / 501
    print(f"{y:20s} interaction b {b:+.4f}  t {t:+.2f}  player-permutation p {p:.3f}  (null |t| 95th pct {np.percentile(np.abs(null), 95):.2f})")

print("\nsubsets / controls (interaction coefficient, two-way clustered p):")
for lab, d, extra in [("tackle vs edge", base_d[(base_d.ol_align == "T") & (base_d.dl_align == "EDGE")], ""),
                      ("interior (G/C vs IDL)", base_d[(base_d.dl_align == "INTERIOR_LINE") & base_d.ol_align.isin(["G", "C"])], ""),
                      ("+ both weights + OL arm", base_d.dropna(subset=["z_dl_w", "z_ol_w", "z_ol_arm"]), "+ z_dl_w + z_ol_w + z_ol_arm"),
                      ("+ weights + blocker 10-yd & rusher 10-yd", base_d.dropna(subset=["z_dl_w", "z_ol_w", "z_ol_ten", "z_dl_ten"]), "+ z_dl_w + z_ol_w + z_ol_ten + z_dl_ten"),
                      ("2023-24 seasons", base_d[base_d.season <= 2024], ""), ("2025 season", base_d[base_d.season == 2025], "")]:
    row = []
    for y in ["dl_pressure", "ppa", "ol_pressure_allowed"]:
        dd = d.dropna(subset=[y])
        g = np.column_stack([pd.factorize(dd.dl_id)[0], pd.factorize(dd.ol_id)[0]])
        m_ = smf.ols(f"{y} ~ z_dl_hoop * z_ol_wave {extra} + {CTXF}", data=dd).fit(cov_type="cluster", cov_kwds={"groups": g})
        row.append(f"{y} {m_.params['z_dl_hoop:z_ol_wave']:+.4f} (p {m_.pvalues['z_dl_hoop:z_ol_wave']:.3f})")
    print(f"  {lab:42s} reps {len(d):5d} | " + " | ".join(row))

# simple slopes: rusher bend effect on rusher pressure vs slow / average / quick blockers (terciles of blocker quickness)
print("\nrusher pressure rate by rusher bend tercile x blocker quickness tercile (REG+POST reps):")
d = base_d.copy()
d["rusher_bend"] = pd.qcut(d.z_dl_hoop, 3, labels=["low bend", "mid bend", "high bend"])
d["blocker_feet"] = pd.qcut(d.z_ol_wave, 3, labels=["slow feet", "mid feet", "quick feet"])
t = d.pivot_table(index="rusher_bend", columns="blocker_feet", values="dl_pressure", aggfunc="mean").round(3)
n = d.pivot_table(index="rusher_bend", columns="blocker_feet", values="dl_pressure", aggfunc="size")
print(t.to_string()); print("reps per cell:\n" + n.to_string())
print("\nblocker peak pressure probability by the same cells:")
print(d.pivot_table(index="rusher_bend", columns="blocker_feet", values="ppa", aggfunc="mean").round(3).to_string())
