"""Which drills carry the completion-over-expected information in the year-comparable (v2) features?
Ridge on one drill's v2 features at a time (+ the transition drill's turn-anchored curves alone), draft class in the
context where estimable; leave-one-draft-class-out and new-prospect splits; CB-permutation null (100)."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/ahanjain/bdb_ssl"
ns = {}; exec(open(f"{ROOT}/code/67b_cb_full_models_v2.py").read().split("OUTS = [")[0], ns)
cov, Vw0, CBS = ns["cov"], ns["Vw"], ns["CBS"]
rng = np.random.default_rng(41)
y, ctx = "cpoe", "rec_target_prop + C(team_coverage_man_zone) + C(cls)"
d = cov.dropna(subset=[y]).copy()
splits = {"draft-class-out": lambda d: [((d.cls != c).values, (d.cls == c).values) for c in sorted(d.cls.unique())],
          "new-prospect": lambda d: [((d.cls <= 2024).values, (d.cls == 2025).values)]}
drills = sorted({c.split("|")[0] for c in Vw0.columns if "|" in c})
blocks = {dr: [c for c in Vw0.columns if c.startswith(dr + "|")] for dr in drills}
blocks["TRANSITION turn-anchored curves"] = [c for c in Vw0.columns if "|turn_" in c]
blocks["TRANSITION real-time curves"] = [c for c in Vw0.columns if c.startswith("BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION|rt_")]
rows = []
for name, cols in blocks.items():
    if len(cols) < 2:
        continue
    for sname, sp in splits.items():
        ns["FE"] = cols; ns["Vw"] = Vw0[["nfl_id", "view"] + cols]
        obs = ns["run_split"](d, y, ctx, sp(d), ["ridge_all"])["ridge_all"]
        null = []
        for _ in range(100):
            perm = dict(zip(CBS, rng.permutation(CBS)))
            ns["Vw"] = Vw0[["nfl_id", "view"] + cols].assign(nfl_id=Vw0.nfl_id.map(lambda k: perm.get(k, k)))
            null.append(ns["run_split"](d, y, ctx, sp(d), ["ridge_all"])["ridge_all"][0])
        rows.append({"block": name, "split": sname, "n_features": len(cols), "oos_mse_reduction_%": obs[0], "cb_effect_r": obs[1],
                     "perm_p": (1 + (np.array(null) >= obs[0]).sum()) / 101})
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/68b_cpoe_drills_v2.csv", index=False)
pd.set_option("display.width", 200)
print(R.pivot_table(index="block", columns="split", values=["oos_mse_reduction_%", "perm_p"]).round(3).sort_values(("oos_mse_reduction_%", "draft-class-out"), ascending=False).to_string())
