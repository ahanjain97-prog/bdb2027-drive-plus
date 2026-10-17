"""Hand-built per-play physics features: the fair baseline the learned embedding has to beat.

Same sequences, same frame of reference as the SSL model. Post-snap = frames 10.. (k >= 0).
Output data/blocks/phys_play.npy [N, P] float32 + names in phys_names.json
"""
import json, os
import numpy as np

SEQ = "/Users/ahanjain/bdb_ssl/data/seq"
OUT = "/Users/ahanjain/bdb_ssl/data/blocks"
os.makedirs(OUT, exist_ok=True)
voc = json.load(open(f"{SEQ}/vocab.json"))
F = voc["feats"]; PRE = voc["pre"]
X = np.load(f"{SEQ}/X.npy", mmap_mode="r")
V = np.load(f"{SEQ}/valid.npy")
N, T, _ = X.shape
ch = lambda name: np.asarray(X[:, :, F.index(name)], dtype=np.float32)

post = V.copy(); post[:, :PRE] = False
pre = V.copy(); pre[:, PRE:] = False
s, al, alat, am = ch("s"), ch("a_long"), ch("a_lat"), ch("a_mag")
cod, sod, orate, xr, yr = ch("cos_od"), ch("sin_od"), ch("o_rate"), ch("x_rel"), ch("y_rel")


def masked(a, m, fn, fill=np.nan):
    return fn(np.where(m, a, fill), axis=1)


def window(lo, hi):
    m = post.copy(); m[:, :PRE + lo] = False; m[:, PRE + hi:] = False
    return m


out = {}
out["s_max"] = masked(s, post, np.nanmax)
out["s_mean_0_3s"] = masked(s, window(0, 30), np.nanmean)
out["dist_1s"] = np.where(window(0, 10), s, 0).sum(1) * 0.1
out["dist_2s"] = np.where(window(0, 20), s, 0).sum(1) * 0.1
out["accel_max"] = masked(al, post, np.nanmax)
out["decel_max"] = -masked(al, post, np.nanmin)
out["alat_max"] = masked(np.abs(alat), post, np.nanmax)
out["amag_p90"] = np.nanpercentile(np.where(post, am, np.nan), 90, axis=1)
# time from snap to first frame at >= 90% of the play's top speed
thr = out["s_max"][:, None] * 0.9
hit = post & (s >= thr)
out["t_to_90pct"] = np.where(hit.any(1), (hit.argmax(1) - PRE) / 10.0, np.nan)
moving = post & (s > 1.0)
nm = moving.sum(1).clip(1)
out["backpedal_share"] = (moving & (cod < -0.5)).sum(1) / nm
out["sideways_share"] = (moving & (np.abs(sod) > 0.7)).sum(1) / nm
out["o_rate_p90"] = np.nanpercentile(np.where(post, np.abs(orate), np.nan), 90, axis=1)
out["x_rel_snap"] = xr[:, PRE]
i2 = PRE + 20
ok2 = V[:, i2]
out["dx_2s"] = np.where(ok2, xr[:, i2] - xr[:, PRE], np.nan)
out["abs_dy_2s"] = np.where(ok2, np.abs(yr[:, i2] - yr[:, PRE]), np.nan)
out["pre_snap_speed"] = masked(s, pre, np.nanmean)

names = list(out)
P = np.stack([out[k] for k in names], 1).astype(np.float32)
np.save(f"{OUT}/phys_play.npy", P)
json.dump(names, open(f"{OUT}/phys_names.json", "w"))
print("phys_play", P.shape)
for k, col in zip(names, P.T):
    print(f"{k:16s} nan {np.isnan(col).mean():.3f}  p10 {np.nanpercentile(col,10):7.2f}  p50 {np.nanpercentile(col,50):7.2f}  p90 {np.nanpercentile(col,90):7.2f}")
