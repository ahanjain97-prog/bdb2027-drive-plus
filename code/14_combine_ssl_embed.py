"""Embed every combine drill attempt with the GAME-pretrained encoder (shared representation space).

Combine has no orientation and no play context, which the encoder was trained to tolerate (15% orientation
dropout with a missing flag, 15% context dropout). Each attempt is put in the game frame of reference:
  "snap" (index 10) = movement onset (first frame with s >= 1 yd/s); 10 frames before, 69 after
  rotated so the initial direction of travel (first 0.5 s after onset) points +x
  x_rel / y_rel relative to the onset position; y_field = 0 (field position is meaningless at the combine)
  vx, vy, s, a_long (= ds/dt), a_lat, a_mag as in 01_build_sequences; orientation channels zeroed + flag
  context = all UNK
Embedding = mean final-layer frame token, averaged with the left/right mirror (same as 03_embed).
Output data/blocks/combine_ssl_attempt.parquet (event_id, nfl_id, drill, 192 dims)
"""
import json
import numpy as np
import pandas as pd
import torch
from model import TrajEncoder

ROOT = "/Users/ahanjain/bdb_ssl"
PRE, POST = 10, 69
T = PRE + 1 + POST
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(f"{ROOT}/checkpoints/ssl_v1/best.pt", map_location="cpu", weights_only=False)
a = ck["args"]
kin_names, flag_names = ck["kin_names"], ck["flag_names"]
voc = json.load(open(f"{ROOT}/data/seq/vocab.json"))
model = TrajEncoder(len(kin_names), len(flag_names), len(kin_names), ck["cat_sizes"], len(voc["num"]),
                    d=a["d"], L=a["layers"], H=a["heads"]).to(dev)
model.load_state_dict(ck["model"]); model.eval()
mu = torch.tensor(ck["mu"], device=dev); sd = torch.tensor(ck["sd"], device=dev)
o_sel = torch.tensor([k in voc["o_feats"] for k in kin_names], device=dev)
flip = {"y_rel", "y_field", "vy", "a_lat", "cos_o", "sin_od", "o_rate"}
msign = torch.tensor([-1.0 if k in flip else 1.0 for k in kin_names], device=dev)

ct = pd.read_parquet(f"{ROOT}/data/raw/combine_tracking.parquet")
ct = ct[ct.entity_type == "PLAYER"].sort_values(["event_id", "time"])
ct["drill"] = ct.drill_type + ":" + ct.drill_name

kins, flags, valids, meta = [], [], [], []
for eid, g in ct.groupby("event_id"):
    s = g.s.values.astype(float)
    if len(s) < 8 or s.max() < 1.0:
        continue
    t = (g.time - g.time.iat[0]).dt.total_seconds().values
    th = np.radians(g.dir.values.astype(float))
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    on = int(np.argmax(s >= 1.0))
    vx0, vy0 = s * np.sin(th), s * np.cos(th)
    w = slice(on, min(on + 5, len(s)))
    ang = np.arctan2(vy0[w].mean(), vx0[w].mean())          # direction of travel after onset (math angle)
    c, sn = np.cos(-ang), np.sin(-ang)                       # rotate it onto +x
    xr = c * (x - x[on]) - sn * (y - y[on]); yr = sn * (x - x[on]) + c * (y - y[on])
    vx, vy = c * vx0 - sn * vy0, sn * vx0 + c * vy0
    along = np.gradient(s, t)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    lat = np.where(s > 0.5, (vx * ay - vy * ax) / np.maximum(s, 1e-6), 0.0)
    k = np.round((t - t[on]) * 10).astype(int)
    keep = (k >= -PRE) & (k <= POST)
    feat = {"x_rel": xr, "y_rel": yr, "y_field": np.zeros_like(s), "vx": vx, "vy": vy, "s": s,
            "a_long": along, "a_lat": lat, "a_mag": g.a.values.astype(float)}
    K = np.zeros((T, len(kin_names)), np.float32); V = np.zeros(T, bool); Fl = np.zeros((T, len(flag_names)), np.float32)
    for j, nm in enumerate(kin_names):
        if nm in feat:
            K[k[keep] + PRE, j] = feat[nm][keep]
    V[k[keep] + PRE] = True
    Fl[:, flag_names.index("t_snap")] = (np.arange(T) - PRE) / 10.0
    Fl[:, flag_names.index("pre_snap")] = (np.arange(T) < PRE).astype(float)
    kins.append(K); flags.append(Fl); valids.append(V)
    meta.append({"event_id": eid, "nfl_id": g.nfl_id.iat[0], "drill": g.drill.iat[0], "n_valid": int(V.sum())})

Kt = torch.tensor(np.stack(kins), device=dev); Ft = torch.tensor(np.stack(flags), device=dev)
Vt = torch.tensor(np.stack(valids), device=dev)
N = len(meta)
emb = np.zeros((N, a["d"]), np.float32)
with torch.no_grad():
    for lo in range(0, N, 512):
        sl = slice(lo, min(lo + 512, N))
        v = Vt[sl]; B = v.shape[0]
        acc = 0
        for mirror in (False, True):
            raw = Kt[sl] * (msign if mirror else 1.0)
            kin = ((raw - mu) / sd).clamp(-6, 6) * v[..., None]
            kin = kin.masked_fill(o_sel[None, None], 0.0)      # orientation missing at the combine
            cat = torch.zeros(B, len(ck["cat_sizes"]), dtype=torch.long, device=dev)
            num = torch.zeros(B, len(voc["num"]), device=dev)
            om = torch.ones(B, dtype=torch.bool, device=dev)
            _, z = model(kin, Ft[sl], v, torch.zeros_like(v), om, cat, num)
            vf = v[..., None].float()
            acc = acc + (z[:, 1:] * vf).sum(1) / vf.sum(1).clamp_min(1)
        emb[sl] = (acc / 2).cpu().numpy()
out = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
out = pd.concat([pd.DataFrame(meta), out], axis=1)
out.to_parquet(f"{ROOT}/data/blocks/combine_ssl_attempt.parquet", index=False)
print("combine attempts embedded:", N, "players", out.nfl_id.nunique(), "finite", np.isfinite(emb).all())
print(out.groupby("drill").size().sort_values().tail(5).to_string())
