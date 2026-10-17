"""Per-play outputs of the pretrained encoder.

ssl_emb.npy      [N, d] mean of final-layer frame tokens on the full (unmasked) sequence, context visible,
                 averaged with the same play mirrored left/right (y -> -y, LEFT<->RIGHT assignments) so the
                 embedding cannot encode which side of the formation the player is on (v1 did: OL/DL/DB PC1-2)
ssl_surprise.npy [N, 4] masked-reconstruction loss (2 fixed random masks) by channel group:
                 position, velocity, acceleration, orientation -> how predictable the player's movement is
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from data import SeqData
from model import TrajEncoder, span_mask

name = sys.argv[1] if len(sys.argv) > 1 else "ssl_v1"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(f"/Users/ahanjain/bdb_ssl/checkpoints/{name}/best.pt", map_location="cpu", weights_only=False)
a = ck["args"]
D = SeqData(dev)
assert np.allclose(D.mu, ck["mu"]) and np.allclose(D.sd, ck["sd"]), "standardization drifted from training"
nk, nf = len(D.kin_names), len(D.flag_names)
model = TrajEncoder(nk, nf, nk, D.cat_sizes, D.num.shape[1], d=a["d"], L=a["layers"], H=a["heads"]).to(dev)
model.load_state_dict(ck["model"]); model.eval()
print("loaded", name, "epoch", ck["epoch"], flush=True)

groups = {"position": ["x_rel", "y_rel", "y_field"], "velocity": ["vx", "vy", "s"],
          "accel": ["a_long", "a_lat", "a_mag"], "orientation": ["sin_o", "cos_o", "sin_od", "cos_od", "o_rate"]}
gidx = {g: [D.kin_names.index(c) for c in cs] for g, cs in groups.items()}
N = len(D.meta)
emb = np.zeros((N, a["d"]), dtype=np.float16)
sur = np.zeros((N, len(groups)), dtype=np.float32)
gen = torch.Generator(device=dev); gen.manual_seed(7)
BS = 1024
with torch.no_grad():
    for lo in range(0, N, BS):
        idx = np.arange(lo, min(lo + BS, N))
        e_sum, acc = 0.0, 0.0
        for mirror in (False, True):
            kin, flags, valid, cat, num = D.batch(idx, mirror=mirror)
            B = kin.shape[0]
            nomask = torch.zeros_like(valid)
            om = torch.zeros(B, dtype=torch.bool, device=dev)
            _, z = model(kin, flags, valid, nomask, om, cat, num)
            v = valid[..., None].float()
            e_sum = e_sum + (z[:, 1:] * v).sum(1) / v.sum(1).clamp_min(1)
            for rep in range(2):
                fm = span_mask(valid, gen)
                pred, _ = model(kin, flags, valid, fm, om, cat, num)
                l = F.smooth_l1_loss(pred, kin, reduction="none", beta=1.0) * fm[..., None]
                acc = acc + torch.stack([l[..., gidx[g]].sum((1, 2)) / (fm.sum(1).clamp_min(1) * len(gidx[g]))
                                         for g in groups], 1)
        emb[idx] = (e_sum / 2).cpu().numpy().astype(np.float16)
        sur[idx] = (acc / 4).cpu().numpy()
        if (lo // BS) % 50 == 0:
            print(f"  {lo + B}/{N}", flush=True)
np.save("/Users/ahanjain/bdb_ssl/data/blocks/ssl_emb.npy", emb)
np.save("/Users/ahanjain/bdb_ssl/data/blocks/ssl_surprise.npy", sur)
json.dump(list(groups), open("/Users/ahanjain/bdb_ssl/data/blocks/ssl_surprise_names.json", "w"))
print("emb", emb.shape, "finite", np.isfinite(emb.astype(np.float32)).all(), "surprise mean", sur.mean(0).round(3))
