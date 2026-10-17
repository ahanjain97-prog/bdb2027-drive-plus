"""Load sequence tensors, standardize with train statistics, serve batches on device."""
import json
import numpy as np
import pandas as pd
import torch

SEQ = "/Users/ahanjain/bdb_ssl/data/seq"
CLIP = 6.0


class SeqData:
    def __init__(self, device, data_device=None):
        data_device = data_device or device
        self.voc = json.load(open(f"{SEQ}/vocab.json"))
        feats = self.voc["feats"]
        self.kin_names = self.voc["target_feats"]                      # 14 reconstructable channels
        self.flag_names = [f for f in feats if f not in self.kin_names]  # t_snap, pre_snap, post_* (always visible)
        self.o_idx = [self.kin_names.index(f) for f in self.voc["o_feats"]]
        self.meta = pd.read_parquet(f"{SEQ}/meta.parquet")
        X = np.load(f"{SEQ}/X.npy", mmap_mode="r")
        valid = np.load(f"{SEQ}/valid.npy")
        ki = [feats.index(f) for f in self.kin_names]
        fi = [feats.index(f) for f in self.flag_names]
        tr = (self.meta.split == "train").values
        # train-only standardization over valid frames (subsample for speed)
        rs = np.random.default_rng(0).choice(np.where(tr)[0], 40000, replace=False)
        rs.sort()
        sub = np.asarray(X[rs])[valid[rs]].astype(np.float32)
        self.mu = sub[:, ki].mean(0)
        self.sd = sub[:, ki].std(0) + 1e-6
        self.device = device
        self.kin = torch.from_numpy(np.asarray(X[:, :, ki])).to(data_device)     # fp16, raw units
        self.flags = torch.from_numpy(np.asarray(X[:, :, fi])).to(data_device)
        self.valid = torch.from_numpy(valid).to(data_device)
        self.cat = torch.from_numpy(np.load(f"{SEQ}/ctx_cat.npy").astype(np.int64)).to(data_device)
        self.num = torch.from_numpy(np.load(f"{SEQ}/ctx_num.npy")).to(data_device)
        self.mu_t = torch.tensor(self.mu, device=device)
        self.sd_t = torch.tensor(self.sd, device=device)
        self.cat_sizes = [len(self.voc["vocab"][c]) for c in self.voc["cat"]]
        self.train_idx = np.where(tr)[0]
        self.val_idx = np.where(~tr)[0]
        # left/right mirror (y -> -y): channels whose sign flips; compass angles theta -> 180 - theta
        flip = {"y_rel", "y_field", "vy", "a_lat", "cos_o", "sin_od", "o_rate"}
        self.mirror_sign = torch.tensor([-1.0 if f in flip else 1.0 for f in self.kin_names], device=device)
        cov = self.voc["vocab"]["coverage_assignment"]
        swap = np.arange(len(cov) + 1)
        for name, code in cov.items():
            other = name.replace("LEFT", "\0").replace("RIGHT", "LEFT").replace("\0", "RIGHT")
            swap[code] = cov.get(other, code)
        self.cov_col = self.voc["cat"].index("coverage_assignment")
        self.cov_swap = torch.as_tensor(swap, device=device)

    def batch(self, idx, mirror=False):
        i = torch.as_tensor(np.asarray(idx), device=self.kin.device)
        dv = self.device
        v = self.valid[i].to(dv)
        raw = self.kin[i].to(dv).float()
        cat = self.cat[i].to(dv)
        if mirror:
            raw = raw * self.mirror_sign
            cat = cat.clone()
            cat[:, self.cov_col] = self.cov_swap[cat[:, self.cov_col]]
        k = ((raw - self.mu_t) / self.sd_t).clamp(-CLIP, CLIP)
        k = k * v[..., None]
        return k, self.flags[i].to(dv).float(), v, cat, self.num[i].to(dv)
