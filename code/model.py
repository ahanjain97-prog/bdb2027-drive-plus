"""Masked-trajectory transformer: one context token + 80 frame tokens, reconstruct hidden kinematics."""
import math
import torch
import torch.nn as nn


class TrajEncoder(nn.Module):
    def __init__(self, n_kin, n_flag, n_target, cat_sizes, n_num, T=80, d=192, L=6, H=6, drop=0.1):
        super().__init__()
        self.T, self.d = T, d
        self.kin = nn.Linear(n_kin + 1, d)            # kinematics (+ orientation-missing flag)
        self.flag = nn.Linear(n_flag, d)              # time / event-phase channels, never masked
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        self.ctx_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.cat_emb = nn.ModuleList([nn.Embedding(n + 1, d) for n in cat_sizes])
        self.num = nn.Linear(n_num, d)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_target)
        for p in (self.mask_tok, self.pos, self.ctx_tok):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, kin, flags, valid, frame_mask, o_missing, cat, num):
        """kin [B,T,n_kin] standardized (orientation channels already zeroed where missing);
        flags [B,T,n_flag]; valid/frame_mask [B,T] bool; o_missing [B] bool; cat [B,C] long; num [B,K]."""
        B = kin.shape[0]
        om = o_missing[:, None, None].expand(B, self.T, 1).to(kin.dtype)
        h = self.kin(torch.cat([kin, om], -1))
        h = torch.where(frame_mask[..., None], self.mask_tok.expand_as(h), h)
        h = h + self.flag(flags) + self.pos
        c = self.ctx_tok.expand(B, 1, self.d) + self.num(num)[:, None]
        for j, emb in enumerate(self.cat_emb):
            c = c + emb(cat[:, j])[:, None]
        x = torch.cat([c, h], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=kin.device), ~valid], 1)
        z = self.norm(self.enc(x, src_key_padding_mask=pad))
        return self.head(z[:, 1:]), z


def span_mask(valid, gen, n_spans=6, min_len=3, max_len=12, p_forecast=0.25, snap_idx=10):
    """Random span masks plus, for a share of sequences, everything after a random cut (forecasting)."""
    B, T = valid.shape
    dev = valid.device
    pos = torch.arange(T, device=dev)[None, None]
    starts = torch.randint(0, T, (B, n_spans, 1), generator=gen, device=dev)
    lens = torch.randint(min_len, max_len + 1, (B, n_spans, 1), generator=gen, device=dev)
    m = ((pos >= starts) & (pos < starts + lens)).any(1)
    cut = torch.randint(snap_idx + 5, snap_idx + 40, (B, 1), generator=gen, device=dev)
    fc = torch.rand(B, 1, generator=gen, device=dev) < p_forecast
    m = torch.where(fc, pos[0] >= cut, m)
    m = m & valid
    # guarantee at least one masked frame and at least one visible frame per sequence
    none = ~m.any(1)
    if none.any():
        first_valid = valid.float().argmax(1)
        m[none, first_valid[none]] = True
    return m


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
