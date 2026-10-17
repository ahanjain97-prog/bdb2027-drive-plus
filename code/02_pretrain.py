"""Self-supervised pretraining: masked-span + forecast reconstruction of player trajectories.

No outcome columns are used anywhere in training. Validation = held-out games (10% by hash).
Reference baseline printed once: linear interpolation / carry-forward from the nearest visible frames.

Run with MPS watermark limits on the command line, e.g.
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 python code/02_pretrain.py --epochs 20
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from data import SeqData
from model import TrajEncoder, span_mask, cosine_lr

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=192)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--p_o_drop", type=float, default=0.15)
ap.add_argument("--p_ctx_drop", type=float, default=0.15)
ap.add_argument("--max_steps", type=int, default=0, help="stop early (speed test)")
ap.add_argument("--name", default="ssl_v1")
ap.add_argument("--amp", type=int, default=0, help="fp16 autocast + GradScaler")
args = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)
D = SeqData(dev)
nk, nf = len(D.kin_names), len(D.flag_names)
model = TrajEncoder(nk, nf, nk, D.cat_sizes, D.num.shape[1], d=args.d, L=args.layers, H=args.heads).to(dev)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  train {len(D.train_idx)}  val {len(D.val_idx)}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98))
o_idx = torch.tensor(D.o_idx, device=dev)
o_sel = torch.zeros(nk, dtype=torch.bool, device=dev); o_sel[o_idx] = True
CK = f"/Users/ahanjain/bdb_ssl/checkpoints/{args.name}"
os.makedirs(CK, exist_ok=True)


def prep(idx, gen, train):
    kin, flags, valid, cat, num = D.batch(idx)
    B = kin.shape[0]
    fm = span_mask(valid, gen)
    o_missing = torch.rand(B, generator=gen, device=dev) < args.p_o_drop
    if train:
        drop = torch.rand(cat.shape, generator=gen, device=dev) < args.p_ctx_drop
        drop |= (torch.rand(B, 1, generator=gen, device=dev) < 0.1)
        cat = cat.masked_fill(drop, 0)
        num = num.masked_fill(drop[:, :1], 0.0)
    target = kin
    kin_in = kin.masked_fill(o_missing[:, None, None] & o_sel[None, None], 0.0)
    w = fm[..., None].float() * torch.where(o_missing[:, None, None] & o_sel[None, None], 0.0, 1.0)
    return kin_in, flags, valid, fm, o_missing, cat, num, target, w


def loss_fn(pred, target, w):
    l = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0) * w
    per_feat = l.sum((0, 1)) / w.sum((0, 1)).clamp_min(1)
    return per_feat.mean(), per_feat


def interp_baseline(target, valid, fm, w):
    """Linear interpolation between nearest visible valid frames; carry nearest at the edges."""
    B, T, _ = target.shape
    vis = valid & ~fm
    ar = torch.arange(T, device=dev)[None].expand(B, T)
    prev = torch.where(vis, ar, torch.full_like(ar, -1)).cummax(1).values
    nxt = torch.where(vis, ar, torch.full_like(ar, T)).flip(1).cummin(1).values.flip(1)
    has_p, has_n = prev >= 0, nxt < T
    pi, ni = prev.clamp(0, T - 1), nxt.clamp(0, T - 1)
    tp = torch.gather(target, 1, pi[..., None].expand_as(target))
    tn = torch.gather(target, 1, ni[..., None].expand_as(target))
    frac = ((ar - prev).float() / (nxt - prev).clamp_min(1).float())[..., None]
    both = (has_p & has_n)[..., None]
    pred = torch.where(both, tp + frac * (tn - tp), torch.where(has_p[..., None], tp, tn))
    return loss_fn(pred, target, w)


@torch.no_grad()
def evaluate(n_max=20000):
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    idx = D.val_idx[:n_max]
    tot, per, base_tot, base_per, nb = 0.0, 0.0, 0.0, 0.0, 0
    for lo in range(0, len(idx), 1024):
        kin_in, flags, valid, fm, om, cat, num, target, w = prep(idx[lo:lo + 1024], gen, train=False)
        pred, _ = model(kin_in, flags, valid, fm, om, cat, num)
        l, pf = loss_fn(pred, target, w)
        bl, bpf = interp_baseline(target, valid, fm, w)
        tot += l.item(); per = per + pf; base_tot += bl.item(); base_per = base_per + bpf; nb += 1
    model.train()
    return tot / nb, (per / nb).cpu().numpy(), base_tot / nb, (base_per / nb).cpu().numpy()


steps_per_epoch = len(D.train_idx) // args.bs
total = steps_per_epoch * args.epochs if not args.max_steps else args.max_steps
gen = torch.Generator(device=dev); gen.manual_seed(0)
rng = np.random.default_rng(0)
step, t0, log = 0, time.time(), []
scaler = torch.amp.GradScaler("mps", enabled=bool(args.amp))
best = float("inf")
for ep in range(args.epochs):
    perm = rng.permutation(D.train_idx)
    run = 0.0
    for b in range(steps_per_epoch):
        lr = cosine_lr(step, total, 500, args.lr)
        for g in opt.param_groups: g["lr"] = lr
        kin_in, flags, valid, fm, om, cat, num, target, w = prep(perm[b * args.bs:(b + 1) * args.bs], gen, train=True)
        with torch.autocast("mps", dtype=torch.float16, enabled=bool(args.amp)):
            pred, _ = model(kin_in, flags, valid, fm, om, cat, num)
        loss, _ = loss_fn(pred.float(), target, w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run += loss.item() if b % 50 == 0 else 0.0
        step += 1
        if step % 200 == 0:
            el = time.time() - t0
            print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} gn {gn.item():.2f} lr {lr:.2e} "
                  f"{step/el:.1f} it/s  eta {(total-step)/(step/el)/60:.1f} min", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    vl, vper, bl, bper = evaluate()
    rec = {"epoch": ep, "step": step, "val": vl, "interp_baseline": bl,
           "per_feat": dict(zip(D.kin_names, np.round(vper, 4).tolist())),
           "per_feat_interp": dict(zip(D.kin_names, np.round(bper, 4).tolist())), "min": (time.time() - t0) / 60}
    log.append(rec)
    print(f"== epoch {ep} val {vl:.4f} (interp baseline {bl:.4f})", flush=True)
    print("   per-feat model ", rec["per_feat"], flush=True)
    if ep == 0:
        print("   per-feat interp", rec["per_feat_interp"], flush=True)
    state = {"model": model.state_dict(), "args": vars(args), "mu": D.mu.tolist(), "sd": D.sd.tolist(),
             "kin_names": D.kin_names, "flag_names": D.flag_names, "cat_sizes": D.cat_sizes, "epoch": ep}
    torch.save(state, f"{CK}/last.pt")
    if vl < best:
        best = vl
        torch.save(state, f"{CK}/best.pt")
    json.dump(log, open(f"{CK}/log.json", "w"), indent=1)
    if args.max_steps and step >= args.max_steps:
        break
print("done; best val", best)
