"""3D CNN on the striatal SBR crops, 5-fold CV, compared + ensembled with the feature baseline.

  python train_cnn.py --folds 1 --epochs 5      # smoke test
  python train_cnn.py                           # full 5-fold CV
  python train_cnn.py --seeds 3                 # 3 seeds per fold, averaged

No early stopping / model selection on the held-out fold: fixed schedule, so the CV score is honest.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import SIDE_AGNOSTIC

OUT = Path(__file__).resolve().parent / "output"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_data():
    feats = pd.read_csv(OUT / "features.csv")  # written by baseline.py; same uid order as manifest
    crops = np.stack([np.load(OUT / "crops" / f"{u}.npy") for u in feats.uid])  # (N, x, y, z)
    # -> (N, 1, D=z, H=y, W=x) so grid_sample's (x, y, z) grid axes = (L-R, P-A, I-S)
    X = torch.from_numpy(crops).permute(0, 3, 2, 1).unsqueeze(1).contiguous()
    y = torch.from_numpy(feats.label.to_numpy(np.float32))
    return feats, X, y


# ---------------------------------------------------------------- augmentation (on GPU)
def augment(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    dev = x.device
    # random L-R flip: disease side varies by patient, anatomy is ~symmetric
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None, None], x.flip(-1), x)

    # random rigid-ish transform: axial rotation +/-15 deg (head tilt), small pitch/roll +/-5 deg,
    # translation +/-4 voxels, isotropic scale 0.92-1.08 (head size)
    def u(a, b):
        return torch.empty(n, device=dev).uniform_(a, b)

    yaw, pitch, roll = u(-15, 15) * math.pi / 180, u(-5, 5) * math.pi / 180, u(-5, 5) * math.pi / 180
    s = u(0.92, 1.08)
    cz, sz = torch.cos(yaw), torch.sin(yaw)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cx, sx = torch.cos(roll), torch.sin(roll)
    o, z = torch.ones(n, device=dev), torch.zeros(n, device=dev)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(n, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(n, 3, 3)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(n, 3, 3)
    R = (Rz @ Ry @ Rx) * s[:, None, None]
    D, H, W = x.shape[2:]
    t = torch.stack([u(-4, 4) * 2 / W, u(-4, 4) * 2 / H, u(-3, 3) * 2 / D], 1)
    theta = torch.cat([R, t[:, :, None]], 2)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # intensity: global SBR scale (site/reference variability) + mild noise
    x = x * u(0.85, 1.15)[:, None, None, None, None]
    x = x + 0.05 * torch.randn_like(x)
    return x


# ---------------------------------------------------------------- model
def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, w=16, drop=0.3):
        super().__init__()
        self.f = nn.Sequential(
            block(1, w), nn.MaxPool3d(2),            # 32x48x48 -> 16x24x24
            block(w, 2 * w), nn.MaxPool3d(2),        # -> 8x12x12
            block(2 * w, 4 * w), nn.MaxPool3d(2),    # -> 4x6x6
            block(4 * w, 8 * w),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(8 * w, 1))

    def forward(self, x):
        return self.head(self.f(x)).squeeze(1)


# ---------------------------------------------------------------- train / predict
@torch.no_grad()
def predict(model, X, bs=128):
    model.eval()
    ps = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(DEV, non_blocking=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            logit = (model(xb) + model(xb.flip(-1))) / 2  # TTA: L-R flip
        ps.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(ps).numpy()


def train_fold(Xtr, ytr, Xte, yte, args, seed, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(w=args.width, drop=args.dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(Xtr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    Xtr_d, ytr_d = Xtr.to(DEV), ytr.to(DEV)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_d), device=DEV)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            if len(idx) < 2:
                continue
            xb, yb = augment(Xtr_d[idx]), ytr_d[idx]
            yb_s = yb * (1 - args.smooth) + 0.5 * args.smooth
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(xb).float(), yb_s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == args.epochs - 1:
            p = predict(model, Xte)  # printed for monitoring only, never used to pick a model
            print(f"{log_prefix} ep {ep + 1:3d}  train_loss {tot / len(Xtr):.3f}  "
                  f"heldout auc {roc_auc_score(yte, p):.3f}  logloss {log_loss(yte, p.clip(1e-4, 1 - 1e-4)):.3f}",
                  flush=True)
    return model


def metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p), "acc": accuracy_score(y, p > 0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="run only the first N of 5 folds")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--tag", default="cnn")
    args = ap.parse_args()
    print(vars(args), "| device:", DEV)

    feats, X, y = load_data()
    yn = y.numpy().astype(int)
    print(f"data {tuple(X.shape)}, params {sum(p.numel() for p in Net(args.width).parameters()) / 1e6:.2f}M")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cnn = np.full(len(yn), np.nan)
    oof_lr = np.full(len(yn), np.nan)
    Xf = feats[SIDE_AGNOSTIC].to_numpy(float)
    t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(Xf, yn)):
        if k >= args.folds:
            break
        ps = []
        for s in range(args.seeds):
            m = train_fold(X[tr], y[tr], X[te], yn[te], args, seed=1000 * k + s, log_prefix=f"[fold {k} seed {s}]")
            ps.append(predict(m, X[te]))
        oof_cnn[te] = np.mean(ps, 0)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xf[tr], yn[tr])
        oof_lr[te] = lr.predict_proba(Xf[te])[:, 1]
        print(f"fold {k}: CNN {metrics(yn[te], oof_cnn[te])['auc']:.3f} | "
              f"logreg {metrics(yn[te], oof_lr[te])['auc']:.3f} | {time.time() - t0:.0f}s", flush=True)

    done = ~np.isnan(oof_cnn)
    yy = yn[done]
    rows = {"CNN": oof_cnn[done], "logreg features": oof_lr[done],
            "ensemble (mean prob)": (oof_cnn[done] + oof_lr[done]) / 2}
    print(f"\n=== out-of-fold over {done.sum()} scans ===")
    print(pd.DataFrame({k: metrics(yy, v) for k, v in rows.items()}).T.round(3).to_string())
    pd.DataFrame({"uid": feats.uid[done], "label": yy, "cnn": oof_cnn[done], "logreg": oof_lr[done]}) \
        .to_csv(OUT / f"oof_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
