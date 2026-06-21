"""Train the CNN-BiLSTM hero (PyTorch) and evaluate it + the seasonal-naive baseline.

Strategy: the hero is trained once on the default region in *normalised* space.
Because every region is z-scored by its own scaler, the normalised dynamics are
shared, so the single model generalises across regions (we de-normalise with the
per-region scaler at inference). This keeps CPU training to a few minutes while
supporting the multi-region selector and in-browser what-if.

This script is intentionally torch-only (no XGBoost) — XGBoost lives in
baselines.py because mixing the two OpenMP runtimes deadlocks on macOS.

Pipeline order:  data.py -> train.py -> baselines.py -> export.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn

from config import ARTIFACTS, DEFAULT_REGION, REGIONS, SEED
from metrics_utils import metrics, per_horizon_mae
from models import MBDLSTM
from pipeline import build_samples

torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 18
BATCH = 512
STRIDE = 1          # pooled windows from all regions (every window)
LR = 1.5e-3


def train_hero(train_samples):
    """Train one CNN-BiLSTM on the pooled (normalised) windows of all regions.

    Pooling 3 regions of z-scored data triples the data and regularises the
    shared model — empirically better per region than per-region specialists."""
    model = MBDLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.SmoothL1Loss()

    Xtr = torch.from_numpy(np.concatenate(
        [s.X_seq[s.train][::STRIDE] for s in train_samples]))
    ytr = torch.from_numpy(np.concatenate(
        [s.y_norm[s.train][::STRIDE] for s in train_samples]))
    n = len(Xtr)
    print(f"  pooled train samples: {n} (stride {STRIDE})")

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"  epoch {ep + 1:2d}/{EPOCHS}  loss {total / n:.4f}")
    return model


@torch.no_grad()
def hero_predict_mw(model, samp, sl):
    model.eval()
    out = model(torch.from_numpy(samp.X_seq[sl])).numpy()
    return out * samp.scaler.load_std + samp.scaler.load_mean


def main():
    t0 = time.time()
    print("Building samples for all regions…")
    samples = {r: build_samples(r) for r in REGIONS}

    model = train_hero(list(samples.values()))
    torch.save(model.state_dict(), ARTIFACTS / "model.pt")
    print(f"  saved checkpoint -> {ARTIFACTS / 'model.pt'}")

    eval_hero = {}
    for region in REGIONS:
        samp = samples[region]
        ytest = samp.y_mw[samp.test]
        hero_pred = hero_predict_mw(model, samp, samp.test)
        naive_pred = samp.naive_mw[samp.test]
        eval_hero[region] = {
            "label": REGIONS[region]["label"],
            "models": {
                "CNN-BiLSTM": metrics(ytest, hero_pred),
                "Seasonal-Naive": metrics(ytest, naive_pred),
            },
            "per_horizon_mae": {
                "CNN-BiLSTM": per_horizon_mae(ytest, hero_pred),
                "Seasonal-Naive": per_horizon_mae(ytest, naive_pred),
            },
        }
        h = eval_hero[region]["models"]["CNN-BiLSTM"]
        print(f"  [{region}] CNN-BiLSTM MAPE {h['mape']:.2f}%  R2 {h['r2']:.3f}")

    with open(ARTIFACTS / "eval_hero.json", "w") as f:
        json.dump(eval_hero, f, indent=2)
    print(f"  saved -> {ARTIFACTS / 'eval_hero.json'}")
    print(f"Done in {time.time() - t0:.1f}s.  Next: python baselines.py")


if __name__ == "__main__":
    main()
