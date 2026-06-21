"""XGBoost baseline — torch-free on purpose.

Run in its own process (no `import torch` anywhere in its import graph) to avoid
the PyTorch/XGBoost dual-OpenMP deadlock on macOS. Trains a multi-output XGBoost
to forecast all HORIZON hours from flat window-summary features and writes
per-region metrics for export.py to merge into the scorecard.

Run after train.py:  python baselines.py
"""
from __future__ import annotations

import json
import time

from xgboost import XGBRegressor

from config import ARTIFACTS, REGIONS, SEED
from metrics_utils import metrics, per_horizon_mae
from pipeline import build_samples


def train_xgb(samp):
    # Joint multi-output XGBoost: one model predicting the whole HORIZON vector at
    # once (multi_output_tree) — the structural match to our seq2seq CNN-BiLSTM,
    # which also emits all HORIZON steps jointly. (Training 24 independent per-step
    # boosters is a different, decoupled setup; we compare like-for-like.)
    reg = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        multi_strategy="multi_output_tree", random_state=SEED,
    )
    reg.fit(samp.tab_X[samp.train], samp.y_mw[samp.train])
    return reg


def main():
    t0 = time.time()
    eval_xgb = {}
    for region in REGIONS:
        samp = build_samples(region)
        ytest = samp.y_mw[samp.test]
        reg = train_xgb(samp)
        pred = reg.predict(samp.tab_X[samp.test])
        eval_xgb[region] = {
            "metrics": metrics(ytest, pred),
            "per_horizon_mae": per_horizon_mae(ytest, pred),
        }
        m = eval_xgb[region]["metrics"]
        print(f"  [{region}] XGBoost MAPE {m['mape']:.2f}%  R2 {m['r2']:.3f}")

    with open(ARTIFACTS / "eval_xgb.json", "w") as f:
        json.dump(eval_xgb, f, indent=2)
    print(f"  saved -> {ARTIFACTS / 'eval_xgb.json'}")
    print(f"Done in {time.time() - t0:.1f}s.  Next: python export.py")


if __name__ == "__main__":
    main()
