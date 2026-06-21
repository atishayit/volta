"""Forecast metrics — torch-free so the XGBoost baseline can import it safely."""
from __future__ import annotations

import numpy as np


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt, yp = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.clip(np.abs(yt), 1e-6, None)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def per_horizon_mae(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """MAE at each step ahead (1..HORIZON), expects [N, HORIZON] arrays."""
    diff = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    return [float(x) for x in diff.mean(axis=0)]
