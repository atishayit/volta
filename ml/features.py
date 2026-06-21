"""Feature engineering for VOLTA.

Two shapes of features from the same source frame:
  * sequence tensors  -> CNN-BiLSTM  ([N, WINDOW, N_FEATURES] -> [N, HORIZON])
  * flat tabular rows  -> XGBoost baseline (calendar + lags + rolling stats)

Normalisation stats (per region) are returned so train/export and the in-browser
what-if simulator all de/normalise consistently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FEATURES, HORIZON, WINDOW


@dataclass
class Scaler:
    load_mean: float
    load_std: float
    temp_mean: float
    temp_std: float

    def to_dict(self):
        return {
            "load_mean": self.load_mean, "load_std": self.load_std,
            "temp_mean": self.temp_mean, "temp_std": self.temp_std,
        }


def _cyc(values, period):
    ang = 2 * np.pi * values / period
    return np.sin(ang), np.cos(ang)


def build_timestep_features(df: pd.DataFrame, scaler: Scaler | None = None):
    """Return (feature_matrix [T, N_FEATURES], scaler) for one region, time-sorted."""
    df = df.sort_values("datetime").reset_index(drop=True)
    if scaler is None:
        scaler = Scaler(
            load_mean=float(df.load_mw.mean()), load_std=float(df.load_mw.std()),
            temp_mean=float(df.temperature.mean()), temp_std=float(df.temperature.std()),
        )
    dt = df.datetime.dt
    hour_sin, hour_cos = _cyc(dt.hour.to_numpy(), 24)
    dow_sin, dow_cos = _cyc(dt.dayofweek.to_numpy(), 7)
    cols = {
        "load": (df.load_mw.to_numpy() - scaler.load_mean) / scaler.load_std,
        "temp": (df.temperature.to_numpy() - scaler.temp_mean) / scaler.temp_std,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "is_weekend": (dt.dayofweek >= 5).astype(float).to_numpy(),
        "is_holiday": df.is_holiday.astype(float).to_numpy(),
    }
    mat = np.column_stack([cols[f] for f in FEATURES]).astype(np.float32)
    return mat, scaler


def make_sequences(feat: np.ndarray, window=WINDOW, horizon=HORIZON):
    """Sliding windows. X:[N,window,F]  y:[N,horizon] (normalised load = channel 0)."""
    load = feat[:, 0]
    n = len(feat) - window - horizon + 1
    X = np.empty((n, window, feat.shape[1]), dtype=np.float32)
    y = np.empty((n, horizon), dtype=np.float32)
    for i in range(n):
        X[i] = feat[i:i + window]
        y[i] = load[i + window:i + window + horizon]
    return X, y


def tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flat features for the XGBoost baseline (predicts t+HORIZON load)."""
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    dt = df.datetime.dt
    df["hour"] = dt.hour
    df["dow"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    for lag in (1, 2, 3, 24, 25, 48, 168):
        df[f"lag_{lag}"] = df.load_mw.shift(lag)
    df["roll_mean_24"] = df.load_mw.shift(1).rolling(24).mean()
    df["roll_std_24"] = df.load_mw.shift(1).rolling(24).std()
    df["roll_mean_168"] = df.load_mw.shift(1).rolling(168).mean()
    df["target"] = df.load_mw.shift(-HORIZON)  # predict HORIZON hours ahead
    return df.dropna().reset_index(drop=True)


TABULAR_COLS = [
    "hour", "dow", "month", "is_weekend", "temperature", "is_holiday",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_25", "lag_48", "lag_168",
    "roll_mean_24", "roll_std_24", "roll_mean_168",
]
