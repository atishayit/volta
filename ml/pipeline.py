"""Unified sample construction shared by train.py and export.py.

Every model is evaluated on the *same* (history window -> next HORIZON hours)
samples so the scorecard comparison is apples-to-apples. Provides:
  * sequence tensors for the CNN-BiLSTM
  * flat summary features for XGBoost
  * seasonal-naive predictions
  * real-MW ground truth + forecast-start timestamps for plotting
all split into train/test by a time-based cut.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import HORIZON, WINDOW
from data import load_energy
from features import Scaler, build_timestep_features

TEST_FRAC = 0.2
SEASONAL_LAG = 168  # same hour one week earlier


@dataclass
class RegionSamples:
    region: str
    scaler: Scaler
    X_seq: np.ndarray        # [N, WINDOW, F]   model input (normalised)
    y_norm: np.ndarray       # [N, HORIZON]     normalised target
    y_mw: np.ndarray         # [N, HORIZON]     real-MW target
    naive_mw: np.ndarray     # [N, HORIZON]     seasonal-naive forecast (MW)
    tab_X: np.ndarray        # [N, K]           flat features for XGBoost
    starts: np.ndarray       # [N]              forecast-start timestamps (ns)
    cut: int                 # index splitting train/test samples

    @property
    def train(self):
        return slice(0, self.cut)

    @property
    def test(self):
        return slice(self.cut, None)


TAB_COLS = [
    "last", "lag_24", "lag_48", "lag_168",
    "roll_mean_24", "roll_std_24", "roll_mean_168",
    "temp", "hour", "dow", "month", "is_weekend", "is_holiday",
]


def build_samples(region: str, scaler: Scaler | None = None) -> RegionSamples:
    df = load_energy()
    df = df[df.region == region].sort_values("datetime").reset_index(drop=True)
    feat, scaler = build_timestep_features(df, scaler)

    load_mw = df.load_mw.to_numpy(np.float32)
    temp = df.temperature.to_numpy(np.float32)
    dt = df.datetime.dt
    hour = dt.hour.to_numpy(); dow = dt.dayofweek.to_numpy()
    month = dt.month.to_numpy(); hol = df.is_holiday.to_numpy()
    starts_all = df.datetime.to_numpy()

    T = len(feat)
    n = T - WINDOW - HORIZON + 1
    Xs, yn, ymw, naive, tab, starts = [], [], [], [], [], []
    for i in range(n):
        e = i + WINDOW            # forecast start index
        fut = slice(e, e + HORIZON)
        Xs.append(feat[i:e])
        yn.append(feat[fut, 0])           # normalised load channel
        ymw.append(load_mw[fut])
        # seasonal naive: value one week before each forecast hour
        if e - SEASONAL_LAG >= 0:
            naive.append(load_mw[e - SEASONAL_LAG:e - SEASONAL_LAG + HORIZON])
        else:
            naive.append(np.full(HORIZON, load_mw[i], np.float32))
        win_load = load_mw[i:e]
        tab.append([
            load_mw[e - 1], load_mw[e - 24], load_mw[e - 48], load_mw[e - 168],
            win_load[-24:].mean(), win_load[-24:].std(), win_load[-168:].mean(),
            temp[e], hour[e], dow[e], month[e],
            1.0 if dow[e] >= 5 else 0.0, hol[e],
        ])
        starts.append(starts_all[e])

    X_seq = np.asarray(Xs, np.float32)
    y_norm = np.asarray(yn, np.float32)
    y_mw = np.asarray(ymw, np.float32)
    naive_mw = np.asarray(naive, np.float32)
    tab_X = np.asarray(tab, np.float32)
    starts_arr = np.asarray(starts)
    cut = int(len(X_seq) * (1 - TEST_FRAC))
    return RegionSamples(region, scaler, X_seq, y_norm, y_mw, naive_mw,
                         tab_X, starts_arr, cut)
