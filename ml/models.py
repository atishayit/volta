"""Models for VOLTA: the CNN-BiLSTM hero plus baselines + metrics.

The hero (`MBDLSTM`) is a 1-D CNN feature extractor feeding a bidirectional LSTM
and a dense head that emits the full HORIZON-step forecast at once (direct multi-
step). Kept deliberately small so it trains on CPU in minutes and exports to a
compact ONNX graph that runs in-browser via onnxruntime-web.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import HORIZON, N_FEATURES


class MBDLSTM(nn.Module):
    """CNN -> Bi-LSTM -> dense multi-step forecaster."""

    def __init__(self, n_features=N_FEATURES, horizon=HORIZON,
                 conv_ch=48, lstm_hidden=128, dropout=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_ch, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_ch, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        # Head sees the LSTM read-out (last ⊕ mean ⊕ max) AND the raw recent loads
        # (direct persistence access) so it can sharpen short horizons while the
        # daily-seasonal skip handles long horizons.
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2 * 3 + horizon, 224),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(224, horizon),
        )

    def forward(self, x):                 # x: [B, WINDOW, F]
        z = self.conv(x.transpose(1, 2))  # -> [B, C, WINDOW]
        z = z.transpose(1, 2)             # -> [B, WINDOW, C]
        out, _ = self.lstm(z)             # -> [B, WINDOW, 2H]
        last = out[:, -1, :]              # final timestep state
        mean = out.mean(dim=1)            # average context
        mx = out.max(dim=1).values        # salient peaks (lag spikes)
        # Daily-seasonal skip: anchor each of the HORIZON steps to the load at the
        # SAME HOUR ONE DAY EARLIER (channel 0 of the window's last 24 steps).
        anchor = x[:, -HORIZON:, 0]       # [B, HORIZON] same-hour-yesterday load
        correction = self.head(torch.cat([last, mean, mx, anchor], dim=1))
        return anchor + correction
