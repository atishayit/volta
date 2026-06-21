"""Shared configuration for the VOLTA ML pipeline.

A single source of truth for paths, the feature schema, and the model window/
horizon so that data generation, training, export and the web app all agree.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ML_DIR = Path(__file__).resolve().parent
ROOT = ML_DIR.parent
ARTIFACTS = ML_DIR / "artifacts"          # raw csv + scalers + checkpoints
WEB_PUBLIC = ROOT / "web" / "public"
DATA_OUT = WEB_PUBLIC / "data"            # forecasts.json, metrics.json, ...
MODEL_OUT = WEB_PUBLIC / "models"         # model.onnx

for _p in (ARTIFACTS, DATA_OUT, MODEL_OUT):
    _p.mkdir(parents=True, exist_ok=True)

RAW_CSV = ARTIFACTS / "energy.csv"        # synthetic or real PJM data lands here

# --- Regions ---------------------------------------------------------------
# Synthetic stand-ins shaped after PJM zones. Swap for real PJM region names
# when you drop a real CSV into ARTIFACTS/energy.csv.
REGIONS = {
    "PJME": {"base": 32000, "amp": 9000, "label": "PJM East"},
    "COMED": {"base": 11500, "amp": 4200, "label": "Commonwealth Edison"},
    "DAYTON": {"base": 2100, "amp": 700, "label": "Dayton"},
}
DEFAULT_REGION = "PJME"

# --- Model windowing -------------------------------------------------------
WINDOW = 168          # hours of history fed to the model (1 week)
HORIZON = 24          # hours forecast ahead (1 day)

# Per-timestep feature channels (order matters; mirrored in JS what-if).
FEATURES = [
    "load",        # normalised load (z-score)
    "temp",        # normalised temperature
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
]
N_FEATURES = len(FEATURES)

SEED = 42
