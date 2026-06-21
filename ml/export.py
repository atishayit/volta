"""Export trained artifacts for the web app.

Produces, under web/public/:
  models/model.onnx          - the CNN-BiLSTM, runs in-browser via onnxruntime-web
  data/metrics.json          - scorecard (all models, all regions) + per-horizon
  data/forecasts.json        - actual vs predicted (+ band) on the test tail
  data/anomalies.json        - residual z-score anomaly flags
  data/scenario.json         - fixed input window + scalers for the what-if sim
  data/meta.json             - feature order, window/horizon, regions, scalers

Run after train.py:  python export.py
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from config import (DATA_OUT, DEFAULT_REGION, FEATURES, HORIZON, MODEL_OUT,
                    REGIONS, WINDOW, ARTIFACTS)
from models import MBDLSTM
from pipeline import build_samples

PLOT_TAIL = 21 * 24      # last 3 weeks of hourly points shown on the hero chart
Z_THRESH = 3.0           # anomaly z-score threshold on forecast residuals


def load_model(name: str = "model.pt") -> MBDLSTM:
    model = MBDLSTM()
    model.load_state_dict(torch.load(ARTIFACTS / name, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def predict_mw(model, samp, sl):
    out = model(torch.from_numpy(samp.X_seq[sl])).numpy()
    return out * samp.scaler.load_std + samp.scaler.load_mean


def export_onnx(model):
    dummy = torch.randn(1, WINDOW, len(FEATURES))
    path = MODEL_OUT / "model.onnx"
    # Remove any stale external-data sidecar from a previous dynamo export.
    sidecar = MODEL_OUT / "model.onnx.data"
    if sidecar.exists():
        sidecar.unlink()
    # Legacy TorchScript exporter (dynamo=False): embeds weights in a single
    # file and produces an ORT-friendly graph that loads in onnxruntime-web.
    torch.onnx.export(
        model, dummy, path.as_posix(),
        input_names=["window"], output_names=["forecast"],
        dynamic_axes={"window": {0: "batch"}, "forecast": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"  onnx -> {path}  ({path.stat().st_size / 1024:.0f} KB)")


def build_forecast_series(model, samp):
    """1-step-rolling actual-vs-predicted over the test tail for the hero chart."""
    test_idx = list(range(samp.cut, len(samp.X_seq)))[-PLOT_TAIL:]
    preds_h1, actual, times, resid = [], [], [], []
    out = predict_mw(model, samp, slice(test_idx[0], test_idx[-1] + 1))
    for k, i in enumerate(test_idx):
        p = float(out[k, 0])           # 1-hour-ahead prediction
        a = float(samp.y_mw[i, 0])
        preds_h1.append(p); actual.append(a)
        resid.append(a - p)
        times.append(np.datetime_as_string(samp.starts[i], unit="h") + ":00")
    resid = np.array(resid)
    sigma = float(resid.std())
    band = 1.96 * sigma
    series = [{
        "t": times[k], "actual": round(actual[k], 1),
        "predicted": round(preds_h1[k], 1),
        "lower": round(preds_h1[k] - band, 1),
        "upper": round(preds_h1[k] + band, 1),
    } for k in range(len(times))]
    return series, resid, sigma, times, actual, preds_h1


def build_anomalies(times, actual, preds, resid, sigma):
    z = resid / (sigma + 1e-9)
    out = []
    for k in range(len(times)):
        if abs(z[k]) >= Z_THRESH:
            out.append({
                "t": times[k], "actual": round(actual[k], 1),
                "predicted": round(preds[k], 1),
                "z": round(float(z[k]), 2),
                "severity": "high" if abs(z[k]) >= 4 else "medium",
            })
    return out


def main():
    model = load_model("model.pt")
    export_onnx(model)

    # Merge torch-trained (hero + naive) and XGBoost evals into one scorecard.
    with open(ARTIFACTS / "eval_hero.json") as f:
        eval_hero = json.load(f)
    with open(ARTIFACTS / "eval_xgb.json") as f:
        eval_xgb = json.load(f)
    scorecard = {}
    for region, h in eval_hero.items():
        scorecard[region] = {
            "label": h["label"],
            "models": {
                "CNN-BiLSTM": h["models"]["CNN-BiLSTM"],
                "XGBoost": eval_xgb[region]["metrics"],
                "Seasonal-Naive": h["models"]["Seasonal-Naive"],
            },
            "per_horizon_mae": {
                "CNN-BiLSTM": h["per_horizon_mae"]["CNN-BiLSTM"],
                "XGBoost": eval_xgb[region]["per_horizon_mae"],
                "Seasonal-Naive": h["per_horizon_mae"]["Seasonal-Naive"],
            },
        }

    samples = {r: build_samples(r) for r in REGIONS}
    scalers = {r: samples[r].scaler.to_dict() for r in REGIONS}

    forecasts, anomalies = {}, {}
    for r, samp in samples.items():
        series, resid, sigma, times, actual, preds = build_forecast_series(model, samp)
        forecasts[r] = series
        anomalies[r] = build_anomalies(times, actual, preds, resid, sigma)
        print(f"  [{r}] forecast points {len(series)}  anomalies {len(anomalies[r])}")

    # What-if scenario: a fixed recent window the browser perturbs with sliders.
    d = samples[DEFAULT_REGION]
    win = d.X_seq[d.cut + len(d.X_seq[d.cut:]) // 2]   # a representative window
    scenario = {
        "region": DEFAULT_REGION,
        "window": win.astype(float).round(5).tolist(),   # [WINDOW, F]
        "features": FEATURES,
        "horizon": HORIZON,
        "start": np.datetime_as_string(
            d.starts[d.cut + len(d.X_seq[d.cut:]) // 2], unit="h") + ":00",
    }

    meta = {
        "window": WINDOW, "horizon": HORIZON, "features": FEATURES,
        "default_region": DEFAULT_REGION,
        "regions": {r: REGIONS[r]["label"] for r in REGIONS},
        "scalers": scalers,
        "z_thresh": Z_THRESH,
    }

    _dump("metrics.json", scorecard)
    _dump("forecasts.json", forecasts)
    _dump("anomalies.json", anomalies)
    _dump("scenario.json", scenario)
    _dump("meta.json", meta)
    print("Export complete.")


def _dump(name, obj):
    path = DATA_OUT / name
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"  data -> {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
