"""Build the REAL dataset: PJM hourly load + Open-Meteo hourly temperature.

Replaces the synthetic generator with genuine data:
  * Load  — PJM Interconnection hourly MW (Kaggle `robikscube/hourly-energy-
            consumption`, mirrored on GitHub), regions PJME / COMED / DAYTON.
  * Weather — real hourly 2 m temperature from the free, key-less Open-Meteo
            archive API for each region's representative city, aligned to the
            load timestamps (US Eastern, PJM's clock).

Writes ml/artifacts/energy.csv (datetime, region, load_mw, temperature,
is_holiday, is_anomaly) — the exact schema load_energy() consumes, so the rest of
the pipeline (train -> baselines -> export) is unchanged.

Run:  python fetch_real_data.py   then   python train.py / baselines.py / export.py
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import holidays as pyholidays

from config import ARTIFACTS, RAW_CSV

WINDOW_START, WINDOW_END = "2015-01-01", "2017-12-31"  # 3 full years, all regions cover it
RAW_DIR = ARTIFACTS / "pjm_raw"
PJM_BASE = "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data"

# region -> (PJM csv name, representative city lat, lon)
REGION_SRC = {
    "PJME": ("PJME_hourly.csv", 39.9526, -75.1652),   # Philadelphia
    "COMED": ("COMED_hourly.csv", 41.8781, -87.6298),  # Chicago
    "DAYTON": ("DAYTON_hourly.csv", 39.7589, -84.1916),  # Dayton, OH
}


def _download_pjm():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for csv, *_ in REGION_SRC.values():
        dest = RAW_DIR / csv
        if dest.exists():
            continue
        url = f"{PJM_BASE}/{csv}"
        print(f"  downloading {csv} …")
        urllib.request.urlretrieve(url, dest)


def _fetch_temperature(lat, lon) -> pd.Series:
    """Hourly 2 m temperature (°C) over the window, indexed in US/Eastern."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": WINDOW_START, "end_date": WINDOW_END,
        "hourly": "temperature_2m",
        "timezone": "America/New_York",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = json.load(r)
    h = data["hourly"]
    return pd.Series(
        h["temperature_2m"],
        index=pd.to_datetime(h["time"]),
        name="temperature",
    )


def _load_region(csv) -> pd.Series:
    df = pd.read_csv(RAW_DIR / csv, parse_dates=["Datetime"]).dropna()
    mwcol = [c for c in df.columns if c != "Datetime"][0]
    df = df.drop_duplicates("Datetime").set_index("Datetime").sort_index()
    s = df[mwcol]
    return s[(s.index >= WINDOW_START) & (s.index <= WINDOW_END + " 23:00")]


def main():
    _download_pjm()
    full_idx = pd.date_range(WINDOW_START, WINDOW_END + " 23:00", freq="h")
    holset = set(pyholidays.UnitedStates(
        years=range(int(WINDOW_START[:4]), int(WINDOW_END[:4]) + 1)).keys())

    frames = []
    for region, (csv, lat, lon) in REGION_SRC.items():
        load = _load_region(csv).reindex(full_idx).interpolate(limit=6).ffill().bfill()
        print(f"  [{region}] fetching weather …")
        temp = _fetch_temperature(lat, lon).reindex(full_idx).interpolate().ffill().bfill()
        frames.append(pd.DataFrame({
            "datetime": full_idx,
            "region": region,
            "load_mw": load.to_numpy().round(1),
            "temperature": temp.to_numpy().round(2),
            "is_holiday": [d.date() in holset for d in full_idx],
            "is_anomaly": 0,
        }))
        print(f"  [{region}] rows {len(full_idx)}  "
              f"MW[{load.min():.0f}-{load.max():.0f}]  "
              f"°C[{temp.min():.0f}-{temp.max():.0f}]")

    out = pd.concat(frames, ignore_index=True)
    out["is_holiday"] = out["is_holiday"].astype(int)
    out.to_csv(RAW_CSV, index=False)
    print(f"\nWrote REAL dataset -> {RAW_CSV}  ({len(out)} rows)")
    print("Next: python train.py && python baselines.py && python export.py")


if __name__ == "__main__":
    main()
