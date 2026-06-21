"""Data loading + realistic synthetic generation for VOLTA.

`load_energy()` prefers a real PJM-style CSV at ARTIFACTS/energy.csv with columns
[datetime, region, load_mw, temperature]. If absent, it synthesises a
physically-plausible multi-region hourly dataset (daily/weekly/seasonal cycles,
temperature-driven demand, holidays, noise, and injected anomalies) so the whole
pipeline and dashboard work end-to-end with zero external downloads.

To use real data later: drop a CSV at ml/artifacts/energy.csv with those columns
(or adapt this loader) and re-run train.py — nothing else changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays as pyholidays

from config import RAW_CSV, REGIONS, SEED


def _us_holidays(years) -> set:
    return set(pyholidays.UnitedStates(years=list(years)).keys())


def _synth_region(idx: pd.DatetimeIndex, base: float, amp: float,
                  rng: np.random.Generator, holset: set):
    """Synthesise one region's hourly load + temperature.

    Deliberately *not* perfectly periodic: weather moves as autocorrelated fronts,
    buildings respond to a smoothed (thermal-inertia) temperature, the weekly shape
    drifts week-to-week, and noise is autocorrelated. This makes naive lag features
    imperfect and rewards a model that reads the whole recent trajectory.
    """
    n = len(idx)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    def ar1(rho, sigma):
        """Autocorrelated (AR(1)) series — persistent, not white noise."""
        e = rng.normal(0, sigma, n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = rho * x[t - 1] + e[t]
        return x

    # Temperature: seasonal + diurnal + multi-day weather fronts (AR(1)).
    seasonal_t = 12.0 - 13.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    diurnal_t = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    fronts = ar1(0.985, 0.7)  # slow-moving warm/cold spells lasting days
    temp = seasonal_t + diurnal_t + fronts + rng.normal(0, 0.6, n)

    # Demand: daily double-peak + a weekly shape that slowly drifts over time.
    daily = (
        0.55 * np.sin(2 * np.pi * (hour - 7) / 24)
        + 0.35 * np.sin(2 * np.pi * (hour - 19) / 12)
    )
    is_weekend = (dow >= 5).astype(float)
    week_idx = (np.arange(n) // 168)
    drift = 0.06 * np.sin(2 * np.pi * week_idx / 11 + rng.uniform(0, 6.28))
    weekly = 1.0 - 0.12 * is_weekend + drift

    # Thermal inertia: load responds to a smoothed temperature (building mass),
    # so the recent temperature *trajectory* — not just the instant — matters.
    comfort = 18.0
    temp_smooth = pd.Series(temp).rolling(8, min_periods=1).mean().to_numpy()
    temp_load = 0.013 * (temp_smooth - comfort) ** 2

    annual = 1.0 + 0.10 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    growth = 1.0 + 0.04 * (np.arange(n) / n)  # ~4% demand growth across the span

    holiday_mask = np.array([d.date() in holset for d in idx], dtype=float)
    holiday_factor = 1.0 - 0.08 * holiday_mask

    load = (
        base
        + amp * daily
        + amp * temp_load
    ) * weekly * annual * growth * holiday_factor
    load *= 1.0 + 0.012 * ar1(0.7, 1.0)  # autocorrelated multiplicative noise

    return load, temp, holiday_mask, is_weekend


def _inject_anomalies(load: np.ndarray, rng: np.random.Generator):
    """Inject a handful of realistic spikes/dips/flatlines for the anomaly view."""
    out = load.copy()
    labels = np.zeros(len(load), dtype=int)
    n_events = max(6, len(load) // 4000)
    for _ in range(n_events):
        i = rng.integers(200, len(load) - 50)
        kind = rng.choice(["spike", "dip", "flatline"])
        dur = int(rng.integers(2, 7))
        if kind == "spike":
            out[i:i + dur] *= rng.uniform(1.25, 1.6)
        elif kind == "dip":
            out[i:i + dur] *= rng.uniform(0.45, 0.7)
        else:  # sensor flatline
            out[i:i + dur] = out[i - 1]
        labels[i:i + dur] = 1
    return out, labels


def make_synthetic(start="2021-01-01", end="2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    holset = _us_holidays(range(idx[0].year, idx[-1].year + 1))

    frames = []
    for region, cfg in REGIONS.items():
        load, temp, hol, wknd = _synth_region(idx, cfg["base"], cfg["amp"], rng, holset)
        load, anom = _inject_anomalies(load, rng)
        frames.append(pd.DataFrame({
            "datetime": idx,
            "region": region,
            "load_mw": np.round(load, 1),
            "temperature": np.round(temp, 2),
            "is_holiday": hol.astype(int),
            "is_anomaly": anom,
        }))
    return pd.concat(frames, ignore_index=True)


def load_energy() -> pd.DataFrame:
    """Return the energy dataframe, generating + caching synthetic data if needed."""
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV, parse_dates=["datetime"])
        if "is_holiday" not in df:
            holset = _us_holidays(range(df.datetime.dt.year.min(),
                                        df.datetime.dt.year.max() + 1))
            df["is_holiday"] = df.datetime.dt.date.isin(holset).astype(int)
        if "is_anomaly" not in df:
            df["is_anomaly"] = 0
        return df
    df = make_synthetic()
    df.to_csv(RAW_CSV, index=False)
    return df


if __name__ == "__main__":
    df = load_energy()
    print(df.groupby("region").agg(
        rows=("load_mw", "size"),
        mean_mw=("load_mw", "mean"),
        anomalies=("is_anomaly", "sum"),
    ))
    print(f"\nRange: {df.datetime.min()} -> {df.datetime.max()}")
    print(f"Cached at: {RAW_CSV}")
