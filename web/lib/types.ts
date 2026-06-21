export interface ModelMetrics {
  mae: number;
  rmse: number;
  mape: number;
  r2: number;
}

export type ModelName = "CNN-BiLSTM" | "XGBoost" | "Seasonal-Naive";

export interface RegionScore {
  label: string;
  models: Record<ModelName, ModelMetrics>;
  per_horizon_mae: Record<ModelName, number[]>;
}

export type Scorecard = Record<string, RegionScore>;

export interface ForecastPoint {
  t: string;
  actual: number;
  predicted: number;
  lower: number;
  upper: number;
}
export type Forecasts = Record<string, ForecastPoint[]>;

export interface Anomaly {
  t: string;
  actual: number;
  predicted: number;
  z: number;
  severity: "medium" | "high";
}
export type Anomalies = Record<string, Anomaly[]>;

export interface Scaler {
  load_mean: number;
  load_std: number;
  temp_mean: number;
  temp_std: number;
}

export interface Meta {
  window: number;
  horizon: number;
  features: string[];
  default_region: string;
  regions: Record<string, string>;
  scalers: Record<string, Scaler>;
  z_thresh: number;
}

export interface Scenario {
  region: string;
  window: number[][]; // [WINDOW, F]
  features: string[];
  horizon: number;
  start: string;
}
