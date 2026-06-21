import type {
  Anomalies,
  Forecasts,
  Meta,
  Scenario,
  Scorecard,
} from "./types";

// Static JSON lives in /public/data and is fetched at runtime (static export).
const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

async function getJSON<T>(name: string): Promise<T> {
  const res = await fetch(`${base}/data/${name}`, { cache: "force-cache" });
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return res.json() as Promise<T>;
}

export const loadMeta = () => getJSON<Meta>("meta.json");
export const loadScorecard = () => getJSON<Scorecard>("metrics.json");
export const loadForecasts = () => getJSON<Forecasts>("forecasts.json");
export const loadAnomalies = () => getJSON<Anomalies>("anomalies.json");
export const loadScenario = () => getJSON<Scenario>("scenario.json");

export async function loadAll() {
  const [meta, scorecard, forecasts, anomalies, scenario] = await Promise.all([
    loadMeta(),
    loadScorecard(),
    loadForecasts(),
    loadAnomalies(),
    loadScenario(),
  ]);
  return { meta, scorecard, forecasts, anomalies, scenario };
}

export const fmtMW = (v: number) =>
  v >= 1000 ? `${(v / 1000).toFixed(1)} GW` : `${v.toFixed(0)} MW`;

export const fmtPct = (v: number) => `${v.toFixed(2)}%`;
