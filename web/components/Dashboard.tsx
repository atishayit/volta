"use client";
import { useEffect, useMemo, useState } from "react";
import { loadAll, fmtMW, fmtPct } from "@/lib/data";
import type {
  Anomalies,
  Forecasts,
  Meta,
  Scenario,
  Scorecard as ScorecardT,
} from "@/lib/types";
import { Header } from "./Header";
import { Panel, Stat } from "./ui";
import { ForecastChart } from "./ForecastChart";
import { Scorecard } from "./Scorecard";
import { AnomalyTimeline } from "./AnomalyTimeline";
import { WhatIfSimulator } from "./WhatIfSimulator";

interface Bundle {
  meta: Meta;
  scorecard: ScorecardT;
  forecasts: Forecasts;
  anomalies: Anomalies;
  scenario: Scenario;
}

export function Dashboard() {
  const [data, setData] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [region, setRegion] = useState<string>("");

  useEffect(() => {
    loadAll()
      .then((d) => {
        setData(d);
        setRegion(d.meta.default_region);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const hero = useMemo(() => {
    if (!data || !region) return null;
    const score = data.scorecard[region];
    const cnn = score.models["CNN-BiLSTM"];
    const xgb = score.models["XGBoost"];
    const ph = score.per_horizon_mae;
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    // Short-horizon (1-6h) edge — where the CNN-BiLSTM beats joint XGBoost.
    const hShort = mean(ph["CNN-BiLSTM"].slice(0, 6));
    const xShort = mean(ph["XGBoost"].slice(0, 6));
    const shortEdge = ((xShort - hShort) / xShort) * 100;
    // Horizon up to which the CNN-BiLSTM leads.
    let leadsThrough = 0;
    for (let i = 0; i < ph["CNN-BiLSTM"].length; i++) {
      if (ph["CNN-BiLSTM"][i] <= ph["XGBoost"][i]) leadsThrough = i + 1;
      else break;
    }
    return { score, cnn, xgb, shortEdge, leadsThrough };
  }, [data, region]);

  if (error)
    return (
      <div className="grid min-h-screen place-items-center p-8 text-center text-sm text-ink-dim">
        Failed to load dashboard data.
        <br />
        <span className="text-ink-faint">{error}</span>
      </div>
    );

  if (!data || !region || !hero)
    return (
      <div className="grid min-h-screen place-items-center text-sm text-ink-faint">
        <span className="animate-pulse">initialising control room…</span>
      </div>
    );

  const forecast = data.forecasts[region] ?? [];
  const anomalies = data.anomalies[region] ?? [];

  return (
    <div className="min-h-screen">
      <Header regions={data.meta.regions} region={region} onRegion={setRegion} />

      <main className="mx-auto max-w-7xl space-y-6 px-5 py-8">
        {/* Headline ribbon */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="1–6h MAE vs XGBoost"
            value={`−${hero.shortEdge.toFixed(0)}%`}
            sub={`CNN-BiLSTM short-horizon edge · ${data.meta.regions[region]}`}
            accent
          />
          <Stat
            label="CNN-BiLSTM leads"
            value={`through h+${hero.leadsThrough}`}
            sub="before XGBoost overtakes long-horizon"
          />
          <Stat
            label="24h MAPE"
            value={fmtPct(hero.cnn.mape)}
            sub={`XGBoost ${fmtPct(hero.xgb.mape)} · competitive overall`}
          />
          <Stat
            label="Anomalies flagged"
            value={String(anomalies.length)}
            sub={`z-score ≥ ${data.meta.z_thresh}`}
          />
        </div>

        {/* Hero forecast */}
        <Panel
          kicker="forecast"
          title="Actual vs Predicted — held-out test window"
          className="panel-grid"
          right={
            <div className="hidden gap-2 sm:flex">
              <span className="chip">
                <span className="h-2 w-2 rounded-full bg-ink" /> actual
              </span>
              <span className="chip">
                <span className="h-2 w-2 rounded-full bg-amber" /> CNN-BiLSTM
              </span>
              <span className="chip">
                <span className="h-2 w-2 rounded-full bg-bad" /> anomaly
              </span>
            </div>
          }
        >
          <ForecastChart data={forecast} anomalies={anomalies} />
        </Panel>

        {/* What-if simulator — the showcase */}
        <Panel
          kicker="what-if · live in-browser inference"
          title="Re-forecast the next 24 hours under new conditions"
        >
          <WhatIfSimulator scenario={data.scenario} meta={data.meta} />
        </Panel>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel kicker="benchmark" title="Model scorecard">
            <Scorecard score={hero.score} />
          </Panel>
          <Panel kicker="anomaly detection" title="Flagged deviations">
            <AnomalyTimeline anomalies={anomalies} />
          </Panel>
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-white/5 pt-6 text-xs text-ink-faint">
          <span>
            VOLTA · CNN-BiLSTM forecasting · trained offline, served as static JSON + ONNX
          </span>
          <span className="font-mono">Atishay Jain</span>
        </footer>
      </main>
    </div>
  );
}
