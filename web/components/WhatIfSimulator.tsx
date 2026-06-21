"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Meta, Scenario } from "@/lib/types";
import { fmtMW } from "@/lib/data";
import { applyScenario, forecast, type Scenario3, warmup } from "@/lib/inference";

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="stat-label">{label}</span>
        <span className="font-mono text-sm font-semibold text-amber">
          {value > 0 && unit === "°C" ? "+" : ""}
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="volta-range w-full"
      />
    </label>
  );
}

export function WhatIfSimulator({
  scenario,
  meta,
}: {
  scenario: Scenario;
  meta: Meta;
}) {
  const scaler = meta.scalers[scenario.region];
  const [temp, setTemp] = useState(0);
  const [weekend, setWeekend] = useState(false);
  const [hour, setHour] = useState(0);
  const [baseline, setBaseline] = useState<number[] | null>(null);
  const [live, setLive] = useState<number[] | null>(null);
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const reqId = useRef(0);

  const run = useCallback(
    async (s: Scenario3, setter: (v: number[]) => void) => {
      const data = applyScenario(scenario.window, s, meta, scaler);
      return forecast(data, meta, scaler).then(setter);
    },
    [scenario.window, meta, scaler],
  );

  // Warm up + compute the unperturbed baseline once.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await warmup();
        await run({ tempOffsetC: 0, weekend: false, hourShift: 0 }, (v) => {
          if (alive) setBaseline(v);
        });
        if (alive) setReady(true);
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [run]);

  // Re-forecast live as sliders move (debounced via request id).
  useEffect(() => {
    if (!ready) return;
    const id = ++reqId.current;
    setBusy(true);
    const handle = setTimeout(() => {
      run({ tempOffsetC: temp, weekend, hourShift: hour }, (v) => {
        if (id === reqId.current) {
          setLive(v);
          setBusy(false);
        }
      }).catch(() => setBusy(false));
    }, 60);
    return () => clearTimeout(handle);
  }, [temp, weekend, hour, ready, run]);

  const rows = (live ?? baseline ?? []).map((v, i) => ({
    h: `+${i + 1}`,
    baseline: baseline ? baseline[i] : null,
    forecast: v,
  }));

  const peak = live ? Math.max(...live) : baseline ? Math.max(...baseline) : 0;
  const basePeak = baseline ? Math.max(...baseline) : 0;
  const deltaPct = basePeak ? ((peak - basePeak) / basePeak) * 100 : 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
      <div className="space-y-5">
        <Slider
          label="Temperature shift"
          value={temp}
          min={-15}
          max={15}
          step={1}
          unit="°C"
          onChange={setTemp}
        />
        <Slider
          label="Forecast start hour"
          value={hour}
          min={0}
          max={23}
          step={1}
          unit=":00"
          onChange={setHour}
        />
        <div>
          <div className="stat-label mb-1.5">Day type</div>
          <div className="flex gap-2">
            <button
              onClick={() => setWeekend(false)}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm transition ${
                !weekend ? "bg-amber text-base-900" : "bg-base-700/50 text-ink-dim"
              }`}
            >
              Weekday
            </button>
            <button
              onClick={() => setWeekend(true)}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm transition ${
                weekend ? "bg-amber text-base-900" : "bg-base-700/50 text-ink-dim"
              }`}
            >
              Weekend
            </button>
          </div>
        </div>

        <div className="rounded-lg border border-white/5 bg-base-700/40 p-3">
          <div className="stat-label">Predicted peak (24h)</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-amber">
            {ready ? fmtMW(peak) : "—"}
          </div>
          <div
            className={`text-xs ${deltaPct >= 0 ? "text-bad" : "text-good"}`}
          >
            {ready
              ? `${deltaPct >= 0 ? "+" : ""}${deltaPct.toFixed(1)}% vs baseline`
              : "loading model…"}
          </div>
        </div>
        <p className="text-xs text-ink-faint">
          {busy ? "↻ re-forecasting…" : "Model runs entirely in your browser via onnxruntime-web — no server."}
        </p>
      </div>

      <div className="h-[320px]">
        {err ? (
          <div className="grid h-full place-items-center px-6 text-center text-sm text-bad">
            model error: {err}
          </div>
        ) : !ready ? (
          <div className="grid h-full place-items-center text-sm text-ink-faint">
            initialising in-browser model…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="h" tickLine={false} axisLine={false} minTickGap={16} />
              <YAxis
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}`}
                width={36}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                formatter={(v: number, n) => [fmtMW(v), n as string]}
                labelFormatter={(h) => `${h}h ahead`}
              />
              <Line
                dataKey="baseline"
                stroke="#5b6675"
                strokeWidth={1.4}
                strokeDasharray="4 4"
                dot={false}
                isAnimationActive={false}
                name="Baseline"
              />
              <Line
                dataKey="forecast"
                stroke="#ffb020"
                strokeWidth={2.4}
                dot={false}
                isAnimationActive={false}
                name="What-if forecast"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
