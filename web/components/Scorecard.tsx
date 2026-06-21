"use client";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelName, RegionScore } from "@/lib/types";

const ORDER: ModelName[] = ["CNN-BiLSTM", "XGBoost", "Seasonal-Naive"];
const COLORS: Record<ModelName, string> = {
  "CNN-BiLSTM": "#ffb020",
  XGBoost: "#2dd4ff",
  "Seasonal-Naive": "#9aa7b8",
};

export function Scorecard({ score }: { score: RegionScore }) {
  const best = (key: "mae" | "rmse" | "mape" | "r2") => {
    const vals = ORDER.map((m) => score.models[m][key]);
    return key === "r2" ? Math.max(...vals) : Math.min(...vals);
  };

  const phRows = (() => {
    const len = score.per_horizon_mae["CNN-BiLSTM"].length;
    return Array.from({ length: len }, (_, i) => ({
      h: i + 1,
      "CNN-BiLSTM": score.per_horizon_mae["CNN-BiLSTM"][i],
      XGBoost: score.per_horizon_mae["XGBoost"][i],
      "Seasonal-Naive": score.per_horizon_mae["Seasonal-Naive"][i],
    }));
  })();

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="text-left">
              <th className="stat-label pb-2">Model</th>
              <th className="stat-label pb-2 text-right">MAE</th>
              <th className="stat-label pb-2 text-right">RMSE</th>
              <th className="stat-label pb-2 text-right">MAPE</th>
              <th className="stat-label pb-2 text-right">R²</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {ORDER.map((m) => {
              const r = score.models[m];
              const cell = (v: number, key: "mae" | "rmse" | "mape" | "r2", txt: string) => (
                <td
                  className={`py-2.5 text-right ${
                    v === best(key) ? "font-bold text-amber" : "text-ink-dim"
                  }`}
                >
                  {txt}
                </td>
              );
              return (
                <tr key={m} className="border-t border-white/5">
                  <td className="py-2.5 font-sans">
                    <span
                      className="mr-2 inline-block h-2.5 w-2.5 rounded-full align-middle"
                      style={{ background: COLORS[m] }}
                    />
                    {m}
                  </td>
                  {cell(r.mae, "mae", r.mae.toFixed(0))}
                  {cell(r.rmse, "rmse", r.rmse.toFixed(0))}
                  {cell(r.mape, "mape", `${r.mape.toFixed(2)}%`)}
                  {cell(r.r2, "r2", r.r2.toFixed(3))}
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-ink-faint">
          Held-out test · 24h-averaged. XGBoost (joint multi-output) edges the
          full-horizon average; the CNN-BiLSTM wins the short-horizon regime →
        </p>
      </div>

      <div>
        <div className="stat-label mb-2">MAE by forecast horizon (hours ahead)</div>
        <p className="mb-2 text-xs text-ink-faint">
          CNN-BiLSTM (amber) leads early; lines cross where XGBoost overtakes.
        </p>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={phRows} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="h" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} width={40} />
              <Tooltip
                formatter={(v: number, n) => [`${v.toFixed(0)} MW`, n as string]}
                labelFormatter={(h) => `+${h}h`}
              />
              {ORDER.map((m) => (
                <Line
                  key={m}
                  dataKey={m}
                  stroke={COLORS[m]}
                  strokeWidth={m === "CNN-BiLSTM" ? 2.2 : 1.4}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
