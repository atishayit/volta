"use client";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Anomaly, ForecastPoint } from "@/lib/types";
import { fmtMW } from "@/lib/data";

const shortTime = (t: string) => {
  const d = new Date(t);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
  });
};

export function ForecastChart({
  data,
  anomalies,
}: {
  data: ForecastPoint[];
  anomalies: Anomaly[];
}) {
  const anomSet = new Set(anomalies.map((a) => a.t));
  const rows = data.map((d) => ({
    ...d,
    bandLow: d.lower,
    bandRange: d.upper - d.lower,
    anomaly: anomSet.has(d.t) ? d.actual : null,
  }));

  return (
    <div className="h-[360px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
          <defs>
            <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2dd4ff" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#2dd4ff" stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="t"
            tickFormatter={shortTime}
            minTickGap={48}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
          />
          <YAxis
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}`}
            width={40}
            tickLine={false}
            axisLine={false}
            label={{
              value: "GW",
              angle: -90,
              position: "insideLeft",
              fill: "#5b6675",
              fontSize: 11,
              dy: 12,
            }}
          />
          <Tooltip
            labelFormatter={shortTime}
            formatter={(value: number, name) => [fmtMW(value), name as string]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            iconType="plainline"
          />
          {/* 95% confidence band rendered as a stacked transparent base + range */}
          <Area
            dataKey="bandLow"
            stackId="band"
            stroke="none"
            fill="transparent"
            isAnimationActive={false}
            legendType="none"
            name="band-base"
          />
          <Area
            dataKey="bandRange"
            stackId="band"
            stroke="none"
            fill="url(#bandFill)"
            isAnimationActive={false}
            name="95% band"
          />
          <Line
            dataKey="actual"
            stroke="#e7ecf3"
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
            name="Actual"
          />
          <Line
            dataKey="predicted"
            stroke="#ffb020"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="CNN-BiLSTM"
          />
          <Line
            dataKey="anomaly"
            stroke="none"
            dot={{ r: 4, fill: "#ff5470", stroke: "#0b0f16", strokeWidth: 1.5 }}
            isAnimationActive={false}
            name="Anomaly"
            legendType="circle"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
