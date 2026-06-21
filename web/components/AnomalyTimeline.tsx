"use client";
import type { Anomaly } from "@/lib/types";
import { fmtMW } from "@/lib/data";

const when = (t: string) =>
  new Date(t).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
  });

export function AnomalyTimeline({ anomalies }: { anomalies: Anomaly[] }) {
  if (anomalies.length === 0) {
    return (
      <p className="text-sm text-ink-faint">
        No residual anomalies above threshold in this window.
      </p>
    );
  }
  const sorted = [...anomalies].sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
  return (
    <div className="space-y-2">
      <div className="mb-3 flex items-center gap-4 text-xs text-ink-faint">
        <span>
          <span className="font-mono text-bad">{anomalies.length}</span> flagged
        </span>
        <span>method: forecast-residual z-score</span>
      </div>
      <ul className="max-h-[300px] space-y-2 overflow-y-auto pr-1">
        {sorted.map((a, i) => {
          const dev = a.actual - a.predicted;
          return (
            <li
              key={i}
              className="flex items-center justify-between rounded-lg border border-white/5 bg-base-700/40 px-3 py-2.5"
            >
              <div className="flex items-center gap-3">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    a.severity === "high" ? "bg-bad" : "bg-warn"
                  }`}
                />
                <div>
                  <div className="font-mono text-sm text-ink">{when(a.t)}</div>
                  <div className="text-xs text-ink-faint">
                    actual {fmtMW(a.actual)} · forecast {fmtMW(a.predicted)}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div
                  className={`font-mono text-sm font-semibold ${
                    dev > 0 ? "text-bad" : "text-electric"
                  }`}
                >
                  {dev > 0 ? "+" : ""}
                  {fmtMW(dev)}
                </div>
                <div className="stat-label">z {a.z.toFixed(1)}</div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
