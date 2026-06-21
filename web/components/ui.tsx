import { ReactNode } from "react";

export function Panel({
  title,
  kicker,
  right,
  className = "",
  children,
}: {
  title?: string;
  kicker?: string;
  right?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`panel p-5 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {kicker && <div className="stat-label mb-1 text-amber/80">{kicker}</div>}
            {title && (
              <h2 className="text-lg font-semibold tracking-tight text-ink">{title}</h2>
            )}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-base-700/40 px-4 py-3">
      <div className="stat-label">{label}</div>
      <div
        className={`mt-1 font-mono text-2xl font-semibold tabular-nums ${
          accent ? "text-amber" : "text-ink"
        }`}
      >
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-ink-faint">{sub}</div>}
    </div>
  );
}

export function Pill({
  active,
  onClick,
  children,
}: {
  active?: boolean;
  onClick?: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
        active
          ? "bg-amber text-base-900 shadow-glow"
          : "border border-white/10 bg-base-700/50 text-ink-dim hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}
