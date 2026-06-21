import { Pill } from "./ui";

export function Header({
  regions,
  region,
  onRegion,
}: {
  regions: Record<string, string>;
  region: string;
  onRegion: (r: string) => void;
}) {
  return (
    <header className="sticky top-0 z-20 border-b border-white/5 bg-base-900/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-3">
        <div className="flex items-center gap-3">
          <div className="relative grid h-9 w-9 place-items-center rounded-lg bg-amber/10 ring-1 ring-amber/30">
            <span className="text-lg font-bold text-amber">⚡</span>
          </div>
          <div className="leading-tight">
            <div className="font-mono text-base font-bold tracking-[0.2em] text-ink">
              VOLTA
            </div>
            <div className="text-[11px] text-ink-faint">
              energy forecasting &amp; anomaly control room
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="stat-label mr-1 hidden sm:inline">region</span>
          {Object.entries(regions).map(([code, label]) => (
            <Pill key={code} active={code === region} onClick={() => onRegion(code)}>
              {code}
              <span className="ml-1 hidden text-xs opacity-70 md:inline">{label}</span>
            </Pill>
          ))}
        </div>
      </div>
    </header>
  );
}
