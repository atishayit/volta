import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // "control-room" palette: deep instrument-panel dark + electric amber
        base: {
          900: "#070a0f",
          800: "#0b0f16",
          700: "#11161f",
          600: "#1a212d",
          500: "#28313f",
        },
        amber: {
          DEFAULT: "#ffb020",
          bright: "#ffc24d",
          deep: "#e08a00",
        },
        electric: "#2dd4ff",
        good: "#46d39a",
        warn: "#ffb020",
        bad: "#ff5470",
        ink: {
          DEFAULT: "#e7ecf3",
          dim: "#9aa7b8",
          faint: "#5b6675",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,176,32,0.15), 0 0 24px -4px rgba(255,176,32,0.25)",
        panel: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 12px 40px -16px rgba(0,0,0,0.8)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(circle at 50% 0%, rgba(45,212,255,0.08), transparent 60%)",
      },
    },
  },
  plugins: [],
};

export default config;
