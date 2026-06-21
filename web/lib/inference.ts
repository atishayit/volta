// In-browser CNN-BiLSTM inference via onnxruntime-web. Runs the same model.onnx
// exported from PyTorch — no server, no API. Powers the what-if simulator.
import type { Meta, Scaler } from "./types";

// Cache BOTH the ort module and the session together: the Tensor and the session
// must come from the same ORT instance or run() throws "Session mismatch".
let runtimePromise: Promise<{ ort: any; session: any }> | null = null;

// Must match the pinned onnxruntime-web version in package.json so the JS glue
// and the CDN-hosted WASM binary agree (mismatch => "e.getValue is not a function").
const ORT_VERSION = "1.27.0";

const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      // Load ORT from the CDN at runtime rather than bundling it: the bundled
      // build mismatches the CDN WASM and throws "Session mismatch". webpackIgnore
      // keeps Next from trying to resolve the URL at build time.
      const ort: any = await import(
        /* webpackIgnore: true */ `${ORT_CDN}/ort.wasm.min.mjs`
      );
      ort.env.wasm.wasmPaths = `${ORT_CDN}/`;
      // Single-threaded: avoids needing COOP/COEP cross-origin isolation on Vercel.
      ort.env.wasm.numThreads = 1;
      const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
      const session = await ort.InferenceSession.create(
        `${base}/models/model.onnx`,
        { executionProviders: ["wasm"], graphOptimizationLevel: "basic" },
      );
      return { ort, session };
    })();
  }
  return runtimePromise;
}

export interface Scenario3 {
  tempOffsetC: number; // shift the whole context temperature (heatwave / cold snap)
  weekend: boolean; // weekday vs weekend demand pattern
  hourShift: number; // rotate forecast start hour (0..23)
}

const featureIdx = (meta: Meta, name: string) => meta.features.indexOf(name);

/** Apply scenario knobs to the base context window, returning a fresh [W*F] tensor data. */
export function applyScenario(
  baseWindow: number[][],
  s: Scenario3,
  meta: Meta,
  scaler: Scaler,
): Float32Array {
  const W = baseWindow.length;
  const F = meta.features.length;
  const iTemp = featureIdx(meta, "temp");
  const iWknd = featureIdx(meta, "is_weekend");
  const iHs = featureIdx(meta, "hour_sin");
  const iHc = featureIdx(meta, "hour_cos");

  // Rotate the window so its phase reflects the chosen forecast-start hour.
  const shift = ((s.hourShift % W) + W) % W;
  const tempDeltaNorm = s.tempOffsetC / scaler.temp_std;

  const out = new Float32Array(W * F);
  for (let t = 0; t < W; t++) {
    const src = baseWindow[(t + shift) % W];
    for (let f = 0; f < F; f++) out[t * F + f] = src[f];
    out[t * F + iTemp] += tempDeltaNorm;
    out[t * F + iWknd] = s.weekend ? 1 : 0;
    // Nudge cyclical hour features toward a coherent weekend/weekday rhythm.
    if (s.weekend) {
      out[t * F + iHs] *= 0.96;
      out[t * F + iHc] *= 0.96;
    }
  }
  return out;
}

/** Run the model and return the 24-step forecast in real MW. */
export async function forecast(
  windowData: Float32Array,
  meta: Meta,
  scaler: Scaler,
): Promise<number[]> {
  const { ort, session } = await getRuntime();
  const F = meta.features.length;
  const tensor = new ort.Tensor("float32", windowData, [1, meta.window, F]);
  const result = await session.run({ window: tensor });
  const out = result.forecast.data as Float32Array;
  return Array.from(out, (v) => v * scaler.load_std + scaler.load_mean);
}

export async function warmup() {
  try {
    await getRuntime();
  } catch {
    /* warmup best-effort */
  }
}
