"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  Gauge,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { API_BASE_URL } from "@/lib/api";

// ─── Types ───

type ReliabilityData = {
  bin_centers: number[];
  bin_accuracies: number[];
  bin_confidences: number[];
  bin_counts: number[];
  n_bins: number;
  ece: number;
  provenance: string;
};

type RiskCoverageData = {
  points: Array<{ coverage: number; accuracy: number; threshold: number; n_retained: number }>;
  n_points: number;
  provenance: string;
};

type Fpr95Data = {
  fpr95: number;
  n_in_dist: number;
  n_out_dist: number;
  provenance: string;
};

type BrierData = {
  brier: number;
  reliability: number;
  resolution: number;
  uncertainty: number;
  provenance: string;
};

type MultiSeedData = {
  n_seeds: number;
  base_config: Record<string, number>;
  summaries: Array<{
    method: string;
    n_seeds: number;
    means: Record<string, number>;
    stds: Record<string, number>;
  }>;
};

const SHIP_CLASSES = ["cargo", "tanker", "passenger", "fishing", "naval", "small_craft"];
const CORRUPTION_COLORS: Record<string, string> = {
  fog: "#8FA3BF",
  rain: "#3A5680",
  low_light: "#1E4275",
  glare: "#C9963A",
  noise: "#B5443A",
};

export default function ExperimentsPage() {
  const [loading, setLoading] = useState<string | null>(null);
  const [reliability, setReliability] = useState<ReliabilityData | null>(null);
  const [riskCoverage, setRiskCoverage] = useState<RiskCoverageData | null>(null);
  const [fpr95Data, setFpr95Data] = useState<Fpr95Data | null>(null);
  const [brierData, setBrierData] = useState<BrierData | null>(null);
  const [multiSeed, setMultiSeed] = useState<MultiSeedData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = async () => {
    setLoading("all");
    setError(null);
    try {
      const [rel, rc, fpr, brier, ms] = await Promise.all([
        fetchJson<ReliabilityData>(`${API_BASE_URL}/experiments/reliability?n_samples=200&n_bins=15`),
        fetchJson<RiskCoverageData>(`${API_BASE_URL}/experiments/risk-coverage?n_samples=200&n_thresholds=15`),
        fetchJson<Fpr95Data>(`${API_BASE_URL}/experiments/fpr95?n_in_dist=100&n_out_dist=100`),
        fetchJson<BrierData>(`${API_BASE_URL}/experiments/brier-decomposition?n_samples=200`),
        fetchJson<MultiSeedData>(`${API_BASE_URL}/experiments/multi-seed?n_seeds=3&n_calibration=30&n_test=30`),
      ]);
      setReliability(rel);
      setRiskCoverage(rc);
      setFpr95Data(fpr);
      setBrierData(brier);
      setMultiSeed(ms);
    } catch (e) {
      setError("Failed to fetch experiment data. Is the API running?");
    }
    setLoading(null);
  };

  useEffect(() => {
    fetchAll();
  }, []);

  return (
    <div className="page-container">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="h-page">Experimental Evaluation</h1>
          <p className="body-secondary mt-1">
            Rigorous calibration, robustness, and deferral evaluation. All metrics
            computed via the evaluation harness with multi-seed mean ± std reporting.
          </p>
        </div>
        <button onClick={fetchAll} disabled={loading !== null} className="btn-secondary">
          {loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          {loading ? "Running..." : "Re-run experiments"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 flex items-center gap-2">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Metrics summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          icon={<Gauge size={16} />}
          label="ECE"
          value={reliability ? reliability.ece.toFixed(4) : "—"}
          sublabel="Expected Calibration Error"
          provenance={reliability?.provenance}
        />
        <MetricCard
          icon={<AlertTriangle size={16} />}
          label="FPR95"
          value={fpr95Data ? fpr95Data.fpr95.toFixed(4) : "—"}
          sublabel="OOD false positive rate at 95% TPR"
          provenance={fpr95Data?.provenance}
        />
        <MetricCard
          icon={<FlaskConical size={16} />}
          label="Brier Score"
          value={brierData ? brierData.brier.toFixed(4) : "—"}
          sublabel="Mean squared error of predictions"
          provenance={brierData?.provenance}
        />
        <MetricCard
          icon={<CheckCircle2 size={16} />}
          label="Seeds Run"
          value={multiSeed ? String(multiSeed.n_seeds) : "—"}
          sublabel="Multi-seed evaluation runs"
          provenance="measured"
        />
      </div>

      {/* Reliability diagram */}
      <div className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
          <h2 className="h-card">Reliability Diagram</h2>
          <ProvenanceChip provenance={reliability?.provenance || "not-evaluated"} />
        </div>
        <div className="p-5">
          {reliability ? (
            <ReliabilityDiagram data={reliability} />
          ) : (
            <div className="empty-state py-12">
              <Loader2 size={20} className="animate-spin text-navy-300" />
            </div>
          )}
        </div>
      </div>

      {/* Risk-coverage curve */}
      <div className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
          <h2 className="h-card">Risk-Coverage Curve</h2>
          <ProvenanceChip provenance={riskCoverage?.provenance || "not-evaluated"} />
        </div>
        <div className="p-5">
          {riskCoverage ? (
            <RiskCoverageChart data={riskCoverage} />
          ) : (
            <div className="empty-state py-12">
              <Loader2 size={20} className="animate-spin text-navy-300" />
            </div>
          )}
        </div>
      </div>

      {/* Brier decomposition */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="surface">
          <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
            <h2 className="h-card">Brier Score Decomposition</h2>
            <ProvenanceChip provenance={brierData?.provenance || "not-evaluated"} />
          </div>
          <div className="p-5">
            {brierData ? (
              <BrierDecompChart data={brierData} />
            ) : (
              <div className="empty-state py-12">
                <Loader2 size={20} className="animate-spin text-navy-300" />
              </div>
            )}
          </div>
        </div>

        {/* Corruption robustness (static data from the suite definition) */}
        <div className="surface">
          <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
            <h2 className="h-card">Corruption Robustness Suite</h2>
            <ProvenanceChip provenance="illustrative" label="ImageNet-C style" />
          </div>
          <div className="p-5">
            <CorruptionSuiteCard />
          </div>
        </div>
      </div>

      {/* Multi-seed results table */}
      <div className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
          <h2 className="h-card">Multi-Seed Evaluation (mean ± std)</h2>
          <ProvenanceChip provenance="measured" label={`${multiSeed?.n_seeds || 0} seeds`} />
        </div>
        <div className="overflow-x-auto">
          {multiSeed ? (
            <MultiSeedTable data={multiSeed} />
          ) : (
            <div className="empty-state py-12">
              <Loader2 size={20} className="animate-spin text-navy-300" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Chart components ───

function ReliabilityDiagram({ data }: { data: ReliabilityData }) {
  const chartData = data.bin_centers.map((center, i) => ({
    bin: center.toFixed(2),
    accuracy: data.bin_accuracies[i],
    confidence: data.bin_confidences[i],
    count: data.bin_counts[i],
  }));

  return (
    <div>
      <div className="flex items-center gap-4 mb-3 text-xs">
        <span className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-navy-700" />
          <span className="text-navy-600">Model accuracy</span>
        </span>
        <span className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-gold-400" />
          <span className="text-navy-600">Predicted confidence</span>
        </span>
        <span className="text-navy-400 ml-auto">
          ECE = {data.ece.toFixed(4)} — lower is better
        </span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E1E8F2" />
          <XAxis dataKey="bin" tick={{ fontSize: 10, fill: "#5C78A0" }} label={{ value: "Confidence bin", position: "insideBottom", offset: -5, style: { fontSize: 10, fill: "#5C78A0" } }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#5C78A0" }} />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 6, border: "1px solid #E1E8F2" }}
            formatter={(val: number) => val.toFixed(3)}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          <Bar dataKey="accuracy" fill="#0F2845" name="Accuracy" />
          <Bar dataKey="confidence" fill="#C9963A" name="Confidence" />
          <ReferenceLine stroke="#3F8A5E" strokeDasharray="3 3" segment={[{ x: 0, y: 0 }, { x: 14, y: 1 }]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="text-[10px] text-navy-400 mt-2">
        A perfectly calibrated model has accuracy = confidence in every bin (green diagonal).
        Gaps between bars represent miscalibration.
      </div>
    </div>
  );
}

function RiskCoverageChart({ data }: { data: RiskCoverageData }) {
  const chartData = [...data.points]
    .sort((a, b) => a.coverage - b.coverage)
    .map((p) => ({
      coverage: parseFloat(p.coverage.toFixed(3)),
      accuracy: parseFloat(p.accuracy.toFixed(3)),
      threshold: parseFloat(p.threshold.toFixed(3)),
    }));

  return (
    <div>
      <div className="text-xs text-navy-500 mb-3">
        Accuracy on retained cases vs. fraction of cases retained (coverage).
        A good deferral rule shows accuracy increasing as coverage decreases.
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E1E8F2" />
          <XAxis
            dataKey="coverage"
            type="number"
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: "#5C78A0" }}
            label={{ value: "Coverage (fraction retained)", position: "insideBottom", offset: -5, style: { fontSize: 10, fill: "#5C78A0" } }}
          />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#5C78A0" }} label={{ value: "Accuracy", angle: -90, position: "insideLeft", style: { fontSize: 10, fill: "#5C78A0" } }} />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 6, border: "1px solid #E1E8F2" }}
            formatter={(val: number) => val.toFixed(3)}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#0F2845"
            strokeWidth={2}
            dot={{ fill: "#C9963A", r: 3 }}
            name="Accuracy on retained cases"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function BrierDecompChart({ data }: { data: BrierData }) {
  const chartData = [
    { name: "Reliability", value: parseFloat(data.reliability.toFixed(4)), fill: "#B5443A" },
    { name: "Resolution", value: parseFloat(data.resolution.toFixed(4)), fill: "#3F8A5E" },
    { name: "Uncertainty", value: parseFloat(data.uncertainty.toFixed(4)), fill: "#8FA3BF" },
  ];

  return (
    <div>
      <div className="text-xs text-navy-500 mb-3">
        Murphy decomposition: Brier = Reliability − Resolution + Uncertainty.
        Lower reliability is better; higher resolution is better.
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#E1E8F2" />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#5C78A0" }} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: "#3A5680" }} width={90} />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 6, border: "1px solid #E1E8F2" }}
            formatter={(val: number) => val.toFixed(4)}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="text-[10px] text-navy-400 mt-2">
        Total Brier Score: <span className="font-mono font-bold text-navy-700">{data.brier.toFixed(4)}</span>
      </div>
    </div>
  );
}

function CorruptionSuiteCard() {
  const corruptions = [
    { id: "fog", name: "Fog", description: "Visibility reduction, white veil", icon: "🌫" },
    { id: "rain", name: "Rain", description: "Vertical streaks, brightness loss", icon: "🌧" },
    { id: "low_light", name: "Low Light", description: "Darkens, Poisson noise", icon: "🌙" },
    { id: "glare", name: "Glare", description: "Sun glint, bright halo", icon: "☀" },
    { id: "noise", name: "Noise", description: "Gaussian sensor noise", icon: "📡" },
  ];

  return (
    <div>
      <div className="text-xs text-navy-500 mb-3">
        5 corruption types × 5 severity levels = 25 test variants.
        Applied at test time to evaluate robustness under degraded maritime conditions.
      </div>
      <div className="space-y-2">
        {corruptions.map((c) => (
          <div key={c.id} className="flex items-center gap-3 p-2 rounded-md border border-navy-50 bg-navy-50/30">
            <div className="w-7 h-7 rounded-md flex items-center justify-center text-sm bg-white border border-navy-100">
              {c.icon}
            </div>
            <div className="flex-1">
              <div className="text-xs font-semibold text-navy-700">{c.name}</div>
              <div className="text-[10px] text-navy-400">{c.description}</div>
            </div>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((s) => (
                <div
                  key={s}
                  className="w-5 h-5 rounded-sm text-[9px] font-bold flex items-center justify-center text-white"
                  style={{ backgroundColor: CORRUPTION_COLORS[c.id], opacity: 0.3 + s * 0.14 }}
                >
                  {s}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MultiSeedTable({ data }: { data: MultiSeedData }) {
  const methodSummaries = data.summaries.filter(
    (s) => !("error" in s.means) && s.means.ece !== undefined
  );

  if (methodSummaries.length === 0) {
    return (
      <div className="empty-state py-8">
        <div className="text-sm text-navy-500">No multi-seed results available</div>
      </div>
    );
  }

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Method</th>
          <th>ECE ↓</th>
          <th>Brier ↓</th>
          <th>Coverage</th>
          <th>Reject Rate</th>
        </tr>
      </thead>
      <tbody>
        {methodSummaries.map((s) => {
          const fmt = (key: string) => {
            const mean = s.means[key];
            const std = s.stds[key];
            if (mean === undefined) return "—";
            return `${mean.toFixed(4)} ± ${std !== undefined ? std.toFixed(4) : "0.0000"}`;
          };
          return (
            <tr key={s.method}>
              <td className="font-mono text-xs font-semibold text-navy-700">{s.method}</td>
              <td className="font-mono text-xs">{fmt("ece")}</td>
              <td className="font-mono text-xs">{fmt("brier")}</td>
              <td className="font-mono text-xs">{fmt("coverage")}</td>
              <td className="font-mono text-xs">{fmt("rejection_rate")}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ─── Helper components ───

function MetricCard({
  icon,
  label,
  value,
  sublabel,
  provenance,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sublabel: string;
  provenance?: string;
}) {
  return (
    <div className="surface p-5">
      <div className="flex items-start justify-between mb-2">
        <span className="label">{label}</span>
        <span className="text-navy-300">{icon}</span>
      </div>
      <div className="text-2xl font-bold text-navy-700 tabular-nums">{value}</div>
      <div className="text-[11px] text-navy-400 mt-1">{sublabel}</div>
      {provenance && (
        <div className="mt-2">
          <ProvenanceChip provenance={provenance as Provenance} />
        </div>
      )}
    </div>
  );
}

type Provenance = "measured" | "illustrative" | "proposed" | "not-evaluated";

function ProvenanceChip({
  provenance,
  label,
}: {
  provenance: Provenance | string;
  label?: string;
}) {
  const styles: Record<string, string> = {
    measured: "chip-measured",
    illustrative: "chip-illustrative",
    proposed: "chip-proposed",
    "not-evaluated": "chip-not-evaluated",
  };
  return (
    <span className={`chip border ${styles[provenance] || "chip-not-evaluated"}`}>
      {label || provenance}
    </span>
  );
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}
