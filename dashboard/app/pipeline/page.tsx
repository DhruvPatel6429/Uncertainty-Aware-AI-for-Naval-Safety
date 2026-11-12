"use client";

import { useState } from "react";
import {
  Ship,
  Crosshair,
  Activity,
  Route,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Info,
} from "lucide-react";
import {
  maritimeDetections,
  maritimeScenarioName,
  scenarioProvenance,
  ownVesselState,
  type MaritimeDetection,
  type DataProvenance,
} from "@/lib/data";

export default function PipelinePage() {
  const [selected, setSelected] = useState<MaritimeDetection | null>(maritimeDetections[0]);

  return (
    <div className="page-container">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="h-page">Pipeline Demonstration</h1>
          <p className="body-secondary mt-1">
            Interactive maritime perception, trajectory forecasting, and decision
            analysis. Click any contact in the scene to inspect its uncertainty
            decomposition and forecast.
          </p>
        </div>
        <ProvenanceChip provenance={scenarioProvenance} label={`Scenario: ${maritimeScenarioName}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        {/* ─── LEFT: Maritime scene ─── */}
        <div className="lg:col-span-3 surface">
          <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
            <h2 className="h-card">Maritime Scene — Tactical View</h2>
            <span className="text-[10px] text-navy-400 uppercase tracking-wider">Top-down · North-up</span>
          </div>
          <div className="p-4">
            <MaritimeScene
              detections={maritimeDetections}
              selectedId={selected?.id}
              onSelect={setSelected}
            />

            {/* Legend */}
            <div className="mt-4 pt-3 border-t border-navy-50 flex flex-wrap gap-x-4 gap-y-2 text-[10px]">
              <LegendItem color="#0F2845" label="Own vessel" shape="triangle" />
              <LegendItem color="#3F8A5E" label="Accepted detection" shape="circle" />
              <LegendItem color="#E0A53D" label="High epistemic" shape="circle" />
              <LegendItem color="#B5443A" label="OOD / rejected" shape="diamond" />
              <LegendItem color="#C9963A" label="Predicted trajectory" shape="dashed-line" />
              <LegendItem color="rgba(201,150,58,0.2)" label="Uncertainty cone" shape="cone" />
            </div>
          </div>
        </div>

        {/* ─── RIGHT: Technical analysis ─── */}
        <div className="lg:col-span-2 space-y-5">
          <DetectionInspection detection={selected} />
          {selected?.forecast && <TrajectoryAnalysis detection={selected} />}
        </div>
      </div>

      {/* Decision analysis */}
      <DecisionAnalysis detections={maritimeDetections} />
    </div>
  );
}

// ─── Maritime scene SVG visualization ───

function MaritimeScene({
  detections,
  selectedId,
  onSelect,
}: {
  detections: MaritimeDetection[];
  selectedId?: string;
  onSelect: (d: MaritimeDetection) => void;
}) {
  return (
    <div className="relative aspect-[4/3] bg-navy-50 rounded-md overflow-hidden border border-navy-100">
      <svg
        viewBox="0 0 400 300"
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#E1E8F2" strokeWidth="0.5" />
          </pattern>
          <pattern id="sea" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="10" cy="10" r="0.5" fill="#C2D0E2" opacity="0.4" />
          </pattern>
        </defs>
        <rect width="400" height="300" fill="url(#sea)" />
        <rect width="400" height="300" fill="url(#grid)" />

        {/* Compass / scale */}
        <g transform="translate(360, 30)">
          <circle cx="0" cy="0" r="14" fill="white" stroke="#C2D0E2" strokeWidth="1" />
          <path d="M 0 -10 L 3 0 L 0 -2 L -3 0 Z" fill="#0F2845" />
          <path d="M 0 10 L 3 0 L 0 2 L -3 0 Z" fill="#8FA3BF" />
          <text x="0" y="-16" textAnchor="middle" fontSize="8" fill="#5C78A0" fontWeight="600">N</text>
        </g>

        {/* Scale */}
        <g transform="translate(20, 275)">
          <line x1="0" y1="0" x2="40" y2="0" stroke="#5C78A0" strokeWidth="1.5" />
          <line x1="0" y1="-3" x2="0" y2="3" stroke="#5C78A0" strokeWidth="1.5" />
          <line x1="40" y1="-3" x2="40" y2="3" stroke="#5C78A0" strokeWidth="1.5" />
          <text x="20" y="12" textAnchor="middle" fontSize="8" fill="#5C78A0">~200 m</text>
        </g>

        {/* Own vessel */}
        <g transform={`translate(${ownVesselState.x * 400}, ${ownVesselState.y * 300})`}>
          <polygon
            points="0,-10 6,8 0,4 -6,8"
            fill="#0F2845"
            stroke="white"
            strokeWidth="1"
          />
          <text x="0" y="-14" textAnchor="middle" fontSize="8" fill="#0F2845" fontWeight="700">
            OWN
          </text>
          {/* Planned path */}
          <path
            d={`M 0 0 L 0 -60`}
            stroke="#0F2845"
            strokeWidth="1.5"
            strokeDasharray="4 3"
            fill="none"
            opacity="0.6"
          />
        </g>

        {/* Detections */}
        {detections.map((d) => {
          const cx = d.x * 400;
          const cy = d.y * 300;
          const isSel = d.id === selectedId;
          const color = detectionColor(d);
          return (
            <g
              key={d.id}
              transform={`translate(${cx}, ${cy})`}
              onClick={() => onSelect(d)}
              className="cursor-pointer"
            >
              {/* Uncertainty cone */}
              {d.forecast && (
                <UncertaintyCone
                  heading={d.heading}
                  halfWidthDeg={d.forecast.uncertaintyConeDeg}
                  color={color}
                />
              )}

              {/* Trajectory forecast line */}
              {d.forecast && (
                <TrajectoryLine
                  heading={d.heading}
                  color={color}
                />
              )}

              {/* Detection marker */}
              <DetectionMarker type={d.detectionType} color={color} selected={isSel} />

              {/* Label */}
              <text
                x="0"
                y="-14"
                textAnchor="middle"
                fontSize="8"
                fill={color}
                fontWeight={isSel ? 700 : 500}
              >
                {d.label}
              </text>
              <text
                x="0"
                y="20"
                textAnchor="middle"
                fontSize="7"
                fill="#5C78A0"
              >
                {d.id}
              </text>

              {/* Selection ring */}
              {isSel && (
                <circle
                  cx="0"
                  cy="0"
                  r="14"
                  fill="none"
                  stroke={color}
                  strokeWidth="1.5"
                  strokeDasharray="2 2"
                />
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function detectionColor(d: MaritimeDetection): string {
  if (d.detectionType === "ood") return "#B5443A";
  if (d.detectionType === "high-uncertainty") return "#E0A53D";
  return "#3F8A5E";
}

function DetectionMarker({
  type,
  color,
  selected,
}: {
  type: string;
  color: string;
  selected: boolean;
}) {
  if (type === "ood") {
    // Diamond
    return (
      <g>
        <polygon points="0,-6 6,0 0,6 -6,0" fill={color} stroke="white" strokeWidth="1" />
        {selected && <polygon points="0,-9 9,0 0,9 -9,0" fill="none" stroke={color} strokeWidth="1" />}
      </g>
    );
  }
  // Circle
  return (
    <g>
      <circle cx="0" cy="0" r="5" fill={color} stroke="white" strokeWidth="1" />
      {selected && <circle cx="0" cy="0" r="8" fill="none" stroke={color} strokeWidth="1" />}
    </g>
  );
}

function TrajectoryLine({ heading, color }: { heading: number; color: string }) {
  const rad = (heading - 90) * (Math.PI / 180);
  const length = 50;
  const x2 = Math.cos(rad) * length;
  const y2 = Math.sin(rad) * length;
  return (
    <line
      x1="0"
      y1="0"
      x2={x2}
      y2={y2}
      stroke={color}
      strokeWidth="1.2"
      strokeDasharray="3 2"
      opacity="0.7"
    />
  );
}

function UncertaintyCone({
  heading,
  halfWidthDeg,
  color,
}: {
  heading: number;
  halfWidthDeg: number;
  color: string;
}) {
  const length = 50;
  const baseRad = (heading - 90) * (Math.PI / 180);
  const halfWidthRad = halfWidthDeg * (Math.PI / 180);
  const leftRad = baseRad - halfWidthRad;
  const rightRad = baseRad + halfWidthRad;
  const x1 = Math.cos(leftRad) * length;
  const y1 = Math.sin(leftRad) * length;
  const x2 = Math.cos(rightRad) * length;
  const y2 = Math.sin(rightRad) * length;
  return (
    <path
      d={`M 0 0 L ${x1} ${y1} A ${length} ${length} 0 0 1 ${x2} ${y2} Z`}
      fill={color}
      opacity="0.12"
      stroke={color}
      strokeWidth="0.5"
      strokeDasharray="1 1"
      opacity-stroke="0.3"
    />
  );
}

function LegendItem({
  color,
  label,
  shape,
}: {
  color: string;
  label: string;
  shape: string;
}) {
  return (
    <div className="flex items-center gap-1.5 text-navy-500">
      {shape === "circle" && <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />}
      {shape === "triangle" && (
        <svg width="10" height="10" viewBox="-5 -5 10 10">
          <polygon points="0,-4 3,3 0,1 -3,3" fill={color} />
        </svg>
      )}
      {shape === "diamond" && (
        <svg width="10" height="10" viewBox="-5 -5 10 10">
          <polygon points="0,-4 4,0 0,4 -4,0" fill={color} />
        </svg>
      )}
      {shape === "dashed-line" && (
        <svg width="14" height="4">
          <line x1="0" y1="2" x2="14" y2="2" stroke={color} strokeWidth="1.5" strokeDasharray="3 2" />
        </svg>
      )}
      {shape === "cone" && <div className="w-3 h-2.5" style={{ backgroundColor: color, opacity: 0.5 }} />}
      <span>{label}</span>
    </div>
  );
}

// ─── Detection inspection panel ───

function DetectionInspection({ detection }: { detection: MaritimeDetection | null }) {
  if (!detection) {
    return (
      <div className="surface">
        <div className="px-5 py-3.5 border-b border-navy-100">
          <h2 className="h-card">Detection Inspection</h2>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">
            <Crosshair size={20} />
          </div>
          <div className="text-sm text-navy-500">Select a contact to inspect</div>
        </div>
      </div>
    );
  }

  const color = detectionColor(detection);

  return (
    <div className="surface">
      <div className="px-5 py-3.5 border-b border-navy-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-navy-400">{detection.id}</span>
          <h2 className="h-card">{detection.label}</h2>
        </div>
        <span className="chip border" style={{ backgroundColor: `${color}15`, color }}>
          {detection.detectionType === "ood" ? "OOD" : detection.detectionType === "high-uncertainty" ? "HIGH UNC" : "ACCEPTED"}
        </span>
      </div>
      <div className="p-5 space-y-4">
        {/* Classification + state */}
        <div className="grid grid-cols-2 gap-3 text-xs">
          <Field label="Classification" value={detection.classification} />
          <Field label="Source Model" value={detection.sourceModel} />
          <Field label="Heading" value={`${detection.heading}°`} />
          <Field label="Speed" value={`${detection.speed} kn`} />
        </div>

        {/* Confidence + uncertainty */}
        <div className="grid grid-cols-3 gap-2">
          <MetricBox label="Confidence" value={detection.confidence} unit="" color="#0F2845" />
          <MetricBox label="Epistemic" value={detection.epistemicUnc} unit="" color="#E0A53D" />
          <MetricBox label="Aleatoric" value={detection.aleatoricUnc} unit="" color="#8FA3BF" />
        </div>

        {/* Status */}
        <div className="pt-2 border-t border-navy-50">
          {detection.oodIndicator ? (
            <div className="rounded-md bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-700 flex items-center gap-2">
              <XCircle size={13} />
              <div>
                <div className="font-semibold">OUT-OF-DISTRIBUTION</div>
                <div className="text-[10px] text-red-600 mt-0.5">
                  Reject option triggered — escalate to human supervisor.
                </div>
              </div>
            </div>
          ) : detection.epistemicUnc !== null && detection.epistemicUnc > 0.35 ? (
            <div className="rounded-md bg-amber-50 border border-amber-200 px-3 py-2 text-xs text-amber-700 flex items-center gap-2">
              <AlertTriangle size={13} />
              <div>
                <div className="font-semibold">HIGH EPISTEMIC UNCERTAINTY</div>
                <div className="text-[10px] text-amber-600 mt-0.5">
                  Flagged for human review — confidence below operational threshold.
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-md bg-green-50 border border-green-200 px-3 py-2 text-xs text-green-700 flex items-center gap-2">
              <CheckCircle2 size={13} />
              <div>
                <div className="font-semibold">ACCEPTED</div>
                <div className="text-[10px] text-green-600 mt-0.5">
                  Detection within operational envelope — propagated to prediction.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Trajectory analysis ───

function TrajectoryAnalysis({ detection }: { detection: MaritimeDetection }) {
  if (!detection.forecast) {
    return null;
  }
  const f = detection.forecast;
  return (
    <div className="surface">
      <div className="px-5 py-3.5 border-b border-navy-100 flex items-center justify-between">
        <h2 className="h-card">Trajectory Forecast</h2>
        <ProvenanceChip provenance={f.provenance} />
      </div>
      <div className="p-5 space-y-4">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <Field label="Horizon" value={`${f.horizonSeconds} s`} />
          <Field label="Uncertainty Cone (95%)" value={`±${f.uncertaintyConeDeg}°`} />
        </div>

        {/* Course forecast mini-chart */}
        <div>
          <div className="label mb-2">Predicted Course (deg)</div>
          <div className="bg-navy-50 rounded-md p-3">
            <svg viewBox="0 0 200 60" className="w-full h-12">
              <line x1="0" y1="30" x2="200" y2="30" stroke="#E1E8F2" strokeWidth="0.5" />
              {f.coursePredictions.map((c, i) => {
                const x = (i / Math.max(f.coursePredictions.length - 1, 1)) * 200;
                const y = 30 - (c - 80) * 0.5; // simple scaling
                return <circle key={i} cx={x} cy={y} r="2" fill={detectionColor(detection)} />;
              })}
              {f.coursePredictions.length > 1 && (
                <polyline
                  points={f.coursePredictions.map((c, i) => {
                    const x = (i / Math.max(f.coursePredictions.length - 1, 1)) * 200;
                    const y = 30 - (c - 80) * 0.5;
                    return `${x},${y}`;
                  }).join(" ")}
                  fill="none"
                  stroke={detectionColor(detection)}
                  strokeWidth="1.5"
                />
              )}
            </svg>
          </div>
        </div>

        <div className="text-[11px] text-navy-500 leading-relaxed bg-navy-50 rounded-md p-3">
          <Info size={11} className="inline mr-1 text-navy-400" />
          Forecast produced by the trajectory prediction module (Stage 2). Uncertainty
          cone grows with horizon, reflecting accumulating epistemic uncertainty.
        </div>
      </div>
    </div>
  );
}

// ─── Decision analysis ───

function DecisionAnalysis({ detections }: { detections: MaritimeDetection[] }) {
  const nOod = detections.filter((d) => d.detectionType === "ood").length;
  const nHighUnc = detections.filter((d) => d.detectionType === "high-uncertainty").length;
  const nAccepted = detections.filter((d) => d.detectionType === "accepted").length;
  const maxEpistemic = Math.max(...detections.map((d) => d.epistemicUnc ?? 0));
  const totalEpistemic = detections.reduce((s, d) => s + (d.epistemicUnc ?? 0), 0);
  const meanEpistemic = totalEpistemic / detections.length;
  const shouldReject = maxEpistemic > 0.7;
  const decision = shouldReject ? "REJECT_HUMAN" : "EXECUTE";
  const reason = shouldReject
    ? `Maximum epistemic uncertainty (${maxEpistemic.toFixed(2)}) exceeds operational threshold (0.70).`
    : "All detections within operational envelope. Conformal risk bound satisfied.";

  return (
    <div className="surface">
      <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
        <h2 className="h-card">Decision Analysis</h2>
        <ProvenanceChip provenance="illustrative" label="Stage 4 — Conformal Risk Control" />
      </div>
      <div className="p-6 space-y-5">
        {/* Decision banner */}
        <div
          className={`rounded-lg border px-4 py-3 flex items-center gap-3 ${
            decision === "EXECUTE"
              ? "bg-green-50 border-green-200"
              : "bg-amber-50 border-amber-200"
          }`}
        >
          {decision === "EXECUTE" ? (
            <CheckCircle2 size={20} className="text-green-600" />
          ) : (
            <AlertTriangle size={20} className="text-amber-600" />
          )}
          <div className="flex-1">
            <div
              className={`text-sm font-bold ${
                decision === "EXECUTE" ? "text-green-700" : "text-amber-700"
              }`}
            >
              {decision}
            </div>
            <div className="text-[11px] text-navy-600 mt-0.5">{reason}</div>
          </div>
        </div>

        {/* Aggregated metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricTile label="Total Detections" value={String(detections.length)} />
          <MetricTile label="Accepted" value={String(nAccepted)} color="#3F8A5E" />
          <MetricTile label="High Uncertainty" value={String(nHighUnc)} color="#E0A53D" />
          <MetricTile label="OOD / Rejected" value={String(nOod)} color="#B5443A" />
        </div>

        {/* Risk assessment */}
        <div className="border border-navy-100 rounded-md p-4 space-y-3">
          <div className="label">Pipeline-Level Risk Assessment</div>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <Field label="Mean Epistemic Uncertainty" value={meanEpistemic.toFixed(3)} />
            <Field label="Max Epistemic Uncertainty" value={maxEpistemic.toFixed(3)} />
            <Field label="COLREGs Compliance" value="Conditional" />
            <Field label="Conformal Bound (α)" value="Conditional on exchangeability" />
          </div>
          <div className="text-[11px] text-navy-500 leading-relaxed bg-navy-50 rounded p-2.5 mt-2">
            <Info size={11} className="inline mr-1 text-navy-400" />
            The decision module aggregates per-stage epistemic uncertainty into a
            pipeline-level risk score. The conformal risk bound is statistical and
            conditional on calibration data exchangeability — it is not a physical
            collision probability guarantee.
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Shared small components ───

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="label text-navy-400 mb-0.5">{label}</div>
      <div className="text-navy-700 font-medium text-xs">{value}</div>
    </div>
  );
}

function MetricBox({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: number | null;
  unit: string;
  color: string;
}) {
  return (
    <div className="rounded-md border border-navy-100 p-3 bg-white">
      <div className="label text-navy-400 mb-1">{label}</div>
      {value !== null ? (
        <div className="flex items-baseline gap-0.5">
          <span className="text-lg font-bold font-mono tabular-nums" style={{ color }}>
            {(value * 100).toFixed(0)}
          </span>
          <span className="text-[10px] text-navy-400">%</span>
        </div>
      ) : (
        <div className="text-sm text-navy-400 italic">Unavailable</div>
      )}
    </div>
  );
}

function MetricTile({
  label,
  value,
  color = "#0F2845",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="rounded-md border border-navy-100 p-3 bg-white">
      <div className="label text-navy-400 mb-1">{label}</div>
      <div className="text-2xl font-bold tabular-nums" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function ProvenanceChip({
  provenance,
  label,
}: {
  provenance: DataProvenance;
  label?: string;
}) {
  const styles: Record<DataProvenance, string> = {
    "measured": "chip-measured",
    "illustrative": "chip-illustrative",
    "proposed": "chip-proposed",
    "not-evaluated": "chip-not-evaluated",
  };
  return (
    <span className={`chip border ${styles[provenance]}`}>
      {label || provenance}
    </span>
  );
}
