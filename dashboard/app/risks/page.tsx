"use client";

import { useState } from "react";
import { X, AlertTriangle, ShieldCheck, AlertCircle } from "lucide-react";
import { risks, type Risk } from "@/lib/data";

function riskColor(l: number, i: number) {
  const score = l * i;
  if (score <= 6) return { bg: "#E5F0E0", border: "#7FB37A", text: "#3F8A5E", label: "Low" };
  if (score <= 12) return { bg: "#FBEED0", border: "#E0A53D", text: "#8B6320", label: "Medium" };
  return { bg: "#F4D6D2", border: "#B5443A", text: "#7A2A22", label: "High" };
}

export default function RisksPage() {
  const [selected, setSelected] = useState<Risk | null>(risks[0]);

  const high = risks.filter((r) => r.likelihood * r.impact > 12);
  const medium = risks.filter((r) => {
    const s = r.likelihood * r.impact;
    return s > 6 && s <= 12;
  });
  const low = risks.filter((r) => r.likelihood * r.impact <= 6);

  return (
    <div className="page-container">
      {/* Header */}
      <div>
        <h1 className="h-page">Risk Analysis</h1>
        <p className="body-secondary mt-1">
          Eight identified technical and operational risks with concrete mitigation
          strategies. Click any cell in the heat map to inspect the risk detail.
        </p>
      </div>

      {/* Risk summary */}
      <div className="grid grid-cols-3 gap-4">
        <RiskSummaryCard
          count={high.length}
          label="High Risk"
          description="Mitigate actively"
          color={riskColor(5, 5)}
        />
        <RiskSummaryCard
          count={medium.length}
          label="Medium Risk"
          description="Monitor continuously"
          color={riskColor(3, 3)}
        />
        <RiskSummaryCard
          count={low.length}
          label="Low Risk"
          description="Accept and document"
          color={riskColor(1, 1)}
        />
      </div>

      {/* Heat map + Risk table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Heat map */}
        <div className="lg:col-span-2 surface">
          <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
            <h2 className="h-card">Likelihood × Impact Heat Map</h2>
            <span className="text-[10px] text-navy-400">Click a cell to inspect</span>
          </div>
          <div className="p-6">
            <HeatMap risks={risks} selected={selected} onSelect={setSelected} />
          </div>
        </div>

        {/* Selected risk detail */}
        <div className="surface">
          <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
            <h2 className="h-card">{selected ? selected.id : "Select a Risk"}</h2>
            {selected && (
              <span
                className="chip border"
                style={{
                  backgroundColor: riskColor(selected.likelihood, selected.impact).bg,
                  color: riskColor(selected.likelihood, selected.impact).text,
                }}
              >
                {riskColor(selected.likelihood, selected.impact).label}
              </span>
            )}
          </div>
          <div className="p-5 space-y-4">
            {selected ? (
              <RiskDetail risk={selected} />
            ) : (
              <div className="empty-state py-8">
                <div className="empty-state-icon">
                  <AlertCircle size={18} />
                </div>
                <div className="text-xs text-navy-500">Click a heat-map cell</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Complete risk register */}
      <div className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100">
          <h2 className="h-card">Complete Risk Register</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="w-12">#</th>
                <th>Risk</th>
                <th className="w-24">Category</th>
                <th className="w-12 text-center">L</th>
                <th className="w-12 text-center">I</th>
                <th className="w-20 text-center">Level</th>
                <th>Mitigation</th>
              </tr>
            </thead>
            <tbody>
              {risks.map((r) => {
                const c = riskColor(r.likelihood, r.impact);
                return (
                  <tr
                    key={r.id}
                    onClick={() => setSelected(r)}
                    className="cursor-pointer"
                  >
                    <td>
                      <span className="font-mono text-xs font-bold text-navy-700">{r.id}</span>
                    </td>
                    <td className="text-xs font-medium text-navy-700">{r.label}</td>
                    <td>
                      <span className="text-[10px] text-navy-500 uppercase tracking-wider">
                        {r.category}
                      </span>
                    </td>
                    <td className="text-center font-mono text-xs text-navy-700">{r.likelihood}</td>
                    <td className="text-center font-mono text-xs text-navy-700">{r.impact}</td>
                    <td className="text-center">
                      <span
                        className="chip border text-[10px]"
                        style={{ backgroundColor: c.bg, color: c.text }}
                      >
                        {c.label}
                      </span>
                    </td>
                    <td className="text-[11px] text-navy-600 leading-snug">{r.mitigation}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail drawer */}
      {selected && (
        <RiskDrawer risk={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function RiskSummaryCard({
  count,
  label,
  description,
  color,
}: {
  count: number;
  label: string;
  description: string;
  color: { bg: string; border: string; text: string; label: string };
}) {
  return (
    <div
      className="surface p-5 border-l-4"
      style={{ borderLeftColor: color.border }}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="label" style={{ color: color.text }}>{label}</span>
        {label === "High Risk" ? (
          <AlertTriangle size={14} style={{ color: color.text }} />
        ) : label === "Low Risk" ? (
          <ShieldCheck size={14} style={{ color: color.text }} />
        ) : (
          <AlertCircle size={14} style={{ color: color.text }} />
        )}
      </div>
      <div className="text-3xl font-bold text-navy-700 tabular-nums">{count}</div>
      <div className="text-[11px] text-navy-500 mt-1">{description}</div>
    </div>
  );
}

function HeatMap({
  risks,
  selected,
  onSelect,
}: {
  risks: Risk[];
  selected: Risk | null;
  onSelect: (r: Risk) => void;
}) {
  return (
    <div className="flex">
      {/* Y-axis label */}
      <div className="flex items-center mr-3">
        <div className="label text-navy-400 rotate-180 [writing-mode:vertical-rl]">
          Impact →
        </div>
      </div>

      <div className="flex-1">
        {/* Grid */}
        <div className="grid grid-cols-5 gap-1.5">
          {Array.from({ length: 5 }, (_, rowIdx) => {
            const impact = 5 - rowIdx;
            return Array.from({ length: 5 }, (_, colIdx) => {
              const likelihood = colIdx + 1;
              const cellRisks = risks.filter(
                (r) => r.likelihood === likelihood && r.impact === impact
              );
              const c = riskColor(likelihood, impact);
              return (
                <div
                  key={`${impact}-${likelihood}`}
                  className="aspect-square rounded-md border flex flex-col items-center justify-center p-1 relative cursor-pointer hover:scale-[1.04] transition-transform"
                  style={{ backgroundColor: c.bg, borderColor: c.border }}
                  onClick={() => cellRisks[0] && onSelect(cellRisks[0])}
                  title={`L=${likelihood}, I=${impact}`}
                >
                  {cellRisks.map((r) => (
                    <span
                      key={r.id}
                      className="text-[9px] font-bold leading-tight"
                      style={{ color: c.text }}
                    >
                      {r.id}
                    </span>
                  ))}
                </div>
              );
            });
          }).flat()}
        </div>

        {/* X-axis label */}
        <div className="label text-navy-400 text-center mt-2">
          Likelihood →
        </div>

        {/* X-axis tick labels */}
        <div className="grid grid-cols-5 gap-1.5 mt-1">
          {["Rare", "Unlikely", "Possible", "Likely", "Frequent"].map((l) => (
            <div key={l} className="text-[9px] text-navy-400 text-center">{l}</div>
          ))}
        </div>
      </div>

      {/* Right margin Y-axis labels */}
      <div className="flex flex-col justify-between ml-3 py-0">
        {["Severe", "Major", "Significant", "Moderate", "Minor"].map((l) => (
          <div key={l} className="text-[9px] text-navy-400">{l}</div>
        ))}
      </div>
    </div>
  );
}

function RiskDetail({ risk }: { risk: Risk }) {
  const c = riskColor(risk.likelihood, risk.impact);
  return (
    <>
      <div>
        <div className="label mb-1">Risk</div>
        <div className="text-sm font-semibold text-navy-700">{risk.label}</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="label mb-1">Likelihood</div>
          <div className="font-mono text-xl font-bold text-navy-700">
            {risk.likelihood}/5
          </div>
        </div>
        <div>
          <div className="label mb-1">Impact</div>
          <div className="font-mono text-xl font-bold text-navy-700">
            {risk.impact}/5
          </div>
        </div>
      </div>
      <div>
        <div className="label mb-1">Category</div>
        <div className="text-xs text-navy-600 capitalize">{risk.category}</div>
      </div>
      <div>
        <div className="label mb-1">Mitigation</div>
        <div className="text-xs text-navy-600 leading-relaxed">{risk.mitigation}</div>
      </div>
      {risk.residualRisk && (
        <div>
          <div className="label mb-1">Residual Risk</div>
          <div className="text-xs text-navy-600 leading-relaxed">{risk.residualRisk}</div>
        </div>
      )}
    </>
  );
}

function RiskDrawer({ risk, onClose }: { risk: Risk; onClose: () => void }) {
  const c = riskColor(risk.likelihood, risk.impact);
  return (
    <div
      className="fixed inset-0 bg-navy-700/30 z-40 flex justify-end"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md bg-white h-full overflow-y-auto shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="px-6 py-5 border-b border-navy-100"
          style={{ borderTopWidth: 3, borderTopColor: c.border }}
        >
          <div className="flex items-start justify-between">
            <div>
              <div className="label text-navy-400 mb-1">{risk.id}</div>
              <h2 className="text-lg font-bold text-navy-700">{risk.label}</h2>
              <div className="flex items-center gap-2 mt-2">
                <span
                  className="chip border"
                  style={{ backgroundColor: c.bg, color: c.text }}
                >
                  {c.label} Risk
                </span>
                <span className="text-[10px] text-navy-400 uppercase tracking-wider">
                  {risk.category}
                </span>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-navy-400 hover:text-navy-700 transition-colors"
              aria-label="Close"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="px-6 py-5 space-y-5">
          <section className="grid grid-cols-2 gap-4">
            <div>
              <div className="label mb-1">Likelihood</div>
              <div className="font-mono text-2xl font-bold text-navy-700">
                {risk.likelihood}<span className="text-sm text-navy-300">/5</span>
              </div>
            </div>
            <div>
              <div className="label mb-1">Impact</div>
              <div className="font-mono text-2xl font-bold text-navy-700">
                {risk.impact}<span className="text-sm text-navy-300">/5</span>
              </div>
            </div>
          </section>

          <section>
            <div className="label mb-1.5">Mitigation Strategy</div>
            <p className="text-sm text-navy-600 leading-relaxed">{risk.mitigation}</p>
          </section>

          {risk.residualRisk && (
            <section>
              <div className="flex items-center gap-2 mb-1.5">
                <AlertCircle size={12} className="text-amber-600" />
                <div className="label text-amber-700">Residual Risk</div>
              </div>
              <p className="text-sm text-navy-600 leading-relaxed">{risk.residualRisk}</p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
