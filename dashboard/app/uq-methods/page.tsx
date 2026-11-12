"use client";

import { useState } from "react";
import {
  Check,
  X,
  BookOpen,
  Cpu,
  AlertTriangle,
  Clock,
  GitBranch,
  FlaskConical,
} from "lucide-react";
import { uqMethods, type UQMethod, type DataProvenance } from "@/lib/data";
import UQRadarChart from "@/components/UQRadarChart";

export default function UQMethodsPage() {
  const [selectedId, setSelectedId] = useState(uqMethods[0].id);
  const selected = uqMethods.find((m) => m.id === selectedId)!;

  return (
    <div className="page-container">
      {/* Page header */}
      <div>
        <h1 className="h-page">UQ Methods — Scientific Comparison</h1>
        <p className="body-secondary mt-1">
          Four complementary uncertainty quantification techniques, each addressing
          a distinct failure mode of deterministic AI. No single method dominates
          all dimensions — the framework's plug-in architecture assigns each to
          the pipeline stage where its strengths are most valuable.
        </p>
      </div>

      {/* Method selector */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {uqMethods.map((m) => {
          const active = m.id === selectedId;
          return (
            <button
              key={m.id}
              onClick={() => setSelectedId(m.id)}
              className={`surface p-4 text-left transition-all ${
                active ? "border-navy-300 ring-1 ring-navy-200" : "hover:border-navy-200"
              }`}
              style={active ? { borderTopWidth: 3, borderTopColor: m.color } : { borderTopWidth: 3, borderTopColor: "transparent" }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: m.color }} />
                  <span className="label">{m.shortName}</span>
                </div>
                {active && <span className="text-[10px] text-navy-400 uppercase tracking-wider">Selected</span>}
              </div>
              <div className="text-sm font-semibold text-navy-700 mb-1">{m.name}</div>
              <div className="text-[11px] text-navy-500 leading-snug">{m.tagline}</div>
              <div className="mt-3 pt-2 border-t border-navy-50">
                <div className="label text-navy-400 mb-0.5">Assignment</div>
                <div className="text-[10px] text-navy-600 leading-snug">{m.pipelineAssignment}</div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Radar + Method detail */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        <div className="lg:col-span-3 surface">
          <div className="px-6 py-4 border-b border-navy-100 flex items-center justify-between">
            <h2 className="h-section">Comparative Profile</h2>
            <ProvenanceChip provenance="illustrative" label="1=weak · 5=strong" />
          </div>
          <div className="p-4">
            <UQRadarChart highlight={selected.id} />
          </div>
        </div>

        <div className="lg:col-span-2 surface">
          <div className="px-6 py-4 border-b border-navy-100">
            <h2 className="h-section">{selected.name}</h2>
            <p className="text-xs text-navy-500 mt-1 leading-snug">{selected.tagline}</p>
          </div>
          <div className="px-6 py-5 space-y-4">
            <section>
              <div className="label mb-1.5">Method Overview</div>
              <p className="text-xs text-navy-600 leading-relaxed">{selected.description}</p>
            </section>

            <section>
              <div className="label mb-1.5">Formulation</div>
              <div className="bg-navy-50 rounded-md px-3 py-2.5 font-mono text-[11px] text-navy-700 leading-relaxed">
                {selected.formulation}
              </div>
            </section>

            <section>
              <div className="label mb-1.5">Variants</div>
              <div className="space-y-1.5">
                {selected.variants.map((v) => (
                  <div key={v.name} className="text-xs">
                    <div className="font-semibold text-navy-700">{v.name}</div>
                    <div className="text-navy-500 leading-snug mt-0.5">{v.description}</div>
                  </div>
                ))}
              </div>
            </section>

            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <div className="label mb-1">Compute Cost</div>
                <div className="text-[11px] text-navy-600">{selected.computationalCost}</div>
              </div>
              <div>
                <div className="label mb-1">Inference Latency</div>
                <div className="text-[11px] text-navy-600">{selected.inferenceLatency}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Strengths / Weaknesses / Limitations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-3">
            <Check size={14} className="text-green-600" />
            <h3 className="h-card">Strengths</h3>
          </div>
          <ul className="space-y-2">
            {selected.strengths.map((s) => (
              <li key={s} className="text-xs text-navy-600 flex items-start gap-2 leading-relaxed">
                <span className="text-green-500 mt-0.5 flex-shrink-0">+</span>
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-3">
            <X size={14} className="text-amber-600" />
            <h3 className="h-card">Weaknesses</h3>
          </div>
          <ul className="space-y-2">
            {selected.weaknesses.map((s) => (
              <li key={s} className="text-xs text-navy-600 flex items-start gap-2 leading-relaxed">
                <span className="text-amber-500 mt-0.5 flex-shrink-0">−</span>
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={14} className="text-red-600" />
            <h3 className="h-card">Research Limitations</h3>
          </div>
          <ul className="space-y-2">
            {selected.limitations.map((s) => (
              <li key={s} className="text-xs text-navy-600 flex items-start gap-2 leading-relaxed">
                <span className="text-red-500 mt-0.5 flex-shrink-0">!</span>
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Comparison table */}
      <div className="surface">
        <div className="px-6 py-4 border-b border-navy-100 flex items-center justify-between">
          <h2 className="h-section">Side-by-side Comparison</h2>
          <ProvenanceChip provenance="illustrative" label="Method profile — not benchmark data" />
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Dimension</th>
                {uqMethods.map((m) => (
                  <th key={m.id}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: m.color }} />
                      {m.shortName}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { key: "coverage",    label: "Coverage Guarantee" },
                { key: "calibration", label: "Calibration Quality" },
                { key: "compute",     label: "Compute Efficiency" },
                { key: "ood",         label: "OOD Robustness" },
                { key: "scalability", label: "Scalability" },
                { key: "rigor",       label: "Theoretical Rigor" },
              ].map((dim) => (
                <tr key={dim.key}>
                  <td className="font-medium text-navy-700">{dim.label}</td>
                  {uqMethods.map((m) => {
                    // @ts-ignore
                    const v = m.metrics[dim.key];
                    return (
                      <td key={m.id}>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full bg-navy-100 overflow-hidden min-w-[60px]">
                            <div
                              className="h-full rounded-full"
                              style={{ width: `${(v / 5) * 100}%`, backgroundColor: m.color }}
                            />
                          </div>
                          <span className="text-[10px] text-navy-400 font-mono w-6">{v}/5</span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr className="border-t-2 border-navy-100">
                <td className="font-medium text-navy-700">Inference Latency</td>
                {uqMethods.map((m) => (
                  <td key={m.id} className="text-[11px] text-navy-600">
                    {m.inferenceLatency}
                  </td>
                ))}
              </tr>
              <tr>
                <td className="font-medium text-navy-700">Pipeline Assignment</td>
                {uqMethods.map((m) => (
                  <td key={m.id} className="text-[11px] text-navy-600">
                    {m.pipelineAssignment}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* References */}
      <div className="surface p-6">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen size={14} className="text-gold-500" />
          <h2 className="h-section">Key References — {selected.shortName}</h2>
        </div>
        <ol className="space-y-2">
          {selected.keyPapers.map((p, i) => (
            <li key={p.title} className="text-xs text-navy-600 flex gap-3 leading-relaxed">
              <span className="font-mono text-navy-400 flex-shrink-0">[{i + 1}]</span>
              <span>
                <span className="font-semibold text-navy-700">{p.author}</span>{" "}
                <span className="text-navy-400">({p.year})</span>.{" "}
                <span className="italic">{p.title}</span>.
              </span>
            </li>
          ))}
        </ol>
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
