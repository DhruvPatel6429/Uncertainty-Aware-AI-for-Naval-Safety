"use client";

import { useState } from "react";
import {
  ArrowRight,
  X,
  AlertCircle,
  Layers,
  Cpu,
  Box,
  Settings,
} from "lucide-react";
import {
  pipelineStages,
  type PipelineStage,
} from "@/lib/data";

export default function ArchitecturePage() {
  const [selected, setSelected] = useState<PipelineStage | null>(null);

  return (
    <div className="page-container">
      {/* Page header */}
      <div>
        <h1 className="h-page">System Architecture</h1>
        <p className="body-secondary mt-1">
          End-to-end uncertainty-aware pipeline. The UQ layer threads calibrated
          confidence bounds through every stage. Click any stage to inspect its
          inputs, outputs, dependencies, and known limitations.
        </p>
      </div>

      {/* Pipeline diagram */}
      <div className="surface p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="h-section">Four-Stage Autonomous Navigation Pipeline</h2>
          <span className="label text-navy-400">Left-to-right data flow</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
          {pipelineStages.map((stage, i) => (
            <div key={stage.id} className="relative">
              <button
                onClick={() => setSelected(stage)}
                className="surface w-full text-left p-4 hover:border-navy-200 hover:bg-navy-50/30 transition-colors group cursor-pointer"
                style={{ borderTopWidth: 3, borderTopColor: stage.uqColor }}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="label text-navy-400 mb-1">Stage {stage.index}</div>
                    <div className="text-sm font-semibold text-navy-700">{stage.name}</div>
                  </div>
                  <div
                    className="w-7 h-7 rounded-md flex items-center justify-center text-white text-xs font-mono font-bold flex-shrink-0"
                    style={{ backgroundColor: stage.uqColor }}
                  >
                    {stage.index}
                  </div>
                </div>
                <p className="text-[11px] text-navy-500 leading-snug mb-3">
                  {stage.shortDescription}
                </p>
                <div className="pt-2 border-t border-navy-50">
                  <div className="label text-navy-400 mb-1">UQ Method</div>
                  <div className="text-[11px] font-medium" style={{ color: stage.uqColor }}>
                    {stage.primaryUQMethod}
                  </div>
                </div>
              </button>

              {i < pipelineStages.length - 1 && (
                <div className="hidden lg:flex absolute top-1/2 -right-2.5 -translate-y-1/2 z-10">
                  <ArrowRight size={16} className="text-navy-300" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* UQ layer */}
        <div className="mt-5 rounded-lg border border-gold-200 bg-gold-50/60 px-5 py-3">
          <div className="flex items-center justify-between mb-2">
            <span className="label-gold">Uncertainty Quantification Layer</span>
            <span className="text-[10px] text-gold-600">Threads through every stage</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center">
            {["Bayesian DL", "Conformal Prediction", "Deep Ensembles", "Evidential DL"].map((m) => (
              <div
                key={m}
                className="bg-white rounded-md py-2 text-[11px] font-medium text-navy-700 border border-gold-100"
              >
                {m}
              </div>
            ))}
          </div>
        </div>

        {/* Software stack */}
        <div className="mt-4 rounded-lg bg-navy-50 border border-navy-100 px-5 py-3">
          <div className="flex items-center gap-2 mb-2">
            <Layers size={12} className="text-navy-400" />
            <span className="label">Software Stack</span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-navy-600">
            {["PyTorch", "Hydra", "MLflow", "Docker", "FastAPI", "React", "TypeScript"].map((s, i) => (
              <span key={s} className="flex items-center gap-1.5">
                {i > 0 && <span className="text-navy-300">·</span>}
                <span className="font-medium">{s}</span>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Architectural principles */}
      <div className="surface p-6">
        <h2 className="h-section mb-4">Architectural Principles</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Principle
            icon={<Box size={14} />}
            title="Separation of Concerns"
            description="UQ layer is decoupled from pipeline stages. Each stage selects its UQ method via YAML configuration — no code changes required to swap methods."
          />
          <Principle
            icon={<Settings size={14} />}
            title="Plug-in Architecture"
            description="Each UQ method implements a uniform interface. Switching methods is a single-line config change. New methods are added by subclassing UQInterface and registering."
          />
          <Principle
            icon={<Cpu size={14} />}
            title="Audit-Trail by Design"
            description="Every decision is logged with its uncertainty decomposition, conformal risk calculation, and reject-option threshold check — the principal artefact for safety-assurance evidence."
          />
        </div>
      </div>

      {/* Stage detail drawer */}
      {selected && (
        <StageDrawer stage={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function Principle({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="border-l-2 border-gold-300 pl-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-gold-500">{icon}</span>
        <h3 className="text-sm font-semibold text-navy-700">{title}</h3>
      </div>
      <p className="text-xs text-navy-600 leading-relaxed">{description}</p>
    </div>
  );
}

// ─── Stage detail drawer ───

function StageDrawer({
  stage,
  onClose,
}: {
  stage: PipelineStage;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 bg-navy-700/30 z-40 flex justify-end"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md bg-white h-full overflow-y-auto shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Drawer header */}
        <div
          className="px-6 py-5 border-b border-navy-100"
          style={{ borderTopWidth: 3, borderTopColor: stage.uqColor }}
        >
          <div className="flex items-start justify-between">
            <div>
              <div className="label text-navy-400 mb-1">Stage {stage.index}</div>
              <h2 className="text-lg font-bold text-navy-700">{stage.name}</h2>
              <div className="text-xs mt-1 font-medium" style={{ color: stage.uqColor }}>
                {stage.primaryUQMethod}
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

        {/* Drawer body */}
        <div className="px-6 py-5 space-y-5">
          <section>
            <div className="label mb-1.5">Description</div>
            <p className="text-sm text-navy-600 leading-relaxed">{stage.detailedDescription}</p>
          </section>

          <section>
            <div className="label mb-1.5">Algorithmic Approach</div>
            <p className="text-sm text-navy-600 leading-relaxed">{stage.algorithmicApproach}</p>
          </section>

          <section className="grid grid-cols-2 gap-4">
            <div>
              <div className="label mb-1.5">Inputs</div>
              <ul className="space-y-1">
                {stage.inputs.map((x) => (
                  <li key={x} className="text-xs text-navy-600 flex items-start gap-1.5">
                    <span className="text-gold-400 mt-0.5">→</span>
                    <span>{x}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <div className="label mb-1.5">Outputs</div>
              <ul className="space-y-1">
                {stage.outputs.map((x) => (
                  <li key={x} className="text-xs text-navy-600 flex items-start gap-1.5">
                    <span className="text-green-500 mt-0.5">↳</span>
                    <span>{x}</span>
                  </li>
                ))}
              </ul>
            </div>
          </section>

          <section>
            <div className="label mb-1.5">Dependencies</div>
            <ul className="space-y-1">
              {stage.dependencies.map((d) => (
                <li key={d} className="text-xs text-navy-600 flex items-start gap-1.5">
                  <span className="text-navy-300 mt-0.5">○</span>
                  <span>{d}</span>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle size={12} className="text-amber-600" />
              <div className="label text-amber-700">Known Limitations</div>
            </div>
            <ul className="space-y-1.5">
              {stage.knownLimitations.map((l) => (
                <li key={l} className="text-xs text-navy-600 flex items-start gap-2">
                  <span className="text-amber-500 mt-0.5">—</span>
                  <span className="leading-relaxed">{l}</span>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
