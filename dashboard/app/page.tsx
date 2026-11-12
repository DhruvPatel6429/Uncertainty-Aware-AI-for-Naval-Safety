import Link from "next/link";
import {
  ArrowRight,
  Brain,
  Network,
  ShieldAlert,
  FileCheck,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  FlaskConical,
  SplitSquareHorizontal,
} from "lucide-react";
import { uqMethods, deliverables, pipelineStages } from "@/lib/data";

export default function OverviewPage() {
  return (
    <div className="page-container">
      {/* ─── Hero ─── */}
      <section className="surface-elevated overflow-hidden">
        <div className="bg-navy-700 px-8 py-10 md:py-12 relative">
          <div className="absolute top-0 right-0 w-1/3 h-full">
            <div className="absolute top-0 right-0 w-px h-full bg-gold-400/30 mr-12" />
          </div>
          <div className="relative max-w-4xl">
            <div className="label-gold mb-3">NRB · DRDO Research Proposal</div>
            <h1 className="h-display mb-4">UQ-AI Naval</h1>
            <p className="text-lg text-navy-100 leading-relaxed max-w-3xl">
              Uncertainty-Aware AI for Safety-Critical Naval Applications — an
              integrated research framework for uncertainty-quantified perception,
              trajectory forecasting, uncertainty-aware path planning and auditable
              decision-making in autonomous maritime environments.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link href="/architecture" className="btn-gold">
                Explore System Architecture <ArrowRight size={14} />
              </Link>
              <Link
                href="/pipeline"
                className="inline-flex items-center justify-center gap-1.5 px-4 py-2 text-white border border-navy-500 rounded-md text-sm font-medium hover:bg-navy-600 transition-colors"
              >
                Open Pipeline Demonstration
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Research attributes ─── */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <ResearchAttribute
          icon={<Brain size={16} />}
          value="4"
          label="UQ methodologies"
          sublabel="Bayesian · Conformal · Ensemble · Evidential"
        />
        <ResearchAttribute
          icon={<Network size={16} />}
          value="4"
          label="Autonomy pipeline stages"
          sublabel="Perception · Prediction · Planning · Decision"
        />
        <ResearchAttribute
          icon={<FileCheck size={16} />}
          value="5"
          label="Research deliverables"
          sublabel="D1 software stack → D5 DRDO package"
        />
        <ResearchAttribute
          icon={<Calendar size={16} />}
          value="12"
          label="Months proposed duration"
          sublabel="5 phases · 5 milestones"
        />
      </section>

      {/* ─── Problem & Solution ─── */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle size={16} className="text-red-600" />
            <h2 className="h-section">The Problem</h2>
          </div>
          <p className="body-text mb-4">
            Deterministic deep-learning systems deployed in naval autonomy produce
            point predictions with poorly calibrated softmax scores. They cannot
            distinguish high-confidence correct detections from low-confidence
            guesses, and propagate overconfident errors through the pipeline.
          </p>
          <ul className="space-y-2.5 text-sm text-navy-600">
            <ProblemItem>
              <strong>Poor confidence calibration.</strong> ECE of 5–15% on
              benchmarks, two to three times worse under distribution shift.
            </ProblemItem>
            <ProblemItem>
              <strong>Epistemic uncertainty unmodelled.</strong> No mechanism to
              distinguish aleatoric noise from model ignorance.
            </ProblemItem>
            <ProblemItem>
              <strong>Distribution shift brittleness.</strong> Silent performance
              degradation as sea state or sensor configuration drifts.
            </ProblemItem>
            <ProblemItem>
              <strong>No principled reject option.</strong> Forced to emit a
              prediction even when epistemic uncertainty is high.
            </ProblemItem>
            <ProblemItem>
              <strong>Unsafe error propagation.</strong> Overconfident perception
              feeds overconfident planning — catastrophic in collision avoidance.
            </ProblemItem>
          </ul>
        </div>

        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 size={16} className="text-green-600" />
            <h2 className="h-section">The UQ-AI Approach</h2>
          </div>
          <p className="body-text mb-4">
            An integrated, open-source software framework that threads calibrated
            uncertainty quantification through every stage of the autonomy pipeline,
            producing confidence-bounded outputs suitable for autonomous execution
            or human-in-the-loop review.
          </p>
          <ul className="space-y-2.5 text-sm text-navy-600">
            <ProblemItem>
              <strong>Bayesian perception.</strong> MC Dropout + Laplace on YOLO/U-Net
              backbones captures epistemic uncertainty per detection.
            </ProblemItem>
            <ProblemItem>
              <strong>Calibrated predictions.</strong> Conformal calibration wraps the
              detector to provide distribution-free coverage on the detection set.
            </ProblemItem>
            <ProblemItem>
              <strong>Ensemble-based trajectory forecasting.</strong> Deep ensembles
              faithfully reflect multi-modal vessel behaviour.
            </ProblemItem>
            <ProblemItem>
              <strong>Uncertainty-aware planning.</strong> A*/RRT* with cost function
              penalising high-uncertainty regions; COLREGs as soft constraints.
            </ProblemItem>
            <ProblemItem>
              <strong>Conformal risk control.</strong> Bounds expected collision
              probability below user-specified α; reject option for graceful degradation.
            </ProblemItem>
            <ProblemItem>
              <strong>Human-in-the-loop escalation.</strong> Structured handoff when
              epistemic uncertainty exceeds operational threshold.
            </ProblemItem>
          </ul>
        </div>
      </section>

      {/* ─── Four UQ Methods ─── */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="h-section">Four UQ Methodologies</h2>
            <p className="body-secondary mt-0.5">
              Each method addresses a distinct failure mode of deterministic AI.
              Click to inspect the full technical comparison.
            </p>
          </div>
          <Link href="/uq-methods" className="btn-ghost">
            View comparison <ArrowRight size={14} />
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {uqMethods.map((m) => (
            <Link
              key={m.id}
              href="/uq-methods"
              className="surface p-5 hover:border-navy-200 transition-colors block group"
            >
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: m.color }} />
                <span className="label">{m.shortName}</span>
              </div>
              <h3 className="text-sm font-semibold text-navy-700 mb-1.5 leading-snug">
                {m.name}
              </h3>
              <p className="text-xs text-navy-500 leading-relaxed mb-3">{m.tagline}</p>
              <div className="pt-3 border-t border-navy-50 space-y-1.5 text-[11px] text-navy-500">
                <div>
                  <span className="text-navy-400">Pipeline: </span>
                  <span className="text-navy-600">{m.pipelineAssignment}</span>
                </div>
                <div>
                  <span className="text-navy-400">Cost: </span>
                  <span className="text-navy-600">{m.computationalCost}</span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* ─── Pipeline preview ─── */}
      <section className="surface p-6">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="h-section">Pipeline Stages</h2>
            <p className="body-secondary mt-0.5">
              Uncertainty threads through every stage from perception to decision.
            </p>
          </div>
          <Link href="/architecture" className="btn-ghost">
            Detailed architecture <ArrowRight size={14} />
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {pipelineStages.map((stage) => (
            <div
              key={stage.id}
              className="border-l-2 pl-3 py-1"
              style={{ borderColor: stage.uqColor }}
            >
              <div className="label text-navy-400 mb-1">Stage {stage.index}</div>
              <div className="text-sm font-semibold text-navy-700">{stage.name}</div>
              <div className="text-[11px] text-navy-500 mt-0.5 leading-snug">
                {stage.shortDescription}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── Model + Evaluation Protocol ─── */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu size={16} className="text-gold-500" />
            <h2 className="h-section">Model Backbone</h2>
          </div>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="label mb-1">Architecture</div>
                <div className="text-sm font-semibold text-navy-700">ResNet-50</div>
              </div>
              <div>
                <div className="label mb-1">Pretrained on</div>
                <div className="text-sm font-semibold text-navy-700">ImageNet</div>
              </div>
              <div>
                <div className="label mb-1">Fine-tuned on</div>
                <div className="text-sm font-semibold text-navy-700">Ship dataset</div>
              </div>
              <div>
                <div className="label mb-1">Ship classes</div>
                <div className="text-sm font-semibold text-navy-700">6</div>
              </div>
            </div>
            <p className="text-xs text-navy-500 leading-relaxed pt-2 border-t border-navy-50">
              A standard, proven backbone. The novelty is in what we add on top:
              UQ methods (MC Dropout, Deep Ensembles), calibration (temperature scaling +
              conformal prediction), and cost-minimised deferral with asymmetric costs.
            </p>
            <Link href="/experiments" className="btn-ghost text-xs">
              View experimental results <ArrowRight size={12} />
            </Link>
          </div>
        </div>

        <div className="surface p-6">
          <div className="flex items-center gap-2 mb-4">
            <FlaskConical size={16} className="text-gold-500" />
            <h2 className="h-section">Evaluation Protocol</h2>
          </div>
          <div className="space-y-2.5 text-xs text-navy-600">
            <div className="flex items-start gap-2">
              <SplitSquareHorizontal size={12} className="text-navy-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong className="text-navy-700">Vessel-aware splitting</strong> —
                all frames from the same vessel go to the same split, preventing leakage
              </span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 size={12} className="text-navy-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong className="text-navy-700">Four-way split</strong> — train / val /
                calibration / test. Calibration data never used for training.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <AlertTriangle size={12} className="text-navy-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong className="text-navy-700">Hold-out-class OOD</strong> — whole ship
                classes held out during training to test unseen-vessel detection
              </span>
            </div>
            <div className="flex items-start gap-2">
              <FlaskConical size={12} className="text-navy-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong className="text-navy-700">ImageNet-C corruptions</strong> — 5 types
                × 5 severities (fog, rain, low light, glare, noise)
              </span>
            </div>
            <div className="flex items-start gap-2">
              <Calendar size={12} className="text-navy-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong className="text-navy-700">Multi-seed runs</strong> — 3-5 seeds,
                reported as mean ± standard deviation
              </span>
            </div>
            <div className="flex items-start gap-2">
              <Brain size={12} className="text-navy-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong className="text-navy-700">Cost-minimised deferral</strong> —
                asymmetric cost matrix (enemy→friendly = 10× false alarm = 3×)
              </span>
            </div>
          </div>
          <Link href="/experiments" className="btn-ghost text-xs mt-3">
            View experimental results <ArrowRight size={12} />
          </Link>
        </div>
      </section>

      {/* ─── Deliverables ─── */}
      <section className="surface">
        <div className="px-6 py-4 border-b border-navy-100 flex items-center justify-between">
          <h2 className="h-section">Research Deliverables</h2>
          <Link href="/submission" className="btn-ghost">
            Submission package <ArrowRight size={14} />
          </Link>
        </div>
        <div className="divide-y divide-navy-50">
          {deliverables.map((d) => (
            <div
              key={d.id}
              className="grid grid-cols-12 gap-4 px-6 py-3.5 items-center hover:bg-navy-50/40"
            >
              <div className="col-span-1">
                <span className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-navy-700 text-white text-[10px] font-mono font-bold">
                  {d.id}
                </span>
              </div>
              <div className="col-span-5">
                <div className="text-sm font-semibold text-navy-700">{d.name}</div>
                <div className="text-[11px] text-navy-500 mt-0.5">{d.description}</div>
              </div>
              <div className="col-span-4 text-[11px] text-navy-500 leading-snug">
                {d.acceptance}
              </div>
              <div className="col-span-2 text-right">
                <StatusBadge status={d.status} />
                <div className="text-[10px] text-navy-400 mt-1">{d.month}</div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ─── Components ───

function ResearchAttribute({
  icon,
  value,
  label,
  sublabel,
}: {
  icon: React.ReactNode;
  value: string;
  label: string;
  sublabel: string;
}) {
  return (
    <div className="surface p-5">
      <div className="flex items-start justify-between mb-2">
        <span className="label">{label}</span>
        <span className="text-navy-300">{icon}</span>
      </div>
      <div className="text-3xl font-bold text-navy-700 tabular-nums">{value}</div>
      <div className="text-[11px] text-navy-400 mt-1 leading-snug">{sublabel}</div>
    </div>
  );
}

function ProblemItem({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2 text-sm text-navy-600 leading-relaxed">
      <span className="text-gold-400 mt-1.5 flex-shrink-0">•</span>
      <span>{children}</span>
    </li>
  );
}

function StatusBadge({ status }: { status: "proposed" | "implemented" | "tested" | "verified" }) {
  const styles: Record<string, string> = {
    proposed: "bg-navy-50 text-navy-500 border-navy-200",
    implemented: "bg-gold-50 text-gold-700 border-gold-200",
    tested: "bg-blue-50 text-blue-700 border-blue-200",
    verified: "bg-green-50 text-green-700 border-green-200",
  };
  return (
    <span className={`chip border ${styles[status]}`}>
      {status}
    </span>
  );
}
