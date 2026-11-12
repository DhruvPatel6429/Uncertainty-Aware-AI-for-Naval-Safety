import Link from "next/link";
import {
  FileText,
  CheckCircle2,
  Package,
  GitBranch,
  Award,
  ArrowRight,
  CircleDot,
} from "lucide-react";
import {
  researchPhases,
  deliverables,
  submissionArtifacts,
  adoptionPathway,
  type Deliverable,
} from "@/lib/data";

export default function SubmissionPage() {
  return (
    <div className="page-container">
      {/* Header */}
      <div>
        <h1 className="h-page">DRDO Submission Package</h1>
        <p className="body-secondary mt-1">
          Complete research deliverables, package contents, and the proposed
          DRDO adoption pathway. The artefact set is designed for review by
          NRB and DRDO laboratories.
        </p>
      </div>

      {/* ─── Section A: Research Phases ─── */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <GitBranch size={16} className="text-gold-500" />
          <h2 className="h-section">Research Phases</h2>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {researchPhases.map((p) => (
            <PhaseCard key={p.id} phase={p} />
          ))}
        </div>
      </section>

      {/* ─── Section B: Deliverables ─── */}
      <section className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100 flex items-center gap-2">
          <Package size={16} className="text-gold-500" />
          <h2 className="h-section">Five Deliverables</h2>
        </div>
        <div className="divide-y divide-navy-50">
          {deliverables.map((d) => (
            <DeliverableRow key={d.id} d={d} />
          ))}
        </div>
      </section>

      {/* ─── Section C: Submission Package ─── */}
      <section className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100 flex items-center gap-2">
          <FileText size={16} className="text-gold-500" />
          <h2 className="h-section">Submission Package Contents</h2>
        </div>
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-3">
          {submissionArtifacts.map((a) => (
            <ArtifactRow key={a.name} artifact={a} />
          ))}
        </div>
      </section>

      {/* ─── Section D: Adoption Pathway ─── */}
      <section className="surface">
        <div className="px-6 py-3.5 border-b border-navy-100 flex items-center gap-2">
          <Award size={16} className="text-gold-500" />
          <h2 className="h-section">DRDO Adoption Pathway</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {adoptionPathway.map((stage, i) => (
              <div key={stage.id} className="relative">
                <AdoptionCard stage={stage} index={i + 1} />
              </div>
            ))}
          </div>
          <div className="mt-5 pt-4 border-t border-navy-100 text-[11px] text-navy-500 leading-relaxed">
            <strong className="text-navy-600">Note:</strong> The four-step progression
            is deliberately conservative. The framework is not deployed operationally
            within the project&apos;s 12-month horizon; each adoption step produces
            verifiable artefacts that justify progression. Certification is a multi-year
            engagement with certifying authorities — not implied by this proposal.
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <div className="surface-elevated bg-navy-700 text-white p-8 rounded-lg">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="max-w-2xl">
            <div className="label-gold mb-2">Ready for Review</div>
            <h2 className="text-xl font-bold mb-2 text-white">
              Complete artefact set available for inspection
            </h2>
            <p className="text-navy-200 text-sm leading-relaxed">
              The 32-page NRB/DRDO proposal, the framework source code, the
              evaluation report, and this interactive dashboard together provide
              reviewers everything needed to evaluate the project. All artefacts
              are reproducible from a clean environment.
            </p>
          </div>
          <Link
            href="/"
            className="inline-flex items-center justify-center gap-1.5 px-4 py-2 bg-gold-400 text-navy-700 rounded-md text-sm font-medium hover:bg-gold-300 transition-colors"
          >
            Back to Overview <ArrowRight size={14} />
          </Link>
        </div>
      </div>
    </div>
  );
}

function PhaseCard({ phase }: { phase: typeof researchPhases[number] }) {
  return (
    <div className="surface p-5">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center justify-center w-8 h-8 rounded-md bg-navy-700 text-white text-xs font-bold font-mono">
            {phase.id}
          </span>
          <div>
            <div className="text-sm font-semibold text-navy-700 leading-tight">
              {phase.title}
            </div>
            <div className="text-[11px] text-navy-400 mt-0.5">{phase.months}</div>
          </div>
        </div>
        <span className="chip bg-gold-50 text-gold-700 border border-gold-200">
          {phase.milestone}
        </span>
      </div>

      <div className="space-y-3 text-xs">
        <div>
          <div className="label mb-1">Objectives</div>
          <div className="text-navy-600 leading-relaxed">{phase.objectives}</div>
        </div>
        <div>
          <div className="label mb-1">Expected Work</div>
          <div className="text-navy-600 leading-relaxed">{phase.expectedWork}</div>
        </div>
        <div>
          <div className="label mb-1">Expected Outputs</div>
          <ul className="space-y-0.5">
            {phase.expectedOutputs.map((o) => (
              <li key={o} className="text-navy-600 flex items-start gap-1.5">
                <span className="text-navy-300 mt-0.5">•</span>
                <span>{o}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="pt-2 border-t border-navy-50">
          <div className="label mb-1">Milestone</div>
          <div className="text-navy-600 leading-relaxed">{phase.milestoneLabel}</div>
          <div className="text-[10px] text-navy-400 mt-1">
            <strong className="text-navy-500">Acceptance:</strong> {phase.acceptanceCriteria}
          </div>
        </div>
      </div>
    </div>
  );
}

function DeliverableRow({ d }: { d: Deliverable }) {
  return (
    <div className="grid grid-cols-12 gap-4 px-6 py-3.5 items-center hover:bg-navy-50/40">
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
  );
}

function StatusBadge({ status }: { status: Deliverable["status"] }) {
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

function ArtifactRow({
  artifact,
}: {
  artifact: typeof submissionArtifacts[number];
}) {
  const statusIcon = {
    available: <CheckCircle2 size={14} className="text-green-600" />,
    "in-preparation": <CircleDot size={14} className="text-amber-600" />,
    proposed: <CircleDot size={14} className="text-navy-400" />,
  }[artifact.status];

  return (
    <div className="rounded-md border border-navy-100 p-3.5 bg-white hover:border-navy-200 transition-colors">
      <div className="flex items-start justify-between mb-1.5">
        <div className="text-sm font-semibold text-navy-700">{artifact.name}</div>
        {statusIcon}
      </div>
      <div className="text-[11px] text-navy-500 leading-snug mb-2">
        {artifact.description}
      </div>
      {artifact.location && (
        <div className="flex items-center gap-1.5 text-[10px] text-navy-400">
          <span className="label text-navy-400">Location:</span>
          <code className="font-mono text-navy-600">{artifact.location}</code>
        </div>
      )}
    </div>
  );
}

function AdoptionCard({
  stage,
  index,
}: {
  stage: typeof adoptionPathway[number];
  index: number;
}) {
  return (
    <div className="surface p-5 h-full">
      <div className="flex items-center justify-between mb-3">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-navy-700 text-white text-xs font-bold">
          {index}
        </span>
        <span className="chip bg-navy-50 text-navy-600 border border-navy-200 text-[10px]">
          {stage.timeframe}
        </span>
      </div>
      <h3 className="text-sm font-semibold text-navy-700 mb-1.5">{stage.name}</h3>
      <p className="text-[11px] text-navy-500 leading-relaxed mb-3">
        {stage.description}
      </p>
      <div className="pt-2 border-t border-navy-50">
        <div className="label text-navy-400 mb-1">Technical Role</div>
        <div className="text-[11px] text-navy-600 leading-snug">
          {stage.technicalRole}
        </div>
      </div>
    </div>
  );
}
