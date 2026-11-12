// Shared research data for the UQ-AI Naval Dashboard
// All content mirrors the NRB/DRDO proposal document.
// Scientific provenance is explicit on every metric.

// ───────────────────────────────────────────────────────────
// UQ METHODS — extended with full technical detail
// ───────────────────────────────────────────────────────────

export type DataProvenance = "measured" | "illustrative" | "proposed" | "not-evaluated";

export type UQMethod = {
  id: string;
  name: string;
  shortName: string;
  color: string;
  tagline: string;                    // one-line summary
  description: string;
  // Methodology detail
  formulation: string;                 // mathematical formulation summary
  variants: { name: string; description: string }[];
  strengths: string[];
  weaknesses: string[];
  limitations: string[];               // known research limitations
  computationalCost: string;
  inferenceLatency: string;
  pipelineAssignment: string;
  keyPapers: { author: string; year: number; title: string }[];
  // Comparative metrics (1-5 scale, higher = stronger)
  metrics: {
    coverage: number;       // Coverage guarantee strength
    calibration: number;    // Calibration quality
    compute: number;        // Compute efficiency (higher = cheaper)
    ood: number;            // OOD robustness
    scalability: number;    // Scalability to large models
    rigor: number;           // Theoretical rigor
  };
  metricsProvenance: DataProvenance;
};

export const uqMethods: UQMethod[] = [
  {
    id: "bayesian",
    name: "Bayesian Deep Learning",
    shortName: "Bayesian DL",
    color: "#0F2845",
    tagline: "Epistemic uncertainty from posterior over weights",
    description:
      "Treats neural-network weights as random variables and infers their posterior given the data. The predictive distribution marginalises over the posterior, allowing aleatoric and epistemic uncertainty to be decomposed.",
    formulation:
      "p(y*|x*, D) = ∫ p(y*|x*, w) p(w|D) dw  —  marginalised over posterior",
    variants: [
      { name: "MC Dropout", description: "Approximate VI via dropout at inference (Gal & Ghahramani, 2016). T stochastic forward passes; predictive variance decomposes into aleatoric + epistemic." },
      { name: "Bayes-by-Backprop", description: "Variational posterior with reparameterisation trick (Blundell et al., 2015). Doubles parameter count; ELBO objective." },
      { name: "Laplace Approximation", description: "Post-hoc Gaussian fit to loss landscape at MLE solution (Kristiadi et al., 2020). Scales to large pre-trained backbones." },
    ],
    strengths: [
      "Theoretically grounded in Bayesian framework",
      "Clean decomposition of aleatoric vs epistemic uncertainty",
      "Laplace variant scales to large pre-trained backbones",
      "Posterior predictive captures parameter uncertainty",
    ],
    weaknesses: [
      "MC Dropout requires T-fold inference cost",
      "Full VI scales poorly above 50M parameters",
      "Sensitive to prior choice and approximation family",
      "Laplace assumes locally Gaussian posterior — may miss multimodality",
    ],
    limitations: [
      "Approximate posteriors can be poorly calibrated under distribution shift",
      "No finite-sample coverage guarantee (unlike conformal)",
      "MC variance estimation requires sufficient T for stable estimates",
    ],
    computationalCost: "2-5× deterministic training; T× inference",
    inferenceLatency: "High (MC sampling, T=10 typical)",
    pipelineAssignment: "Stage 1 — Perception (object detection, segmentation)",
    keyPapers: [
      { author: "Gal & Ghahramani", year: 2016, title: "Dropout as a Bayesian Approximation" },
      { author: "Blundell et al.", year: 2015, title: "Weight Uncertainty in Neural Networks" },
      { author: "Kendall & Gal", year: 2017, title: "What Uncertainties Do We Need in Bayesian DL" },
      { author: "Kristiadi et al.", year: 2020, title: "Being Bayesian, Even Just a Bit" },
    ],
    metrics: { coverage: 3, calibration: 4, compute: 2, ood: 4, scalability: 2, rigor: 5 },
    metricsProvenance: "illustrative",
  },
  {
    id: "conformal",
    name: "Conformal Prediction",
    shortName: "Conformal",
    color: "#C9963A",
    tagline: "Distribution-free coverage guarantees under exchangeability",
    description:
      "Distribution-free framework producing prediction sets with finite-sample (1-α) coverage guarantees. Conformal Risk Control generalises this to bound arbitrary monotone risk functionals.",
    formulation:
      "P(Y ∈ Ĉ(x)) ≥ 1 - α  —  finite-sample guarantee under exchangeability",
    variants: [
      { name: "Inductive (Split) Conformal", description: "Post-hoc wrapper around any trained model. Single forward pass at inference. Requires only a held-out calibration set." },
      { name: "Mondrian Conformal", description: "Class-conditional coverage guarantees. Per-class quantiles; useful when misclassification costs differ by class." },
      { name: "Conformal Risk Control", description: "Bounds any monotone risk functional (e.g., expected collision cost), not just miscoverage (Angelopoulos et al., 2024)." },
    ],
    strengths: [
      "Distribution-free (1-α) coverage guarantee under exchangeability",
      "Post-hoc — applies to any trained model without retraining",
      "Negligible inference overhead (single forward pass)",
      "Conformal Risk Control bounds arbitrary risk functionals",
    ],
    weaknesses: [
      "Exchangeability assumption violated in time-series data",
      "Coverage degrades under distribution shift",
      "Mondrian variants require class-conditional calibration sets",
      "Prediction set size can be large under high uncertainty",
    ],
    limitations: [
      "Exchangeability is approximately satisfied for maritime scenes drawn from similar conditions; violated under temporal drift",
      "Weighted conformal (Tibshirani et al., 2019) partially addresses covariate shift but not full distribution shift",
      "Coverage guarantee is marginal, not conditional on individual inputs",
    ],
    computationalCost: "1× training (post-hoc); 1× inference",
    inferenceLatency: "Low (single forward pass + threshold check)",
    pipelineAssignment: "Stage 4 — Decision (risk control, reject option)",
    keyPapers: [
      { author: "Vovk, Gammerman, Shafer", year: 2005, title: "Algorithmic Learning in a Random World" },
      { author: "Angelopoulos & Bates", year: 2021, title: "A Gentle Introduction to Conformal Prediction" },
      { author: "Tibshirani et al.", year: 2019, title: "Conformal Prediction Under Covariate Shift" },
      { author: "Angelopoulos et al.", year: 2024, title: "Conformal Risk Control" },
    ],
    metrics: { coverage: 5, calibration: 5, compute: 5, ood: 3, scalability: 5, rigor: 4 },
    metricsProvenance: "illustrative",
  },
  {
    id: "ensembles",
    name: "Deep Ensembles",
    shortName: "Ensembles",
    color: "#3F8A5E",
    tagline: "Empirically strongest calibration via independent members",
    description:
      "Trains M independent copies of the same architecture with different random initialisations. Predictive uncertainty decomposes into mean aleatoric variance and epistemic variance of means.",
    formulation:
      "p(y|x) ≈ (1/M) Σ_m p(y|x, θ_m);  epistemic = Var_m[μ_m],  aleatoric = E_m[σ_m²]",
    variants: [
      { name: "Standard Deep Ensemble", description: "M independently trained models (Lakshminarayanan et al., 2017). M-fold training and inference cost." },
      { name: "Snapshot Ensemble", description: "Collects M snapshots from one cyclic-learning-rate training run (Huang et al., 2017). Halves training cost." },
      { name: "BatchEnsemble", description: "Shares most parameters; learns rank-one perturbations per member (Wen et al., 2020). 10× cheaper." },
    ],
    strengths: [
      "Best-in-class empirical calibration on standard benchmarks",
      "Robust to OOD inputs — members disagree on unseen patterns",
      "Faithfully captures multi-modal predictive distributions",
      "Simple to implement — no specialised training loop",
    ],
    weaknesses: [
      "M-fold training and inference cost",
      "Linear memory footprint growth",
      "No finite-sample theoretical coverage guarantee",
      "Diversity between members is not guaranteed",
    ],
    limitations: [
      "Empirical calibration may degrade under distribution shift unaccounted for in training",
      "Snapshot and BatchEnsemble trade calibration quality for compute savings",
      "No formal mechanism to detect OOD vs aleatoric noise",
    ],
    computationalCost: "M× training; M× inference (parallelisable)",
    inferenceLatency: "High (M forward passes; parallelisable on GPU)",
    pipelineAssignment: "Stage 2 — Trajectory Prediction (multi-modal forecasting)",
    keyPapers: [
      { author: "Lakshminarayanan et al.", year: 2017, title: "Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles" },
      { author: "Huang et al.", year: 2017, title: "Snapshot Ensembles: Train 1, Get M for Free" },
      { author: "Wen, Tran, Ba", year: 2020, title: "BatchEnsemble: An Alternative Approach to Efficient Ensemble" },
    ],
    metrics: { coverage: 3, calibration: 5, compute: 1, ood: 5, scalability: 2, rigor: 3 },
    metricsProvenance: "illustrative",
  },
  {
    id: "evidential",
    name: "Evidential Deep Learning",
    shortName: "Evidential",
    color: "#E0A53D",
    tagline: "Single-forward-pass uncertainty via higher-order priors",
    description:
      "Places a higher-order prior over the predictive distribution and learns its parameters in a single forward pass. Dirichlet for classification; Normal-Inverse-Gamma for regression.",
    formulation:
      "Classification: Dirichlet(α), α = e(x) + 1;  Regression: NIG(γ, ν, α, β)",
    variants: [
      { name: "Evidential Classification", description: "Network outputs evidence vector; Dirichlet concentration encodes belief mass (Sensoy et al., 2018)." },
      { name: "Evidential Regression", description: "Network outputs NIG parameters; aleatoric = β/(α-1), epistemic = β/((α-1)ν) (Amini et al., 2020)." },
    ],
    strengths: [
      "Single forward pass — no MC sampling or ensemble",
      "Decomposes aleatoric vs epistemic in one pass",
      "Suitable for high-frequency inference (>50 Hz)",
      "Theoretically grounded in subjective logic / Dempster-Shafer theory",
    ],
    weaknesses: [
      "Calibration under distribution shift is an open research question",
      "Dirichlet can over-concentrate on OOD inputs",
      "Theory is still developing — fewer mature implementations",
      "Sensitive to evidence regularisation hyperparameters",
    ],
    limitations: [
      "Empirical calibration on OOD data is mixed — Pawlowski et al. (2023) propose regularisation but problem is not fully solved",
      "Dirichlet assumption may be inappropriate for non-categorical outputs",
      "Comparison with Bayesian methods on standard benchmarks is limited in the literature",
    ],
    computationalCost: "1× training; 1× inference",
    inferenceLatency: "Low (single forward pass)",
    pipelineAssignment: "Stage 2 — Latency-critical real-time inference",
    keyPapers: [
      { author: "Sensoy, Kaplan, Kandemir", year: 2018, title: "Evidential Deep Learning to Quantify Classification Uncertainty" },
      { author: "Amini et al.", year: 2020, title: "Deep Evidential Regression" },
      { author: "Pawlowski et al.", year: 2023, title: "Deep Evidential Regression with Kernel Loss" },
    ],
    metrics: { coverage: 2, calibration: 3, compute: 5, ood: 3, scalability: 4, rigor: 4 },
    metricsProvenance: "illustrative",
  },
];

// ───────────────────────────────────────────────────────────
// PIPELINE STAGES — with full technical detail
// ───────────────────────────────────────────────────────────

export type PipelineStage = {
  id: string;
  index: number;
  name: string;
  shortDescription: string;
  detailedDescription: string;
  primaryUQMethod: string;
  uqColor: string;
  algorithmicApproach: string;
  inputs: string[];
  outputs: string[];
  dependencies: string[];
  knownLimitations: string[];
};

export const pipelineStages: PipelineStage[] = [
  {
    id: "perception",
    index: 1,
    name: "Perception",
    shortDescription: "Bayesian object detection + conformal calibration",
    detailedDescription:
      "Processes multi-modal sensor data (radar, AIS, EO/IR, simulated sonar) into calibrated obstacle tracks. A Bayesian detector produces per-detection epistemic uncertainty; a post-hoc conformal calibration layer wraps the detector to provide distribution-free coverage on the detection set.",
    primaryUQMethod: "Bayesian Deep Learning",
    uqColor: "#0F2845",
    algorithmicApproach:
      "MC Dropout (T=10) + Laplace approximation on a YOLO/U-Net backbone, followed by inductive conformal calibration of detection scores.",
    inputs: ["Radar", "AIS", "EO/IR", "Sonar (simulated)"],
    outputs: ["Obstacle tracks", "Calibrated confidence scores", "Per-detection epistemic uncertainty"],
    dependencies: ["Pre-trained detector backbone", "Calibration set of (image, label) pairs"],
    knownLimitations: [
      "Calibration quality depends on calibration set representing deployment distribution",
      "MC Dropout adds T-fold inference cost",
      "OOD detection relies on epistemic threshold — no formal OOD guarantee",
    ],
  },
  {
    id: "prediction",
    index: 2,
    name: "Trajectory Prediction",
    shortDescription: "Multi-modal forecasting with directional uncertainty cones",
    detailedDescription:
      "Forecasts each obstacle's future course over a 30-second horizon. Trajectories are inherently multi-modal (vessel may turn left, right, or hold); the predictor must faithfully reflect this rather than collapse to a mean trajectory.",
    primaryUQMethod: "Deep Ensembles + Evidential DL",
    uqColor: "#3F8A5E",
    algorithmicApproach:
      "Deep Ensemble of LSTMs (M=5) for multi-modal capture; Evidential Transformer for single-pass low-latency inference. Both emit directional uncertainty cones.",
    inputs: ["Obstacle tracks (from Stage 1)", "Historical AIS trajectories"],
    outputs: ["Multi-modal trajectory forecasts", "Directional uncertainty cones", "Per-timestep epistemic + aleatoric"],
    dependencies: ["AIS trajectory dataset", "Trained LSTM/Transformer weights"],
    knownLimitations: [
      "Ensemble M=5 may be insufficient for highly multi-modal encounters",
      "Evidential calibration under shift is open — see UQ Methods limitations",
      "Forecast horizon (30s) may be insufficient for high-speed encounters",
    ],
  },
  {
    id: "planning",
    index: 3,
    name: "Path Planning",
    shortDescription: "Uncertainty-aware A*/RRT* with COLREGs soft constraints",
    detailedDescription:
      "Computes a manoeuvre that minimises expected cost while respecting COLREGs obligations and uncertainty-aware safety margins. The cost function combines path length, COLREGs soft-constraint penalty, uncertainty inflation, and energy cost.",
    primaryUQMethod: "Hybrid (uncertainty-aware cost function)",
    uqColor: "#C9963A",
    algorithmicApproach:
      "A* for short-horizon reactive planning; RRT* for long-horizon (>60s) planning. COLREGs Rules 13-15 encoded as soft-constraint penalty terms.",
    inputs: ["Predicted trajectories (Stage 2)", "Directional uncertainty cones", "Obstacle information"],
    outputs: ["Candidate waypoints", "Planned path with confidence margin", "Cost decomposition"],
    dependencies: ["Stage 2 forecasts", "Own-ship state", "Operational parameters (λ weights)"],
    knownLimitations: [
      "COLREGs encoded as soft constraints — not formal legal compliance",
      "Cost weights (λ_1, λ_2, λ_3) require per-scenario tuning",
      "RRT* completeness depends on sampling budget",
    ],
  },
  {
    id: "decision",
    index: 4,
    name: "Decision & Risk Control",
    shortDescription: "Conformal Risk Control + reject option + audit trail",
    detailedDescription:
      "The safety-critical capstone. Bounds expected collision probability below a user-specified α via Conformal Risk Control. Triggers a reject option (escalation to human supervisor or emergency fallback) when epistemic uncertainty exceeds the operational threshold.",
    primaryUQMethod: "Conformal Risk Control",
    uqColor: "#E0A53D",
    algorithmicApproach:
      "Conformal Risk Control (Angelopoulos et al., 2024) selects policy parameter θ such that empirical risk on calibration set is ≤ α. Reject option aggregates pipeline epistemic uncertainty.",
    inputs: ["Planned path (Stage 3)", "Aggregated epistemic uncertainty", "Calibration set of historical encounters"],
    outputs: ["Manoeuvre recommendation", "Collision probability bound", "Audit record", "Reject / escalation trigger"],
    dependencies: ["Stage 3 planned path", "Historical encounter calibration set", "Operator-configured α"],
    knownLimitations: [
      "Conformal bound is conditional on exchangeability — violated under distribution shift",
      "α = 10⁻⁴ bound is statistical, not physical — does not guarantee zero collisions",
      "Reject threshold is operator-configured; requires operational judgement",
    ],
  },
];

// ───────────────────────────────────────────────────────────
// RESEARCH PHASES — structured records (NOT a timeline)
// ───────────────────────────────────────────────────────────

export type ResearchPhase = {
  id: string;
  title: string;
  months: string;
  objectives: string;
  expectedWork: string;
  milestone: string;
  milestoneLabel: string;
  expectedOutputs: string[];
  acceptanceCriteria: string;
};

export const researchPhases: ResearchPhase[] = [
  {
    id: "P1",
    title: "Foundations & Literature Review",
    months: "Months 1–2",
    objectives: "Establish software stack, complete literature review, define evaluation protocol.",
    expectedWork: "Software environment setup, Hydra/MLflow/Docker configuration, literature survey of four UQ method families, evaluation protocol specification.",
    milestone: "MS1",
    milestoneLabel: "Literature review + evaluation protocol approved",
    expectedOutputs: [
      "Software stack v0.1",
      "Literature review document",
      "Evaluation protocol specification",
    ],
    acceptanceCriteria: "Covers all four UQ methods; protocol defines 8 metrics on 3 datasets + 1 simulator.",
  },
  {
    id: "P2",
    title: "Perception UQ Module",
    months: "Months 3–4",
    objectives: "Implement and benchmark the four UQ methods on a perception task using MAS500 maritime dataset.",
    expectedWork: "Implement Bayesian DL, Conformal, Ensembles, Evidential classifiers; calibrate on MAS500; produce calibration benchmarks.",
    milestone: "MS2",
    milestoneLabel: "Perception prototype with all four UQ methods integrated",
    expectedOutputs: [
      "UQ method implementations (4)",
      "Perception module v0.5",
      "Calibration benchmarks on MAS500",
    ],
    acceptanceCriteria: "All four UQ methods integrated; ECE within 3% of published baselines on MAS500.",
  },
  {
    id: "P3",
    title: "Prediction & Planning Modules",
    months: "Months 5–7",
    objectives: "Extend perception module with trajectory prediction and uncertainty-aware path planning.",
    expectedWork: "Implement LSTM/Transformer trajectory forecasters with ensemble/evidential uncertainty; implement A*/RRT* planner with UQ-aware cost function; integrate perception–prediction–planning pipeline.",
    milestone: "MS3",
    milestoneLabel: "Integrated navigation pipeline running end-to-end",
    expectedOutputs: [
      "Prediction module v1.0",
      "Planning module v1.0",
      "Integrated perception–prediction–planning pipeline",
    ],
    acceptanceCriteria: "Pipeline runs end-to-end at ≥ 3 Hz on a single GPU; trajectory Brier score < 0.15 on AIS test set.",
  },
  {
    id: "P4",
    title: "Decision Module & Integration",
    months: "Months 8–10",
    objectives: "Implement Conformal Risk Control decision module with reject option; integrate full pipeline; run first end-to-end evaluation.",
    expectedWork: "Implement CRC decision module with audit trail; integrate reject option and HITL escalation; run first end-to-end evaluation on stress-test scenarios.",
    milestone: "MS4",
    milestoneLabel: "Decision module delivering conformal risk bound",
    expectedOutputs: [
      "Decision module v1.0",
      "Full pipeline integration",
      "First evaluation report",
    ],
    acceptanceCriteria: "Conformal coverage within ±1% of nominal; COLREGs violation rate < 5% on stress test.",
  },
  {
    id: "P5",
    title: "Evaluation, Documentation & DRDO Package",
    months: "Months 11–12",
    objectives: "Finalise evaluation harness, complete dashboard, write documentation, assemble DRDO submission package.",
    expectedWork: "Complete evaluation harness; finalise React dashboard; write API reference and architecture documentation; assemble DRDO submission package; verify reproducibility on clean environment.",
    milestone: "MS5",
    milestoneLabel: "DRDO submission package complete",
    expectedOutputs: [
      "Evaluation harness v1.0",
      "Dashboard v1.0",
      "Complete documentation",
      "DRDO submission package",
    ],
    acceptanceCriteria: "Source + docs + eval report + briefing; reproducibility verified on clean environment.",
  },
];

// ───────────────────────────────────────────────────────────
// DELIVERABLES
// ───────────────────────────────────────────────────────────

export type Deliverable = {
  id: string;
  name: string;
  description: string;
  month: string;
  acceptance: string;
  status: "proposed" | "implemented" | "tested" | "verified";
};

export const deliverables: Deliverable[] = [
  {
    id: "D1",
    name: "UQ-AI Software Stack v1.0",
    description: "Modular Python stack with PyTorch, Hydra, MLflow, Docker, FastAPI",
    month: "M4",
    acceptance: "All four UQ methods pass unit tests; YAML-driven config; Docker image < 4 GB",
    status: "verified",
  },
  {
    id: "D2",
    name: "Perception & Prediction Modules",
    description: "Bayesian perception + ensemble/evidential trajectory forecasting",
    month: "M7",
    acceptance: "Calibrated detection scores on MAS500; Brier score < 0.15 on AIS test set",
    status: "implemented",
  },
  {
    id: "D3",
    name: "Path Planning & Decision Modules",
    description: "UQ-aware planner + conformal risk control decision module",
    month: "M10",
    acceptance: "COLREGs violation rate < 5% on 50-scenario stress test; coverage within ±1% of nominal",
    status: "implemented",
  },
  {
    id: "D4",
    name: "Evaluation Harness & Dashboard",
    description: "One-command evaluation + React visualization dashboard",
    month: "M11",
    acceptance: "One-command evaluation reproduces all metrics; dashboard < 200 ms latency",
    status: "verified",
  },
  {
    id: "D5",
    name: "DRDO Submission Package",
    description: "Source, docs, evaluation report, executive briefing",
    month: "M12",
    acceptance: "Reproducibility verified on clean environment; complete documentation",
    status: "verified",
  },
];

// ───────────────────────────────────────────────────────────
// RISKS — preserved from proposal
// ───────────────────────────────────────────────────────────

export type Risk = {
  id: string;
  label: string;
  likelihood: number;  // 1-5
  impact: number;      // 1-5
  category: "technical" | "operational";
  mitigation: string;
  residualRisk?: string;
};

export const risks: Risk[] = [
  {
    id: "R1",
    label: "Bayesian DL scalability to large networks",
    likelihood: 4, impact: 5,
    category: "technical",
    mitigation: "Use MC Dropout + Laplace approximation rather than full VI; for backbones above 50M parameters, fall back to frozen-backbone Bayesian head.",
    residualRisk: "Laplace posterior may miss multimodality in loss landscape.",
  },
  {
    id: "R2",
    label: "Data scarcity for naval scenes",
    likelihood: 3, impact: 4,
    category: "technical",
    mitigation: "Transfer learning from large general datasets (ImageNet, COCO); augment with simulated maritime data; explicitly report OOD performance on held-out vessel types.",
    residualRisk: "Sim-to-real gap may persist despite augmentation.",
  },
  {
    id: "R3",
    label: "Conformal exchangeability violation in time-series",
    likelihood: 4, impact: 4,
    category: "technical",
    mitigation: "Use weighted conformal prediction (Tibshirani et al., 2019) and sliding-window recalibration; explicitly measure coverage under temporal drift.",
    residualRisk: "Coverage under aggressive drift may still fall below nominal 1-α.",
  },
  {
    id: "R4",
    label: "Deep ensemble compute cost",
    likelihood: 2, impact: 4,
    category: "technical",
    mitigation: "Snapshot ensembles and batch-ensembled networks reduce cost by 50–90%; framework defaults to M=5 members with snapshot fallback when compute is constrained.",
    residualRisk: "Snapshot diversity may be lower than independent ensemble.",
  },
  {
    id: "R5",
    label: "Evidential DL miscalibration under shift",
    likelihood: 3, impact: 3,
    category: "technical",
    mitigation: "Include evidential-specific regularisation (Pawlowski et al., 2023); explicitly test calibration under shift and report AUROC for OOD detection.",
    residualRisk: "Open research question — no complete solution exists in literature.",
  },
  {
    id: "R6",
    label: "Single-PI bandwidth",
    likelihood: 2, impact: 3,
    category: "operational",
    mitigation: "Phased roadmap designed with slack between phases; PI commits 30 hr/wk; plug-in architecture allows individual modules to be deferred without compromising framework's core contribution.",
    residualRisk: "Illness or competing commitments may delay milestones.",
  },
  {
    id: "R7",
    label: "Reproducibility across environments",
    likelihood: 2, impact: 2,
    category: "operational",
    mitigation: "Docker containers pin all dependencies; deterministic seeds enforced; evaluation harness reports confidence intervals across multiple seeds.",
    residualRisk: "GPU/CUDA version differences may produce small numerical variation.",
  },
  {
    id: "R8",
    label: "Third-party API drift",
    likelihood: 1, impact: 3,
    category: "operational",
    mitigation: "Dependencies pinned to specific versions; framework's plug-in architecture isolates third-party dependencies behind internal interfaces, limiting blast radius.",
    residualRisk: "Security patches may force breaking upgrades.",
  },
];

// ───────────────────────────────────────────────────────────
// NAVIGATION — Roadmap removed
// ───────────────────────────────────────────────────────────

export const navItems = [
  { href: "/",             label: "Overview",        icon: "LayoutDashboard" },
  { href: "/architecture", label: "Architecture",    icon: "Network" },
  { href: "/uq-methods",   label: "UQ Methods",       icon: "BrainCircuit" },
  { href: "/pipeline",     label: "Pipeline Demo",   icon: "Workflow" },
  { href: "/live-demo",    label: "Live Demo",        icon: "Radio" },
  { href: "/experiments",  label: "Experiments",       icon: "FlaskConical" },
  { href: "/risks",        label: "Risk Matrix",      icon: "AlertTriangle" },
  { href: "/submission",   label: "DRDO Submission", icon: "FileCheck" },
];

// ───────────────────────────────────────────────────────────
// SUBMISSION PACKAGE — DRDO adoption pathway
// ───────────────────────────────────────────────────────────

export type SubmissionArtifact = {
  name: string;
  description: string;
  status: "available" | "in-preparation" | "proposed";
  location?: string;
};

export const submissionArtifacts: SubmissionArtifact[] = [
  { name: "Source Code Repository", description: "Complete framework source, Apache 2.0 licensed", status: "available", location: "src/uq_naval/" },
  { name: "Technical Documentation", description: "Installation, architecture, evaluation, API reference", status: "available", location: "docs/" },
  { name: "Evaluation Report", description: "All metrics with confidence intervals and calibration plots", status: "available", location: "reports/evaluation.json" },
  { name: "Research Proposal", description: "32-page NRB/DRDO formal proposal document", status: "available", location: "UQ_Naval_Proposal.pdf" },
  { name: "Visualization Dashboard", description: "Next.js interactive dashboard (this application)", status: "available", location: "dashboard/" },
  { name: "Reproducibility Bundle", description: "Docker images, pinned dependencies, deterministic seeds", status: "available", location: "docker-compose.yml" },
  { name: "Executive Briefing", description: "Quick-reference briefing for DRDO leadership", status: "available", location: "docs/drdo_briefing.md" },
  { name: "Test Suite Report", description: "PyTest suite — 81 tests, all passing", status: "available", location: "tests/" },
];

export type AdoptionStage = {
  id: string;
  name: string;
  timeframe: string;
  description: string;
  technicalRole: string;
};

export const adoptionPathway: AdoptionStage[] = [
  {
    id: "evaluation",
    name: "Evaluation",
    timeframe: "M12–M15",
    description: "DRDO laboratories install the framework, run the evaluation suite on internal datasets, and validate reported calibration properties on their own data.",
    technicalRole: "Independent verification of calibration claims and reproducibility.",
  },
  {
    id: "extension",
    name: "Extension",
    timeframe: "M15–M24",
    description: "DRDO laboratories develop domain-specific modules (sonar perception at NPOL, AUV control at NSTL) that plug into the framework's interfaces.",
    technicalRole: "Domain-specific UQ methods and models, integrated via the plug-in architecture.",
  },
  {
    id: "integration",
    name: "Integration",
    timeframe: "M24–M36",
    description: "DRDO integrates the extended framework into platform-specific deployments, including hardware-in-the-loop testing and operator-interface customisation.",
    technicalRole: "HIL testing, platform-specific tuning, operator training package.",
  },
  {
    id: "certification",
    name: "Certification Pathway",
    timeframe: "M36+",
    description: "Audit-trail artefacts serve as evidence in a future safety-assurance case, contributing to a STANAG-4671-style certification effort.",
    technicalRole: "Safety-assurance evidence package — not certification itself.",
  },
];

// ───────────────────────────────────────────────────────────
// MARITIME SCENE DEMO DATA — synthetic, clearly labelled
// ───────────────────────────────────────────────────────────

export type DetectionType = "accepted" | "high-uncertainty" | "ood";

export type MaritimeDetection = {
  id: string;
  label: string;
  classification: string;          // vessel type
  x: number; y: number;            // position in scene (normalized 0-1)
  heading: number;                 // degrees, 0 = north
  speed: number;                   // knots
  confidence: number | null;      // calibrated confidence [0,1] or null if unavailable
  epistemicUnc: number | null;     // [0,1] or null
  aleatoricUnc: number | null;     // [0,1] or null
  detectionType: DetectionType;
  oodIndicator: boolean;
  sourceModel: string;
  // Trajectory forecast
  forecast?: {
    horizonSeconds: number;
    coursePredictions: number[];    // degrees per timestep
    speedPredictions: number[];     // knots per timestep
    uncertaintyConeDeg: number;     // 95% credible half-width
    provenance: DataProvenance;
  };
};

export const maritimeScenarioName = "Synthetic Coastal Encounter";
export const scenarioProvenance: DataProvenance = "illustrative";

export const ownVesselState = {
  label: "Own Vessel",
  x: 0.5, y: 0.75,
  heading: 0,        // north
  speed: 12,         // knots
};

export const maritimeDetections: MaritimeDetection[] = [
  {
    id: "T1",
    label: "Cargo Vessel",
    classification: "Cargo Vessel",
    x: 0.22, y: 0.32,
    heading: 85, speed: 14,
    confidence: 0.94,
    epistemicUnc: 0.08,
    aleatoricUnc: 0.12,
    detectionType: "accepted",
    oodIndicator: false,
    sourceModel: "Bayesian YOLO + Conformal (T=10)",
    forecast: {
      horizonSeconds: 30,
      coursePredictions: [85, 86, 86, 87, 88],
      speedPredictions: [14, 14, 14.1, 14.2, 14.2],
      uncertaintyConeDeg: 4.2,
      provenance: "illustrative",
    },
  },
  {
    id: "T2",
    label: "Fishing Boat",
    classification: "Fishing Vessel",
    x: 0.55, y: 0.22,
    heading: 180, speed: 8,
    confidence: 0.78,
    epistemicUnc: 0.22,
    aleatoricUnc: 0.28,
    detectionType: "accepted",
    oodIndicator: false,
    sourceModel: "Bayesian YOLO + Conformal (T=10)",
    forecast: {
      horizonSeconds: 30,
      coursePredictions: [180, 178, 175, 172, 170],
      speedPredictions: [8, 8.1, 8.0, 7.9, 7.8],
      uncertaintyConeDeg: 12.5,
      provenance: "illustrative",
    },
  },
  {
    id: "T3",
    label: "Buoy",
    classification: "Navigation Aid",
    x: 0.72, y: 0.58,
    heading: 0, speed: 0,
    confidence: 0.91,
    epistemicUnc: 0.05,
    aleatoricUnc: 0.08,
    detectionType: "accepted",
    oodIndicator: false,
    sourceModel: "Bayesian YOLO + Conformal (T=10)",
  },
  {
    id: "T4",
    label: "Unknown Contact",
    classification: "Unidentified",
    x: 0.38, y: 0.45,
    heading: 250, speed: 22,
    confidence: null,
    epistemicUnc: 0.82,
    aleatoricUnc: null,
    detectionType: "ood",
    oodIndicator: true,
    sourceModel: "Bayesian YOLO + Conformal (T=10)",
  },
  {
    id: "T5",
    label: "Speedboat",
    classification: "Small Craft",
    x: 0.78, y: 0.38,
    heading: 45, speed: 28,
    confidence: 0.83,
    epistemicUnc: 0.38,
    aleatoricUnc: 0.21,
    detectionType: "high-uncertainty",
    oodIndicator: false,
    sourceModel: "Bayesian YOLO + Conformal (T=10)",
    forecast: {
      horizonSeconds: 30,
      coursePredictions: [45, 50, 55, 58, 60],
      speedPredictions: [28, 27.5, 27, 26.5, 26],
      uncertaintyConeDeg: 22.0,
      provenance: "illustrative",
    },
  },
];
