# UQ-AI Naval — Uncertainty-Aware AI for Safety-Critical Naval Applications

A software-based, open-source framework that threads calibrated uncertainty
quantification (UQ) through every stage of autonomous naval navigation —
perception, trajectory prediction, path planning, and decision-making.

Submitted as a self-funded research proposal to the **Naval Research Board
(NRB), DRDO**.

---

## What's in this repository

This is a **complete, runnable implementation** of the framework described in
the accompanying NRB/DRDO proposal. It contains:

| Layer | Location | Purpose |
|---|---|---|
| **UQ methods** | `src/uq_naval/uq/` | Bayesian DL, Conformal Prediction, Deep Ensembles, Evidential DL — all behind a unified plug-in interface |
| **Pipeline** | `src/uq_naval/pipeline/` | Perception → Prediction → Planning → Decision, with UQ threaded through every stage |
| **Models** | `src/uq_naval/models/` | Bayesian YOLO/U-Net, LSTM/Transformer forecasters |
| **Data** | `src/uq_naval/data/` | Loaders for AIS traces, MAS500, and synthetic maritime scenarios |
| **Evaluation** | `src/uq_naval/evaluation/` | Calibration metrics (ECE, Brier, coverage) + scenario library |
| **API** | `src/uq_naval/api/` | FastAPI service layer (REST + WebSocket) |
| **Configs** | `configs/` | Hydra YAML configs for every pipeline + UQ combination |
| **Scripts** | `scripts/` | CLI entry points: train, evaluate, serve |
| **Tests** | `tests/` | PyTest suite covering all UQ methods + pipeline |
| **Dashboard** | `dashboard/` | Next.js + Tailwind visualization dashboard |
| **Docs** | `docs/` | Installation, architecture, evaluation, DRDO briefing |
| **Docker** | `docker/` | Containerised API + dashboard |

## Quick Start

### Verify in 5 minutes

```bash
# 1. Install
pip install -r requirements.txt
pip install -e .

# 2. Run the test suite (81 tests)
pytest tests/ -v

# 3. Run the guided demo
python demo.py

# 4. Run the evaluation harness
python scripts/evaluate.py --output reports/evaluation.json
python scripts/render_report.py --input reports/evaluation.json --output reports/evaluation.md

# 5. Start the API + dashboard
python scripts/serve.py &          # API on :8000
cd dashboard && npm install && npm run dev   # Dashboard on :3000
```

See **`QUICKSTART.md`** for the full one-page verification guide.

### Or run everything with Docker

```bash
docker-compose up --build
# API at http://localhost:8000
# Dashboard at http://localhost:3000
```

## Project Structure

```
uq-naval/
├── src/uq_naval/
│   ├── uq/                          # 4 UQ method packages (11 methods total)
│   │   ├── base.py                  # Unified UQInterface
│   │   ├── bayesian/                # MC Dropout, Bayes-by-Backprop, Laplace
│   │   ├── conformal/               # Inductive, Mondrian, Risk Control
│   │   ├── ensembles/               # Deep, Snapshot, Batch-ensembled
│   │   └── evidential/              # Dirichlet classification, NIG regression
│   ├── pipeline/                    # Perception, Prediction, Planning, Decision
│   ├── models/                      # Bayesian YOLO, U-Net, LSTM, Transformer
│   ├── data/                        # AIS, MAS500, synthetic data loaders
│   ├── evaluation/                  # Metrics + scenario library + harness
│   ├── api/                         # FastAPI service
│   ├── tracking/                    # MLflow helpers
│   └── utils/                       # Config, logging, audit, seeds
├── configs/                         # Hydra YAML configs
├── scripts/                         # CLI entry points (train, evaluate, serve, render_report)
├── tests/                           # PyTest suite (81 tests, all passing)
├── demo.py                          # Guided demo for DRDO reviewers
├── dashboard/                       # Next.js visualization dashboard (8 pages)
├── docs/                            # Documentation (installation, architecture, evaluation, API, DRDO briefing)
├── docker/                          # Dockerfiles
├── reports/                         # Evaluation outputs (generated at runtime)
├── QUICKSTART.md                    # One-page verification guide
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package metadata
├── Makefile                         # Convenience targets
├── docker-compose.yml               # API + dashboard orchestration
└── LICENSE                          # Apache 2.0
```

## The Four UQ Methods (Unified Interface)

Every UQ method implements the same interface, allowing them to be swapped
at runtime via a YAML config change:

```python
from uq_naval.uq import load_uq_method

uq = load_uq_method("bayesian", model=my_model, config={...})
uq.calibrate(calibration_loader)
output = uq.predict(test_input)
# output.prediction       -> point prediction
# output.aleatoric_unc    -> irreducible uncertainty
# output.epistemic_unc    -> model uncertainty
# output.prediction_set   -> conformal prediction set (if applicable)
# output.reject            -> True if uncertainty exceeds threshold
```

## Pipeline Stages

| Stage | UQ Method (default) | Output |
|---|---|---|
| Perception | Bayesian DL (MC Dropout + Laplace) | Calibrated obstacle tracks with epistemic uncertainty |
| Prediction | Deep Ensembles + Evidential DL | Directional uncertainty cones, multi-modal forecasts |
| Planning | Hybrid (uncertainty cost) | COLREGs-compliant path with confidence margin |
| Decision | Conformal Risk Control | Bounded collision probability + reject option + audit log |

## Evaluation Metrics

- **Expected Calibration Error (ECE)** — gap between predicted confidence and empirical accuracy
- **Brier Score** — mean squared error of probabilistic predictions
- **Empirical Coverage** — fraction of test instances in conformal prediction set
- **OOD AUROC** — out-of-distribution detection quality
- **COLREGs Violation Rate** — operational safety metric
- **Rejection Rate** — fraction of inputs deferred to human supervisor

## Safety Assurance

Every decision is logged with its uncertainty decomposition, conformal risk
calculation, and reject-option threshold check. This audit trail is the
principal artefact for future STANAG-4671-style safety-assurance evidence.

## License

Apache License 2.0 — see `LICENSE` file. Permissive for DRDO laboratory
extension and integration into classified follow-on work.

## Author

Principal Investigator — Self-funded research effort, single investigator.
Submitted to NRB/DRDO, November 2026.
