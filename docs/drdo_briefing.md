# DRDO Submission Briefing

## Executive Summary

This document summarises the UQ-AI Naval framework for DRDO/NRB reviewers.
The accompanying 32-page proposal (`UQ_Naval_Proposal.pdf`) provides full
technical depth; this briefing is the quick-reference companion.

## What was delivered

| Artefact | Description |
|---|---|
| `UQ_Naval_Proposal.docx` / `.pdf` | 32-page formal NRB/DRDO proposal |
| `uq-naval/` source tree | Complete, runnable framework implementation |
| `dashboard/` | Next.js interactive visualization dashboard |
| `tests/` | PyTest suite (all tests passing) |
| `docs/` | Installation, architecture, evaluation, briefing docs |
| `docker/` | Containerised API + dashboard |

## Key technical claims (verifiable in code)

1. **Four UQ methods** implemented behind a unified interface:
   - Bayesian DL (MC Dropout + Laplace + Bayes-by-Backprop)
   - Conformal Prediction (Inductive + Mondrian + Risk Control)
   - Deep Ensembles (Standard + Snapshot + Batch)
   - Evidential DL (Dirichlet classification + NIG regression)

2. **End-to-end pipeline** with UQ threaded through every stage:
   - Perception (Bayesian + conformal calibration)
   - Prediction (ensemble + evidential)
   - Planning (uncertainty-aware A*/RRT*)
   - Decision (conformal risk control + reject option)

3. **Audit trail** — every decision logged with full uncertainty
   decomposition for future safety-assurance evidence.

4. **Evaluation harness** — runs all UQ methods on a unified benchmark,
   producing the comparison table reported in the proposal.

5. **FastAPI service** exposing the pipeline via REST endpoints.

6. **Interactive dashboard** — 7-page Next.js app for DRDO reviewers
   to explore the framework interactively.

## How to verify each claim

```bash
# 1. Four UQ methods implemented and registered
python -c "from uq_naval.uq import list_uq_methods; print(list_uq_methods())"

# 2. End-to-end pipeline runs
python -c "
from uq_naval.pipeline import NavalNavigationPipeline
# (see tests/test_pipeline.py for a full demo)
"

# 3. Audit trail populated
python scripts/serve.py &
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"frame_data": [0.5]*1024, "frame_shape": [32,32]}'
curl http://localhost:8000/audit-log

# 4. Evaluation harness produces comparison table
python scripts/evaluate.py --n-calibration 50 --n-test 50

# 5. FastAPI service exposes pipeline
python scripts/serve.py
# Swagger UI at http://localhost:8000/docs

# 6. Interactive dashboard
cd dashboard && npm install && npm run dev
# Open http://localhost:3000
```

## Test suite

```bash
pytest tests/ -v
```

All tests pass. The suite covers:
- UQ method calibration + prediction (Bayesian, Conformal, Ensembles, Evidential)
- End-to-end pipeline execution
- Audit-trail population
- API endpoints
- Evaluation harness
- Utility helpers (config, audit, seeds)

## DRDO adoption pathway

1. **Evaluate (M12–M15)** — Install the framework, run the evaluation suite
   on internal datasets, validate calibration properties.
2. **Extend (M15–M24)** — Develop domain-specific modules (sonar at NPOL,
   AUV control at NSTL) that plug into the framework's interfaces.
3. **Integrate (M24–M36)** — Integrate the extended framework into platform-
   specific deployments, including hardware-in-the-loop testing.
4. **Certify (M36+)** — Use the audit-trail artefacts as evidence in a
   future safety-assurance case for STANAG-4671-style certification.

## Contact

Principal Investigator — self-funded research effort, single investigator.
Submitted to NRB/DRDO, November 2026.
