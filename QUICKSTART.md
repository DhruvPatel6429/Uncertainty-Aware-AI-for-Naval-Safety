# QUICKSTART — Verify the UQ-AI Naval Framework in 5 Minutes

This guide lets a DRDO/NRB reviewer verify that the framework works end-to-end
without reading the full documentation. Each step is a single command with
expected output.

## Prerequisites

- Python 3.9+
- pip 21+
- (Optional) Node.js 18+ for the dashboard

## Step 1 — Install

```bash
cd uq-naval
pip install -r requirements.txt
pip install -e .
```

**Expected**: `Successfully installed uq-naval-1.0.0 ...`

## Step 2 — Run the test suite

```bash
pytest tests/ -v
```

**Expected**: `69 passed in ~3s` — all tests green.

## Step 3 — List registered UQ methods

```bash
python -c "from uq_naval.uq import list_uq_methods; print(list_uq_methods())"
```

**Expected**: A list of 11 method names:
```
['bayesian', 'bayesian_bb', 'bayesian_laplace', 'conformal', 'conformal_mondrian',
 'conformal_rc', 'ensemble', 'ensemble_batch', 'ensemble_snapshot',
 'evidential_clf', 'evidential_reg']
```

## Step 4 — Run the guided demo

```bash
python demo.py
```

**Expected**: A 7-section walkthrough that:
1. Lists UQ methods
2. Calibrates Bayesian DL (prints ECE before/after, temperature)
3. Produces a prediction with aleatoric + epistemic uncertainty
4. Runs the end-to-end pipeline (prints decision + audit trail)
5. Inspects the audit log
6. Runs the evaluation harness across all 11 methods
7. Stress-tests scripted maritime scenarios

## Step 5 — Run the evaluation harness

```bash
python scripts/evaluate.py --output reports/evaluation.json
python scripts/render_report.py --input reports/evaluation.json --output reports/evaluation.md
```

**Expected**: `reports/evaluation.json` + `reports/evaluation.md` with a
comparison table of all 11 UQ methods (ECE, Brier, coverage, set size, reject rate).

## Step 6 — Start the API service

```bash
python scripts/serve.py
```

**Expected**: FastAPI running on `http://localhost:8000`.
Open `http://localhost:8000/docs` for Swagger UI.

Test a prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_type":"eo","frame_data":[0.5,0.5,0.5,0.5],"frame_shape":[2,2],"own_state":{"lat":0.0,"lon":0.0,"heading_deg":0.0,"speed_kn":10.0}}'
```

**Expected**: A JSON response with `action` (EXECUTE/REJECT_HUMAN/EMERGENCY_FALLBACK),
`collision_probability_bound`, and full `audit_trail`.

## Step 7 — Run the dashboard

```bash
cd dashboard
npm install
npm run dev
```

**Expected**: Next.js dashboard on `http://localhost:3000` with 7 pages:
Overview, Architecture, UQ Methods, Pipeline Demo, Roadmap, Risks, Submission.

## Step 8 — (Optional) Run everything with Docker

```bash
docker-compose up --build
```

**Expected**:
- API on `http://localhost:8000`
- Dashboard on `http://localhost:3000`

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'uq_naval'` | Run `pip install -e .` from the project root |
| `pytest: command not found` | Run `pip install pytest` |
| Tests fail with `torch` errors | The framework falls back to numpy if torch is unavailable; install torch for full functionality |
| Dashboard `npm install` fails | Use Node.js 18+; delete `dashboard/node_modules` and `dashboard/package-lock.json` and retry |
| Port 8000/3000 already in use | Use `--port` flag: `python scripts/serve.py --port 8001` |

## What to look for as a reviewer

1. **Calibration quality** — In the evaluation report (`reports/evaluation.md`),
   check that conformal methods achieve coverage close to the nominal 0.90.
   The Mondrian variant should hit ~0.94 (class-conditional guarantee).

2. **Uncertainty decomposition** — In `demo.py` output, every prediction
   includes both `aleatoric_unc` and `epistemic_unc`. The decomposition is
   method-specific (Bayesian: variance of MC samples; Evidential: Dirichlet
   concentration; Conformal: prediction set size).

3. **Audit trail** — Every pipeline decision is logged with the full
   uncertainty decomposition, conformal risk calculation, and reject-option
   threshold check. See `reports/audit_trail.jsonl` after running `demo.py`.

4. **Reject option** — When epistemic uncertainty exceeds the operational
   threshold, the pipeline produces `REJECT_HUMAN` instead of `EXECUTE`.
   This is the graceful-degradation pathway that distinguishes the framework
   from deterministic baselines.

5. **Plug-in architecture** — All 11 UQ methods implement the same
   `UQInterface`. Switching methods at any pipeline stage is a single-line
   YAML config change.

## For deeper review

- **Full proposal**: `UQ_Naval_Proposal.pdf` (32 pages, ~11,600 words)
- **Architecture**: `docs/architecture.md`
- **API reference**: `docs/api_reference.md`
- **Evaluation methodology**: `docs/evaluation.md`
- **DRDO briefing**: `docs/drdo_briefing.md`
- **Installation guide**: `docs/installation.md`

## Contact

Principal Investigator — self-funded research effort, single investigator.
Submitted to NRB/DRDO, November 2026.
