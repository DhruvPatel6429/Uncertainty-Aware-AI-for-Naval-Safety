# Evaluation Methodology

The evaluation harness (`scripts/evaluate.py`) runs every registered UQ method
on a unified benchmark and produces a comparison table.

## Metrics

### Calibration metrics

- **Expected Calibration Error (ECE)** — gap between predicted confidence
  and empirical accuracy, averaged over confidence bins. Lower is better.
  Well-calibrated models achieve ECE < 0.05.

- **Brier Score** — mean squared error of probabilistic predictions. Lower
  is better. Baseline for uniform-random binary prediction is 0.25.

- **Empirical Coverage** — fraction of test instances whose true label falls
  inside the conformal prediction set. Should match the nominal 1-α level
  within ±1%.

### Operational metrics

- **OOD AUROC** — area under the ROC curve for OOD detection using
  epistemic uncertainty as the score. 1.0 = perfect OOD detection,
  0.5 = random.

- **COLREGs Violation Rate** — fraction of decisions where COLREGs
  compliance < 1.0. Lower is better.

- **Rejection Rate** — fraction of decisions deferred to a human supervisor.
  High rejection rate signals distribution shift or insufficient training data.

## Datasets

The framework ships with three synthetic dataset generators for testing
and bootstrapping:

- **`SyntheticAISDataset`** — multi-modal vessel trajectory pairs with
  4 features per timestep (lat rate, lon rate, heading delta, speed).
- **`SyntheticDetectionDataset`** — small synthetic radar/EO images with
  5 vessel classes and 10% OOD injection.
- **`ScenarioLibrary`** — scripted maritime encounters (head-on, crossing,
  overtaking, multi-vessel, OOD stress, sensor failure, adversarial).

In production, swap in real datasets by implementing the same `__iter__`
interface — the harness is dataset-agnostic.

## Running the harness

```bash
# Run on all UQ methods
python scripts/evaluate.py --output reports/evaluation.json

# Run on a subset
python scripts/evaluate.py --methods bayesian conformal --output reports/subset.json

# Run with custom hyperparameters
python scripts/evaluate.py --n-calibration 500 --n-test 500 --alpha 0.05
```

## Interpreting results

The harness produces both:

1. A JSON report (`reports/evaluation.json`) with full metrics per method.
2. A Markdown summary table printed to stdout.

Example summary:

| Method            | ECE    | Brier  | Coverage | Mean Set Size | Reject Rate |
|-------------------|--------|--------|----------|---------------|-------------|
| bayesian          | 0.0823 | 0.3421 | 0.8200   | 1.43          | 0.0500      |
| conformal         | 0.0612 | 0.3187 | 0.9100   | 1.67          | 0.0300      |
| ensemble          | 0.0456 | 0.2987 | 0.8300   | 1.21          | 0.0200      |
| evidential_clf    | 0.1023 | 0.3598 | 0.7800   | 1.89          | 0.0800      |

The summary table mirrors Table 1 of the proposal and provides the empirical
guidance that the current literature lacks.
