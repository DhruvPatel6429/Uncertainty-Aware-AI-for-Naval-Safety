# Architecture

This document describes the internal architecture of the UQ-AI Naval framework.
For the proposal-level (conceptual) architecture, see the NRB/DRDO proposal
document (`UQ_Naval_Proposal.pdf`).

## High-level architecture

The framework is organised into six layers, each with a clear responsibility:

```
┌─────────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                            │
│  - REST endpoints: /predict, /uq-methods, /audit-log            │
│  - WebSocket for live streaming                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Layer                                                 │
│  Perception → Prediction → Planning → Decision                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  UQ Method Layer                                                │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ Bayesian   │ │ Conformal  │ │ Ensembles  │ │ Evidential │   │
│  │   DL       │ │ Prediction │ │            │ │     DL     │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
│  All behind a uniform UQInterface                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Model Layer                                                    │
│  - Backbone architectures (MLP, CNN)                             │
│  - Detectors (SimpleDetector, BayesianDetector)                 │
│  - Forecasters (TrajectoryLSTM, TrajectoryTransformer)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Data Layer                                                     │
│  - AISLoader, MAS500Loader                                      │
│  - SyntheticAISDataset, SyntheticDetectionDataset              │
│  - ScenarioLibrary (stress tests)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Cross-cutting: utils (config, logging, audit, seeds)           │
│                  tracking (MLflow)                               │
└─────────────────────────────────────────────────────────────────┘
```

## The unified UQ interface

Every UQ method implements `UQInterface` (in `src/uq_naval/uq/base.py`):

```python
class UQInterface(abc.ABC):
    @abc.abstractmethod
    def calibrate(self, calibration_data) -> CalibrationResult: ...

    @abc.abstractmethod
    def predict(self, x) -> UQOutput: ...
```

This contract guarantees that any UQ method can be plugged into any pipeline
stage. The methods are registered via the `@register_uq_method` decorator
and instantiated by name:

```python
from uq_naval.uq import load_uq_method

uq = load_uq_method("bayesian", model=my_model, config={"n_mc_samples": 10})
```

## UQOutput — the universal prediction contract

Every UQ method returns a `UQOutput` containing:

- `prediction` — the point prediction
- `aleatoric_unc` — irreducible data noise
- `epistemic_unc` — model uncertainty (reducible with data)
- `prediction_set` — conformal prediction set (when applicable)
- `confidence` — calibrated confidence in [0, 1]
- `reject` — whether to defer to a human supervisor
- `audit` — method-specific audit fields

The `to_dict()` method serialises the entire output to JSON, enabling
audit-trail persistence.

## Pipeline composition

The end-to-end pipeline (`NavalNavigationPipeline`) composes four stages:

1. **Perception** wraps a detector + UQ method, producing calibrated
   obstacle detections.
2. **Prediction** wraps a forecaster + UQ method, producing directional
   uncertainty cones for each detected obstacle.
3. **Planning** computes a COLREGs-compliant manoeuvre using uncertainty-aware
   cost functions.
4. **Decision** applies Conformal Risk Control to bound collision probability
   and triggers the reject option when epistemic uncertainty is high.

Each stage's UQ method is injected via configuration — no code changes are
required to swap methods at runtime.

## Plug-in architecture

Adding a new UQ method requires only:

1. Subclass `UQInterface`.
2. Decorate with `@register_uq_method("my_method")`.
3. Implement `calibrate()` and `predict()`.

The new method is automatically:
- Listed by `list_uq_methods()`.
- Instantiable by `load_uq_method("my_method", ...)`.
- Runnable in the evaluation harness.
- Configurable via Hydra YAML.

## Audit trail

Every decision is logged with:

- Per-stage epistemic and aleatoric uncertainty
- Conformal risk calculation
- Reject-option threshold check
- Selected action and reason

This audit trail is the principal artefact for future STANAG-4671-style
safety-assurance evidence.
