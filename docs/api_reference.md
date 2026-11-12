# API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### `GET /` — Service status

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uq_methods": ["bayesian", "bayesian_bb", "bayesian_laplace", "conformal", ...],
  "pipeline_loaded": false
}
```

---

### `GET /uq-methods` — List available UQ methods

**Response:**
```json
{
  "methods": ["bayesian", "bayesian_bb", "bayesian_laplace", "conformal", ...]
}
```

---

### `POST /predict` — Run the pipeline on a sensor frame

**Request body:**
```json
{
  "sensor_type": "eo",
  "frame_data": [0.5, 0.5, ...],   // flattened sensor frame
  "frame_shape": [32, 32],
  "own_state": {
    "lat": 0.0,
    "lon": 0.0,
    "heading_deg": 0.0,
    "speed_kn": 10.0
  }
}
```

**Response:**
```json
{
  "action": "EXECUTE",
  "collision_probability_bound": 0.00008,
  "reject_reason": null,
  "audit_trail": {
    "action": "EXECUTE",
    "perception_uncertainty": 0.12,
    "prediction_uncertainty": 0.18,
    "total_epistemic": 0.16,
    "collision_prob_bound": 0.00008,
    "path_cost": 0.012,
    "colregs_compliance": 1.0,
    "uncertainty_margin_m": 51.6
  },
  "n_detections": 5,
  "mean_epistemic_uncertainty": 0.16
}
```

The `action` field is one of:
- `"EXECUTE"` — autonomously execute the planned path
- `"REJECT_HUMAN"` — escalate to human supervisor
- `"EMERGENCY_FALLBACK"` — execute pre-defined conservative manoeuvre

---

### `GET /audit-log` — Get the audit log

Returns all decisions logged since pipeline startup.

**Response:**
```json
{
  "entries": [
    {
      "action": "EXECUTE",
      "perception_uncertainty": 0.12,
      "prediction_uncertainty": 0.18,
      "total_epistemic": 0.16,
      "collision_prob_bound": 0.00008,
      "path_cost": 0.012,
      "colregs_compliance": 1.0,
      "uncertainty_margin_m": 51.6
    }
  ]
}
```

## Programmatic access (Python)

```python
import requests

# Status
r = requests.get("http://localhost:8000/").json()
print(r["uq_methods"])

# Predict
r = requests.post("http://localhost:8000/predict", json={
    "sensor_type": "eo",
    "frame_data": [0.5] * 1024,
    "frame_shape": [32, 32],
    "own_state": {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0},
}).json()
print(r["action"], r["collision_probability_bound"])
```

## Swagger UI

Interactive API documentation is available at:
```
http://localhost:8000/docs
```
