# Installation

## Prerequisites

- **Python** 3.9 or later
- **pip** 21.0+
- (Optional) **CUDA** 11.8+ with a CUDA-capable GPU for accelerated training
- (Optional) **Node.js** 18+ and npm for the dashboard

## Step 1 — Clone the repository

```bash
git clone <your-repo-url> uq-naval
cd uq-naval
```

## Step 2 — Set up a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

This installs:

- `torch` and `torchvision` — model implementations
- `hydra-core`, `omegaconf` — configuration management
- `mlflow` — experiment tracking
- `fastapi`, `uvicorn`, `pydantic` — API service layer
- `pytest`, `pytest-cov` — testing

## Step 4 — Verify the installation

```bash
pytest tests/ -v
```

All tests should pass. If any fail, please file an issue with the full traceback.

## Step 5 — (Optional) Install the dashboard

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

## Step 6 — (Optional) Run everything with Docker

```bash
docker-compose up --build
```

This starts:
- The FastAPI service on `http://localhost:8000`
- The Next.js dashboard on `http://localhost:3000`

## Troubleshooting

### PyTorch installation fails

If the default PyTorch wheel is incompatible with your CUDA version, install
a CUDA-specific wheel from the [PyTorch website](https://pytorch.org/get-started/locally/).

### `ModuleNotFoundError: No module named 'uq_naval'`

Make sure you ran `pip install -e .` from the project root. If the issue
persists, verify that `src/uq_naval/__init__.py` exists and that your
PYTHONPATH includes the `src/` directory.

### Tests fail with "torch not available"

The framework's tests gracefully skip torch-only paths when torch is
unavailable, but several advanced features (Bayes-by-Backprop, Laplace
approximation on real models) require torch. Install torch to enable these.

### Dashboard fails to build

Make sure you're using Node.js 18 or later. The dashboard uses Next.js 14,
which requires Node 18+.
