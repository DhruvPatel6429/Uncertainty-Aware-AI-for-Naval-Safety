.PHONY: help install dev test lint clean serve train evaluate dashboard docker

help:
	@echo "UQ-AI Naval — available make targets"
	@echo ""
	@echo "  make install     Install Python package (editable)"
	@echo "  make dev         Install dev dependencies"
	@echo "  make test        Run pytest suite"
	@echo "  make serve       Start FastAPI service on :8000"
	@echo "  make train       Train perception module (default config)"
	@echo "  make evaluate    Run evaluation harness"
	@echo "  make dashboard   Start Next.js dashboard on :3000"
	@echo "  make docker      Build + start all containers (API + dashboard)"
	@echo "  make clean       Remove build artefacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

serve:
	python scripts/serve.py --port 8000

train:
	python scripts/train_perception.py --config configs/pipeline/perception.yaml --uq-method bayesian --epochs 5

evaluate:
	python scripts/evaluate.py --config configs/eval/default.yaml --output reports/evaluation.json

dashboard:
	cd dashboard && npm install && npm run dev

docker:
	docker-compose up --build

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	rm -rf dashboard/.next dashboard/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} +
