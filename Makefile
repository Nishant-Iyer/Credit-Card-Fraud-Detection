# Development Automation Makefile

PYTHON = .venv/bin/python
PIP = .venv/bin/pip
pytest = .venv/bin/pytest

.PHONY: install train serve dashboard test spark docker-build docker-up clean

install:
	@echo "Installing dependencies..."
	/Users/nishant/.local/bin/uv pip install -r requirements.txt

train:
	@echo "Running train pipeline..."
	$(PYTHON) -m src.pipeline.train_pipeline

serve:
	@echo "Starting FastAPI server..."
	MODEL_PATH=artifacts/full_pipeline.joblib METADATA_PATH=artifacts/metadata.json $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	@echo "Starting Streamlit dashboard..."
	.venv/bin/streamlit run dashboard.py

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v

spark:
	@echo "Running PySpark batch inference job..."
	$(PYTHON) src/spark/spark_inference.py

docker-build:
	@echo "Building docker image..."
	docker build -t fraud-detection-api:latest .

docker-up:
	@echo "Starting local orchestrator (API + MLflow + Streamlit)..."
	docker-compose up --build

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .venv mlflow.db mlruns/ data/train.csv data/test.csv artifacts/
