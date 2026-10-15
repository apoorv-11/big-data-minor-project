# ===========================================================
#  FraudShield Makefile — one-command targets for every step
# ===========================================================
SHELL     := /bin/bash
VENV      := .venv
PY        := $(VENV)/bin/python
VENV_AF   := .venv-airflow
PY_AF     := $(VENV_AF)/bin/python

# Explicitly ensure Spark worker processes use the venv Python, not system Python
export PYSPARK_PYTHON        := $(abspath $(PY))
export PYSPARK_DRIVER_PYTHON := $(abspath $(PY))

.PHONY: help setup setup-airflow up down bronze silver features gold \
        train predict stream-start stream-stop dashboard test all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- Environment ----
setup:  ## Create main venv and install deps
	uv venv --python 3.12 $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

setup-airflow:  ## Create Airflow venv with constraints
	uv venv --python 3.12 $(VENV_AF)
	AIRFLOW_VERSION=2.10.3 && \
	PYTHON_VERSION=3.12 && \
	CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-$${AIRFLOW_VERSION}/constraints-$${PYTHON_VERSION}.txt" && \
	$(VENV_AF)/bin/pip install "apache-airflow==$${AIRFLOW_VERSION}" --constraint "$${CONSTRAINT_URL}" pyyaml

# ---- Kafka ----
up:  ## Start Kafka (docker compose)
	docker compose up -d

down:  ## Stop Kafka
	docker compose down

# ---- Pipeline stages ----
bronze:  ## Step 3: Ingest raw CSV → Bronze Parquet
	$(PY) -m src.ingestion.bronze_batch

silver:  ## Step 4: Clean/validate Bronze → Silver
	$(PY) -m src.transform.silver_clean

features:  ## Step 5: Engineer features in PySpark
	$(PY) -m src.transform.features_pyspark

gold:  ## Step 6: Build Gold tables
	$(PY) -m src.transform.gold_build

train:  ## Step 7-8: Train and evaluate ML models
	$(PY) -m src.ml.train_model

predict:  ## Step 8: Batch predictions on test set
	$(PY) -m src.ml.predict_batch

# ---- Streaming ----
stream-produce:  ## Replay transactions to Kafka
	$(PY) -m src.ingestion.kafka_producer

stream-score:  ## Score Kafka stream with trained model
	$(PY) -m src.streaming.stream_score

# ---- Dashboard ----
dashboard:  ## Launch Streamlit dashboard
	$(VENV)/bin/streamlit run src/dashboard/app.py --server.port 8501

# ---- Testing ----
test:  ## Run pytest suite
	$(PY) -m pytest tests/ -v

# ---- Full pipeline ----
all: bronze silver features gold train predict  ## Run full batch pipeline

clean:  ## Remove generated data (keep raw)
	rm -rf data/bronze data/silver data/silver_rejects data/gold \
	       data/predictions data/streaming data/checkpoints data/models \
	       spark-warehouse metastore_db derby.log
	mkdir -p data/{bronze,silver,gold,predictions,streaming,checkpoints}
