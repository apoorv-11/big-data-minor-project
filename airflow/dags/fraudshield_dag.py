"""
fraudshield_dag.py — Airflow DAG for FraudShield End-to-End Pipeline
====================================================================
Orchestrates the Medallion Architecture data pipeline:
  Raw CSV -> Bronze -> Silver -> Feature Engineering -> Gold -> ML Train -> Batch Predict

Schedule: Daily (or manual trigger)
Executor: BashOperator executing in the project virtualenv
"""

from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator

# Path to project root and virtualenv python
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"

default_args = {
    "owner": "fraud_data_engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraudshield_batch_pipeline",
    default_args=default_args,
    description="FraudShield End-to-End Medallion Big Data & ML Pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["fraud_detection", "big_data", "pyspark", "ml"],
) as dag:

    # Step 3: Bronze Layer Ingestion
    bronze_task = BashOperator(
        task_id="ingest_bronze",
        bash_command=f"cd {PROJECT_DIR} && {VENV_PYTHON} -m src.ingestion.bronze_batch",
    )

    # Step 4: Silver Layer Cleaning & Validation
    silver_task = BashOperator(
        task_id="clean_silver",
        bash_command=f"cd {PROJECT_DIR} && {VENV_PYTHON} -m src.transform.silver_clean",
    )

    # Step 5: Silver Feature Engineering (Rolling Windows & Spark SQL)
    features_task = BashOperator(
        task_id="engineer_features",
        bash_command=f"cd {PROJECT_DIR} && {VENV_PYTHON} -m src.transform.features_pyspark",
    )

    # Step 6: Gold Layer Aggregations & ML-Ready Tables
    gold_task = BashOperator(
        task_id="build_gold",
        bash_command=f"cd {PROJECT_DIR} && {VENV_PYTHON} -m src.transform.gold_build",
    )

    # Step 7: Train & Evaluate Spark MLlib Classifiers
    train_task = BashOperator(
        task_id="train_ml_models",
        bash_command=f"cd {PROJECT_DIR} && {VENV_PYTHON} -m src.ml.train_model",
    )

    # Step 8: Batch Predictions & Alert Generation
    predict_task = BashOperator(
        task_id="batch_predict",
        bash_command=f"cd {PROJECT_DIR} && {VENV_PYTHON} -m src.ml.predict_batch",
    )

    # Define Linear Task Dependencies
    bronze_task >> silver_task >> features_task >> gold_task >> train_task >> predict_task
