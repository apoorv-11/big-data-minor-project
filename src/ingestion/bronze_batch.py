"""
bronze_batch.py — Step 2-3: Ingest the raw PaySim CSV into Bronze Parquet.

Why Bronze?  The medallion architecture stores raw data *as-is* with only
metadata columns added.  No cleaning happens here; that's Silver's job.

Usage:  python -m src.ingestion.bronze_batch [--sample 0.05]
"""

import argparse
import glob
import uuid
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType,
)

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)

# ---------- Explicit schema (no inferSchema — faster and safer) ----------
PAYSIM_SCHEMA = StructType([
    StructField("step",            IntegerType(), True),
    StructField("type",            StringType(),  True),
    StructField("amount",          DoubleType(),  True),
    StructField("nameOrig",        StringType(),  True),
    StructField("oldbalanceOrg",   DoubleType(),  True),
    StructField("newbalanceOrig",  DoubleType(),  True),
    StructField("nameDest",        StringType(),  True),
    StructField("oldbalanceDest",  DoubleType(),  True),
    StructField("newbalanceDest",  DoubleType(),  True),
    StructField("isFraud",         IntegerType(), True),
    StructField("isFlaggedFraud",  IntegerType(), True),
])


def ingest_bronze(sample_fraction: float | None = None):
    """Read raw CSV, add metadata columns, write as Bronze Parquet."""
    spark = get_spark("bronze_batch")

    # ---- Discover CSV file(s) ----
    raw_dir = CONFIG["data"]["raw_dir"]
    csv_pattern = CONFIG["dataset"]["csv_glob"]
    csv_files = sorted(glob.glob(f"{raw_dir}/{csv_pattern}"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{csv_pattern}' in {raw_dir}. "
            "Please copy the PaySim CSV there first."
        )
    logger.info("Found CSV file(s): %s", csv_files)

    # ---- Read with explicit schema ----
    df = spark.read.csv(
        csv_files,
        header=True,
        schema=PAYSIM_SCHEMA,
    )
    total_rows = df.count()
    logger.info("Raw CSV row count: %d", total_rows)

    # ---- Optional sample for dev/debug ----
    if sample_fraction and 0 < sample_fraction < 1:
        df = df.sample(fraction=sample_fraction, seed=42)
        sampled_rows = df.count()
        logger.info("Sampled %.1f%% → %d rows", sample_fraction * 100, sampled_rows)

    # ---- Add Bronze metadata columns ----
    batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    df = (
        df
        .withColumn("_ingest_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_row_id", F.monotonically_increasing_id())
        .withColumn("_source_type", F.lit("batch"))
    )

    # ---- Write Bronze Parquet (overwrite for idempotency) ----
    bronze_dir = CONFIG["data"]["bronze_dir"]
    df.write.mode("overwrite").parquet(bronze_dir)
    written = spark.read.parquet(bronze_dir).count()
    logger.info("Bronze Parquet written to %s — %d rows", bronze_dir, written)

    # ---- Quick dataset verification (report real numbers) ----
    logger.info("--- Dataset Verification ---")
    logger.info("Schema:")
    df.printSchema()
    logger.info("Total rows: %d", written)

    # Distinct transaction types
    types = [r["type"] for r in df.select("type").distinct().collect()]
    logger.info("Transaction types: %s", sorted(types))

    # Fraud count & rate
    fraud_stats = df.agg(
        F.sum("isFraud").alias("fraud_count"),
        F.count("*").alias("total"),
    ).collect()[0]
    fraud_count = fraud_stats["fraud_count"]
    total = fraud_stats["total"]
    fraud_rate = fraud_count / total * 100 if total > 0 else 0
    logger.info("Fraud count: %d / %d (%.4f%%)", fraud_count, total, fraud_rate)

    # Fraud by type
    fraud_by_type = (
        df.filter(F.col("isFraud") == 1)
        .groupBy("type")
        .count()
        .orderBy("type")
        .collect()
    )
    logger.info("Fraud by type: %s",
                {r["type"]: r["count"] for r in fraud_by_type})

    # Step range
    step_range = df.agg(
        F.min("step").alias("min_step"),
        F.max("step").alias("max_step"),
    ).collect()[0]
    logger.info("Step range: %d – %d", step_range["min_step"], step_range["max_step"])

    # isFlaggedFraud stats
    flagged = df.agg(F.sum("isFlaggedFraud")).collect()[0][0]
    logger.info("isFlaggedFraud count: %d", flagged)

    spark.stop()
    logger.info("Bronze ingestion complete ✓")


def main():
    parser = argparse.ArgumentParser(description="Ingest raw CSV → Bronze Parquet")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of data to sample (0-1), e.g. 0.05 for 5%%")
    args = parser.parse_args()
    ingest_bronze(sample_fraction=args.sample)


if __name__ == "__main__":
    main()
