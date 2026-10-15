"""
gold_build.py — Step 6: Build Gold-layer analysis tables from enriched Silver.

Outputs:
  - gold_features:  ML-ready table (one row per transaction, numeric + encoded)
  - gold_kpis:      Fraud totals/rates by type, day, hour, amount bucket
  - gold_account_risk:  Top risky destination accounts
  - gold_daily_summary: Daily aggregates

Usage:  python -m src.transform.gold_build
"""

from pyspark.sql import functions as F

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_gold():
    spark = get_spark("gold_build")
    gold_dir = CONFIG["data"]["gold_dir"]

    # ---- Read enriched Silver (with features) ----
    features_dir = CONFIG["data"]["silver_dir"] + "_features"
    df = spark.read.parquet(features_dir)
    total = df.count()
    logger.info("Enriched Silver rows: %d", total)

    # ================================================================
    #  Gold 1: gold_features — ML-ready table
    # ================================================================
    ml_cols = (
        ["step", "type", "amount", "nameOrig", "nameDest",
         "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
         "isFraud",
         "hour_of_day", "day", "is_weekend",
         "errorBalanceOrig", "errorBalanceDest", "dest_is_merchant",
         "txn_count_orig_last_24", "txn_amount_orig_last_24",
         "txn_count_dest_last_24", "txn_amount_dest_last_24",
         "steps_since_last_txn_orig",
         "dest_distinct_senders", "dest_total_inbound",
         "dest_fraud_inbound_count",
         "amount_bucket", "_row_id"]
    )
    # Keep only columns that exist (in case some are missing)
    existing = set(df.columns)
    ml_cols = [c for c in ml_cols if c in existing]

    gold_features = df.select(ml_cols)
    # Fill nulls in numeric feature columns with 0
    numeric_fill = {
        "txn_count_orig_last_24": 0, "txn_amount_orig_last_24": 0.0,
        "txn_count_dest_last_24": 0, "txn_amount_dest_last_24": 0.0,
        "steps_since_last_txn_orig": 0,
        "dest_distinct_senders": 0, "dest_total_inbound": 0.0,
        "dest_fraud_inbound_count": 0,
    }
    gold_features = gold_features.fillna(numeric_fill)
    out = f"{gold_dir}/gold_features"
    gold_features.write.mode("overwrite").parquet(out)
    logger.info("gold_features → %s (%d rows, %d cols)",
                out, gold_features.count(), len(gold_features.columns))

    # ================================================================
    #  Gold 2: gold_kpis — aggregated fraud stats
    # ================================================================
    kpi_by_type = (
        df.groupBy("type").agg(
            F.count("*").alias("total_txns"),
            F.sum("isFraud").alias("fraud_count"),
            F.round(F.sum("isFraud") / F.count("*") * 100, 4).alias("fraud_rate_pct"),
            F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount"))), 2)
            .alias("fraud_amount"),
        )
        .withColumn("dimension", F.lit("type"))
        .withColumnRenamed("type", "dimension_value")
    )
    kpi_by_hour = (
        df.groupBy("hour_of_day").agg(
            F.count("*").alias("total_txns"),
            F.sum("isFraud").alias("fraud_count"),
            F.round(F.sum("isFraud") / F.count("*") * 100, 4).alias("fraud_rate_pct"),
            F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount"))), 2)
            .alias("fraud_amount"),
        )
        .withColumn("dimension", F.lit("hour_of_day"))
        .withColumn("dimension_value", F.col("hour_of_day").cast("string"))
        .drop("hour_of_day")
    )
    kpi_by_day = (
        df.groupBy("day").agg(
            F.count("*").alias("total_txns"),
            F.sum("isFraud").alias("fraud_count"),
            F.round(F.sum("isFraud") / F.count("*") * 100, 4).alias("fraud_rate_pct"),
            F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount"))), 2)
            .alias("fraud_amount"),
        )
        .withColumn("dimension", F.lit("day"))
        .withColumn("dimension_value", F.col("day").cast("string"))
        .drop("day")
    )
    kpi_by_bucket = (
        df.groupBy("amount_bucket").agg(
            F.count("*").alias("total_txns"),
            F.sum("isFraud").alias("fraud_count"),
            F.round(F.sum("isFraud") / F.count("*") * 100, 4).alias("fraud_rate_pct"),
            F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount"))), 2)
            .alias("fraud_amount"),
        )
        .withColumn("dimension", F.lit("amount_bucket"))
        .withColumnRenamed("amount_bucket", "dimension_value")
    )
    # Union all KPIs
    cols = ["dimension", "dimension_value", "total_txns", "fraud_count",
            "fraud_rate_pct", "fraud_amount"]
    gold_kpis = (
        kpi_by_type.select(cols)
        .unionByName(kpi_by_hour.select(cols))
        .unionByName(kpi_by_day.select(cols))
        .unionByName(kpi_by_bucket.select(cols))
    )
    out = f"{gold_dir}/gold_kpis"
    gold_kpis.write.mode("overwrite").parquet(out)
    logger.info("gold_kpis → %s (%d rows)", out, gold_kpis.count())

    # ================================================================
    #  Gold 3: gold_account_risk — top risky destination accounts
    # ================================================================
    gold_account_risk = (
        df.filter(F.col("isFraud") == 1)
        .groupBy("nameDest")
        .agg(
            F.count("*").alias("fraud_txn_count"),
            F.round(F.sum("amount"), 2).alias("total_fraud_amount"),
            F.countDistinct("nameOrig").alias("distinct_fraud_senders"),
            F.round(F.avg("amount"), 2).alias("avg_fraud_amount"),
        )
        .orderBy(F.desc("total_fraud_amount"))
        .limit(500)
    )
    out = f"{gold_dir}/gold_account_risk"
    gold_account_risk.write.mode("overwrite").parquet(out)
    logger.info("gold_account_risk → %s (%d rows)", out, gold_account_risk.count())

    # ================================================================
    #  Gold 4: gold_daily_summary
    # ================================================================
    gold_daily = (
        df.groupBy("day").agg(
            F.count("*").alias("total_txns"),
            F.sum("isFraud").alias("fraud_count"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount"))), 2)
            .alias("fraud_amount"),
            F.countDistinct("nameOrig").alias("unique_senders"),
            F.countDistinct("nameDest").alias("unique_receivers"),
        )
        .orderBy("day")
    )
    out = f"{gold_dir}/gold_daily_summary"
    gold_daily.write.mode("overwrite").parquet(out)
    logger.info("gold_daily_summary → %s (%d rows)", out, gold_daily.count())

    # ---- Overall totals for dashboard ----
    totals = df.agg(
        F.count("*").alias("total_txns"),
        F.sum("isFraud").alias("total_fraud"),
        F.round(F.sum("isFraud") / F.count("*") * 100, 4).alias("fraud_rate_pct"),
        F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount"))), 2)
        .alias("total_fraud_amount"),
    ).collect()[0]
    logger.info("--- Overall Totals ---")
    logger.info("  Total txns:    %d", totals["total_txns"])
    logger.info("  Fraud count:   %d", totals["total_fraud"])
    logger.info("  Fraud rate:    %.4f%%", totals["fraud_rate_pct"])
    logger.info("  Fraud amount:  %.2f", totals["total_fraud_amount"])

    spark.stop()
    logger.info("Gold build complete ✓")


if __name__ == "__main__":
    build_gold()
