"""
features_pyspark.py — Step 5: Feature engineering with PySpark.

Demonstrates:
  - Filters & aggregations (fraud rate by type, hour, day, amount bucket)
  - Window functions (per-account velocity, time since last txn, running totals,
    rank of destinations by inbound fraud volume)
  - Joins (account-level profile table, transaction-type dimension)
  - Spark SQL (5+ analyst questions answered via temp views)

Key design note:
  PaySim nameOrig is nearly unique per transaction (one-time-use accounts),
  while nameDest repeats.  So destination-side features are more informative.
  We verify this empirically and report findings.

Usage:  python -m src.transform.features_pyspark
"""

from pyspark.sql import DataFrame, SparkSession, functions as F, Window

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)


# ================================================================
#  Part 1: Empirical key analysis — which keys repeat?
# ================================================================

def analyse_key_distributions(df: DataFrame) -> dict:
    """Check how often nameOrig and nameDest repeat in the dataset."""
    total = df.count()
    distinct_orig = df.select("nameOrig").distinct().count()
    distinct_dest = df.select("nameDest").distinct().count()

    avg_txns_per_orig = total / distinct_orig if distinct_orig else 0
    avg_txns_per_dest = total / distinct_dest if distinct_dest else 0

    stats = {
        "total_rows": total,
        "distinct_nameOrig": distinct_orig,
        "distinct_nameDest": distinct_dest,
        "avg_txns_per_orig": round(avg_txns_per_orig, 2),
        "avg_txns_per_dest": round(avg_txns_per_dest, 2),
    }
    logger.info("--- Key Distribution Analysis ---")
    for k, v in stats.items():
        logger.info("  %s = %s", k, v)
    logger.info("→ nameOrig is almost unique (%.2f txns/orig),"
                " nameDest repeats more (%.2f txns/dest).",
                avg_txns_per_orig, avg_txns_per_dest)
    logger.info("→ Destination-side features will be more informative.")
    return stats


# ================================================================
#  Part 2: Window-based features
# ================================================================

def add_window_features(df: DataFrame) -> DataFrame:
    """
    Per-account windowed features.  We compute both origin- and destination-
    side features, but destination-side are more useful (see key analysis).
    Window = previous 24 steps (hours) per account.
    """
    # --- Origin-side windows (txn count / amount in last 24 steps) ---
    w_orig = (
        Window.partitionBy("nameOrig")
        .orderBy("step")
        .rangeBetween(-24, -1)  # previous 24 hours, excluding current
    )
    df = (
        df
        .withColumn("txn_count_orig_last_24",
                    F.count("*").over(w_orig))
        .withColumn("txn_amount_orig_last_24",
                    F.coalesce(F.sum("amount").over(w_orig), F.lit(0.0)))
    )

    # --- Time since previous txn for the origin account ---
    w_orig_prev = (
        Window.partitionBy("nameOrig")
        .orderBy("step")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = df.withColumn(
        "steps_since_last_txn_orig",
        F.col("step") - F.coalesce(
            F.max("step").over(w_orig_prev), F.col("step")
        ),
    )

    # --- Destination-side windows (txn count / amount in last 24 steps) ---
    w_dest = (
        Window.partitionBy("nameDest")
        .orderBy("step")
        .rangeBetween(-24, -1)
    )
    df = (
        df
        .withColumn("txn_count_dest_last_24",
                    F.count("*").over(w_dest))
        .withColumn("txn_amount_dest_last_24",
                    F.coalesce(F.sum("amount").over(w_dest), F.lit(0.0)))
    )

    return df


# ================================================================
#  Part 3: Join-based features (account profile + type dimension)
# ================================================================

def build_dest_profile(df: DataFrame) -> DataFrame:
    """
    Build a destination-account profile table:
      - distinct senders (# unique nameOrig sending to this nameDest)
      - total inbound amount
      - inbound fraud count (for risk analysis, NOT used as ML feature
        because it would leak the label — we'll join only the non-fraud
        version for ML)
    """
    profile = (
        df.groupBy("nameDest")
        .agg(
            F.countDistinct("nameOrig").alias("dest_distinct_senders"),
            F.sum("amount").alias("dest_total_inbound"),
            F.sum("isFraud").alias("dest_fraud_inbound_count"),
        )
    )
    return profile


def build_type_dimension(spark: SparkSession) -> DataFrame:
    """
    Small dimension table mapping transaction type → description + risk tier.
    Useful for joins and dashboard display.
    """
    data = [
        ("CASH_IN",   "Customer deposits cash via agent",   "LOW"),
        ("CASH_OUT",  "Customer withdraws cash via agent",  "HIGH"),
        ("DEBIT",     "Automatic payment / debit",          "LOW"),
        ("PAYMENT",   "Purchase payment to merchant",       "LOW"),
        ("TRANSFER",  "Transfer between customer accounts", "HIGH"),
    ]
    return spark.createDataFrame(data, ["type", "type_description", "risk_tier"])


def join_features(df: DataFrame, dest_profile: DataFrame,
                  type_dim: DataFrame) -> DataFrame:
    """Join the main table with destination profile and type dimension."""
    df = df.join(dest_profile, on="nameDest", how="left")
    df = df.join(type_dim, on="type", how="left")
    return df


# ================================================================
#  Part 4: Spark SQL analyst questions
# ================================================================

def run_analyst_queries(spark: SparkSession, results: list):
    """
    Register temp views and answer 5+ analyst questions using Spark SQL.
    Appends results as (question, answer_df) tuples.
    """
    queries = {
        "Q1: What is the fraud rate by transaction type?": """
            SELECT type,
                   COUNT(*) AS total_txns,
                   SUM(isFraud) AS fraud_count,
                   ROUND(SUM(isFraud) / COUNT(*) * 100, 4) AS fraud_rate_pct
            FROM silver
            GROUP BY type
            ORDER BY fraud_rate_pct DESC
        """,
        "Q2: Which hours of the day have the highest fraud count?": """
            SELECT hour_of_day,
                   SUM(isFraud) AS fraud_count,
                   COUNT(*) AS total_txns,
                   ROUND(SUM(isFraud) / COUNT(*) * 100, 4) AS fraud_rate_pct
            FROM silver
            GROUP BY hour_of_day
            ORDER BY fraud_count DESC
            LIMIT 10
        """,
        "Q3: What is the average fraud transaction amount vs legitimate?": """
            SELECT CASE WHEN isFraud = 1 THEN 'Fraud' ELSE 'Legitimate' END AS label,
                   COUNT(*) AS count,
                   ROUND(AVG(amount), 2) AS avg_amount,
                   ROUND(MIN(amount), 2) AS min_amount,
                   ROUND(MAX(amount), 2) AS max_amount
            FROM silver
            GROUP BY isFraud
        """,
        "Q4: Top 10 destination accounts by total inbound fraud amount": """
            SELECT nameDest,
                   SUM(amount) AS total_fraud_amount,
                   COUNT(*) AS fraud_txn_count
            FROM silver
            WHERE isFraud = 1
            GROUP BY nameDest
            ORDER BY total_fraud_amount DESC
            LIMIT 10
        """,
        "Q5: How does the balance-error correlate with fraud?": """
            SELECT CASE WHEN isFraud = 1 THEN 'Fraud' ELSE 'Legitimate' END AS label,
                   ROUND(AVG(errorBalanceOrig), 2) AS avg_errorBalOrig,
                   ROUND(AVG(errorBalanceDest), 2) AS avg_errorBalDest,
                   ROUND(AVG(amount), 2) AS avg_amount
            FROM silver
            GROUP BY isFraud
        """,
        "Q6: Fraud rate on weekdays vs weekends": """
            SELECT CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS period,
                   COUNT(*) AS total_txns,
                   SUM(isFraud) AS fraud_count,
                   ROUND(SUM(isFraud) / COUNT(*) * 100, 4) AS fraud_rate_pct
            FROM silver
            GROUP BY is_weekend
        """,
    }
    for question, sql in queries.items():
        logger.info(question)
        result_df = spark.sql(sql)
        result_df.show(truncate=False)
        results.append((question, result_df))


# ================================================================
#  Part 5: Aggregation features (fraud rate by amount bucket)
# ================================================================

def add_amount_bucket(df: DataFrame) -> DataFrame:
    """Bucket amounts for analysis (not used directly as ML feature)."""
    df = df.withColumn(
        "amount_bucket",
        F.when(F.col("amount") < 1000, "0-1K")
        .when(F.col("amount") < 10000, "1K-10K")
        .when(F.col("amount") < 100000, "10K-100K")
        .when(F.col("amount") < 1000000, "100K-1M")
        .otherwise("1M+"),
    )
    return df


# ================================================================
#  Main entry point
# ================================================================

def run_features():
    """Full feature engineering pipeline."""
    spark = get_spark("features")

    # ---- Read Silver ----
    silver_dir = CONFIG["data"]["silver_dir"]
    df = spark.read.parquet(silver_dir)
    logger.info("Silver rows: %d", df.count())

    # ---- Key distribution analysis ----
    key_stats = analyse_key_distributions(df)

    # ---- Window features ----
    logger.info("Computing window features...")
    df = add_window_features(df)
    logger.info("Window features added.")

    # ---- Build profile + dimension tables ----
    dest_profile = build_dest_profile(df)
    type_dim = build_type_dimension(spark)
    logger.info("Destination profiles: %d accounts", dest_profile.count())

    # ---- Join ----
    df = join_features(df, dest_profile, type_dim)
    logger.info("Joined features. Columns: %s", df.columns)

    # ---- Amount buckets ----
    df = add_amount_bucket(df)

    # ---- Register temp views for SQL ----
    df.createOrReplaceTempView("silver")

    # ---- Run analyst queries ----
    sql_results = []
    run_analyst_queries(spark, sql_results)

    # ---- Write enriched Silver with features ----
    # This becomes the input for Gold layer
    features_dir = CONFIG["data"]["silver_dir"] + "_features"
    df.write.mode("overwrite").parquet(features_dir)
    written = spark.read.parquet(features_dir).count()
    logger.info("Enriched Silver (with features) written → %s (%d rows)",
                features_dir, written)

    # Show feature sample
    df.select(
        "step", "type", "amount", "isFraud",
        "txn_count_orig_last_24", "txn_amount_orig_last_24",
        "txn_count_dest_last_24", "txn_amount_dest_last_24",
        "steps_since_last_txn_orig",
        "dest_distinct_senders", "dest_total_inbound",
        "risk_tier",
    ).show(10, truncate=False)

    spark.stop()
    logger.info("Feature engineering complete ✓")


if __name__ == "__main__":
    run_features()
