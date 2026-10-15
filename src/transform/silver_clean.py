"""
silver_clean.py — Step 4: Clean, validate, and enrich Bronze → Silver.

Rules:
  - Enforce correct types (already done via explicit schema in Bronze).
  - Trim string columns.
  - Drop exact duplicate rows.
  - Validate: amount > 0, type in the 5 known types, balances >= 0,
    step in [1, 743], non-empty ids.
  - Invalid rows → silver_rejects with reject_reason (never silently dropped).
  - Add derived columns: hour_of_day, day, is_weekend, errorBalanceOrig,
    errorBalanceDest, dest_is_merchant.

Usage:  python -m src.transform.silver_clean [--sample 0.05]
"""

import argparse

from pyspark.sql import DataFrame, functions as F

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)


# ---------- Validation helpers ----------

def validate_rows(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Split df into (valid, rejects).  Each reject row gets a reject_reason.
    We build a list of conditions and OR them; rows failing ANY are rejected.
    Assumption: step 1 = hour 1 of day 1 (Monday).
    """
    valid_types = CONFIG["dataset"]["valid_types"]
    max_step = CONFIG["dataset"]["max_step"]

    # Build individual rejection flags
    # F.when without .otherwise() returns NULL for passing conditions.
    # We collect all reason strings, then filter out NULLs.
    reasons_array = F.array(
        F.when(F.col("amount") <= 0, F.lit("amount<=0")),
        F.when(~F.col("type").isin(valid_types), F.lit("invalid_type")),
        F.when(F.col("oldbalanceOrg") < 0, F.lit("oldbalanceOrg<0")),
        F.when(F.col("newbalanceOrig") < 0, F.lit("newbalanceOrig<0")),
        F.when(F.col("oldbalanceDest") < 0, F.lit("oldbalanceDest<0")),
        F.when(F.col("newbalanceDest") < 0, F.lit("newbalanceDest<0")),
        F.when(
            (F.col("step") < 1) | (F.col("step") > max_step),
            F.lit("step_out_of_range"),
        ),
        F.when(
            F.col("nameOrig").isNull() | (F.trim("nameOrig") == ""),
            F.lit("empty_nameOrig"),
        ),
        F.when(
            F.col("nameDest").isNull() | (F.trim("nameDest") == ""),
            F.lit("empty_nameDest"),
        ),
        F.when(F.col("step").isNull(), F.lit("null_step")),
        F.when(F.col("amount").isNull(), F.lit("null_amount")),
        F.when(F.col("isFraud").isNull(), F.lit("null_isFraud")),
    )
    # Filter out NULLs (passing conditions) from the array
    df = df.withColumn(
        "_reject_reasons",
        F.filter(reasons_array, lambda x: x.isNotNull()),
    )

    # Split
    has_rejects = F.size("_reject_reasons") > 0
    rejects = (
        df.filter(has_rejects)
        .withColumn("reject_reason", F.array_join("_reject_reasons", "; "))
        .drop("_reject_reasons")
    )
    valid = df.filter(~has_rejects).drop("_reject_reasons")

    return valid, rejects


def add_derived_columns(df: DataFrame) -> DataFrame:
    """
    Add columns useful for analysis and ML.
    Assumption: step 1 = hour 1 of day 1.  We treat day 1 as Monday,
    so is_weekend = True for days 6 & 7 (and 13 & 14, etc.).
    """
    df = (
        df
        # Time features
        .withColumn("hour_of_day", F.col("step") % 24)
        .withColumn("day", (F.col("step") / 24).cast("int") + 1)
        .withColumn("is_weekend",
                    ((((F.col("step") / 24).cast("int")) % 7) >= 5).cast("int"))
        # Balance-error features (captures simulator artefacts / fraud signals)
        .withColumn(
            "errorBalanceOrig",
            F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"),
        )
        .withColumn(
            "errorBalanceDest",
            F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"),
        )
        # Merchant flag (nameDest starting with 'M')
        .withColumn("dest_is_merchant",
                    F.col("nameDest").startswith("M").cast("int"))
    )
    return df


def run_silver(sample_fraction: float | None = None):
    """Main Silver pipeline: read Bronze, dedupe, validate, enrich, write."""
    spark = get_spark("silver_clean")

    # ---- Read Bronze ----
    bronze_dir = CONFIG["data"]["bronze_dir"]
    df = spark.read.parquet(bronze_dir)
    bronze_count = df.count()
    logger.info("Bronze rows read: %d", bronze_count)

    if sample_fraction and 0 < sample_fraction < 1:
        df = df.sample(fraction=sample_fraction, seed=42)
        logger.info("Sampled %.1f%% → %d rows", sample_fraction * 100, df.count())

    # ---- Trim strings ----
    string_cols = ["type", "nameOrig", "nameDest"]
    for col_name in string_cols:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))

    # ---- Drop exact duplicates (based on original columns only) ----
    original_cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    ]
    before_dedup = df.count()
    df = df.dropDuplicates(original_cols)
    after_dedup = df.count()
    dupes_removed = before_dedup - after_dedup
    logger.info("Deduplication: %d → %d (removed %d exact duplicates)",
                before_dedup, after_dedup, dupes_removed)

    # ---- Validate ----
    valid, rejects = validate_rows(df)
    valid_count = valid.count()
    reject_count = rejects.count()
    logger.info("Validation: %d valid, %d rejected", valid_count, reject_count)

    # ---- Add derived columns ----
    silver = add_derived_columns(valid)

    # ---- Write Silver ----
    silver_dir = CONFIG["data"]["silver_dir"]
    silver.write.mode("overwrite").parquet(silver_dir)
    written = spark.read.parquet(silver_dir).count()
    logger.info("Silver written to %s — %d rows", silver_dir, written)

    # ---- Write rejects ----
    rejects_dir = CONFIG["data"]["silver_rejects_dir"]
    if reject_count > 0:
        rejects.write.mode("overwrite").parquet(rejects_dir)
        logger.info("Rejects written to %s — %d rows", rejects_dir, reject_count)
    else:
        logger.info("No rejected rows")

    # ---- Log reject reasons breakdown ----
    if reject_count > 0:
        logger.info("--- Reject Reasons ---")
        rejects.groupBy("reject_reason").count().orderBy(
            F.desc("count")
        ).show(truncate=False)

    # ---- Summary for data quality report ----
    logger.info("--- Data Quality Summary ---")
    logger.info("Bronze rows:        %d", bronze_count)
    logger.info("After dedup:        %d (-%d)", after_dedup, dupes_removed)
    logger.info("Valid (Silver):     %d", valid_count)
    logger.info("Rejected:           %d", reject_count)
    logger.info("Silver columns:     %s", silver.columns)

    # Show derived column sample
    silver.select(
        "step", "type", "amount", "hour_of_day", "day", "is_weekend",
        "errorBalanceOrig", "errorBalanceDest", "dest_is_merchant",
    ).show(5, truncate=False)

    spark.stop()
    logger.info("Silver cleaning complete ✓")


def main():
    parser = argparse.ArgumentParser(description="Clean Bronze → Silver")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of data to sample (0-1)")
    args = parser.parse_args()
    run_silver(sample_fraction=args.sample)


if __name__ == "__main__":
    main()
