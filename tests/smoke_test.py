"""
smoke_test.py — Verify Spark + Parquet works in this environment.
Run: python -m tests.smoke_test
"""
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger("smoke_test")


def main():
    logger.info("Starting Spark smoke test...")
    spark = get_spark("smoke_test")

    # Create a tiny DataFrame
    data = [(1, "Alice", 100.0), (2, "Bob", 200.0), (3, "Carol", 300.0)]
    df = spark.createDataFrame(data, ["id", "name", "amount"])
    logger.info("Created DataFrame with %d rows", df.count())
    df.show()

    # Write to Parquet
    out_path = tempfile.mkdtemp() + "/smoke_test.parquet"
    df.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote Parquet to %s", out_path)

    # Read it back
    df2 = spark.read.parquet(out_path)
    assert df2.count() == 3, f"Expected 3 rows, got {df2.count()}"
    logger.info("Read back %d rows — Spark + Parquet OK ✓", df2.count())

    # Show Spark version
    logger.info("Spark version: %s", spark.version)
    logger.info("Java version: %s",
                spark.sparkContext._jvm.System.getProperty("java.version"))

    spark.stop()
    logger.info("Smoke test PASSED ✓")


if __name__ == "__main__":
    main()
