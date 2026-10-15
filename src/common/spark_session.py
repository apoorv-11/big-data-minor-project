"""
spark_session.py — Create and configure the SparkSession.
Reads settings from config.yaml so nothing is hard-coded.
"""

import os
import sys
from pyspark.sql import SparkSession
from src.common.config import CONFIG
from src.common.logger import get_logger

# Guarantee worker and driver use the exact same virtualenv Python binary
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logger = get_logger(__name__)


def get_spark(app_name_suffix: str = "") -> SparkSession:
    """
    Build a SparkSession in local mode with tuned settings.
    app_name_suffix lets callers add context, e.g. 'bronze', 'ml'.
    """
    spark_cfg = CONFIG["spark"]
    app_name = spark_cfg["app_name"]
    if app_name_suffix:
        app_name = f"{app_name} — {app_name_suffix}"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(spark_cfg["master"])
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"])
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        # Silence verbose INFO logs from Spark internals
        .config("spark.ui.showConsoleProgress", "false")
        # Use Parquet as default data source
        .config("spark.sql.sources.default", "parquet")
        # Prevent Spark from creating metastore_db in the project root
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .config(
            "javax.jdo.option.ConnectionURL",
            "jdbc:derby:memory:metastore_db;create=true",
        )
    )

    spark = builder.getOrCreate()
    # Reduce Spark's own log noise
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created: %s (master=%s, driver_memory=%s)",
                app_name, spark_cfg["master"], spark_cfg["driver_memory"])
    return spark
