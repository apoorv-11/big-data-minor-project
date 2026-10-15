"""
stream_score.py — Step 10: Score Kafka stream with the trained model.

Uses Spark Structured Streaming to:
  1. Read JSON messages from Kafka topic 'transactions'.
  2. Parse using the PaySim schema.
  3. Build the same features the model expects.
  4. Apply the saved PipelineModel.
  5. Write scored events (with alert flags) to data/streaming/.

Usage:  python -m src.streaming.stream_score
"""

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType,
)

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)

# Schema matching the JSON produced by kafka_producer.py
KAFKA_SCHEMA = StructType([
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


def add_streaming_features(df):
    """
    Add the same derived columns the batch model expects.
    Window features are set to 0 for streaming (no historical context
    in a micro-batch). This is a known limitation of the demo.
    """
    df = (
        df
        .withColumn("hour_of_day", F.col("step") % 24)
        .withColumn("day", (F.col("step") / 24).cast("int") + 1)
        .withColumn("is_weekend",
                    ((((F.col("step") / 24).cast("int")) % 7) >= 5).cast("int"))
        .withColumn("errorBalanceOrig",
                    F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"))
        .withColumn("errorBalanceDest",
                    F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"))
        .withColumn("dest_is_merchant",
                    F.col("nameDest").startswith("M").cast("int"))
        # Window features unavailable in streaming — set to 0
        .withColumn("txn_count_orig_last_24", F.lit(0))
        .withColumn("txn_amount_orig_last_24", F.lit(0.0))
        .withColumn("txn_count_dest_last_24", F.lit(0))
        .withColumn("txn_amount_dest_last_24", F.lit(0.0))
        .withColumn("steps_since_last_txn_orig", F.lit(0))
        .withColumn("dest_distinct_senders", F.lit(0))
        .withColumn("dest_total_inbound", F.lit(0.0))
        # classWeight column (required by model pipeline but unused for scoring)
        .withColumn("classWeight", F.lit(1.0))
    )
    return df


def run_stream_scoring():
    """Start the Structured Streaming job."""
    spark_cfg = CONFIG["spark"]
    spark = get_spark("stream_score")

    # Add Kafka package to Spark
    # (already configured if running via spark-submit with --packages,
    #  but for local mode we need to set it)

    kafka_cfg = CONFIG["kafka"]
    streaming_cfg = CONFIG["streaming"]
    models_dir = CONFIG["data"]["models_dir"]

    # Load the trained model
    model_path = f"{models_dir}/best_pipeline_model"
    model = PipelineModel.load(model_path)
    logger.info("Loaded model from %s", model_path)

    # Read from Kafka
    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_cfg["bootstrap_servers"])
        .option("subscribe", kafka_cfg["topic"])
        .option("startingOffsets", "latest")
        .load()
    )

    # Parse JSON value
    parsed = (
        kafka_df
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(F.from_json("json_str", KAFKA_SCHEMA).alias("data"))
        .select("data.*")
    )

    # Add features
    featured = add_streaming_features(parsed)

    # Score
    scored = model.transform(featured)

    # Extract probability and alert flag
    threshold = CONFIG["ml"]["alert_threshold"]
    from pyspark.ml.functions import vector_to_array
    output = (
        scored
        .withColumn("prob_fraud", vector_to_array(F.col("probability"))[1])
        .withColumn("is_alert", (vector_to_array(F.col("probability"))[1] >= threshold).cast("int"))
        .withColumn("scored_at", F.current_timestamp())
        .select(
            "step", "type", "amount", "nameOrig", "nameDest",
            "isFraud", "prediction", "prob_fraud", "is_alert", "scored_at",
        )
    )

    # Write to parquet (append mode for streaming)
    query = (
        output.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", CONFIG["data"]["streaming_dir"])
        .option("checkpointLocation", streaming_cfg["checkpoint_location"])
        .trigger(processingTime=streaming_cfg["trigger_interval"])
        .start()
    )

    logger.info("Streaming query started. Writing to %s", CONFIG["data"]["streaming_dir"])
    logger.info("Checkpoint: %s", streaming_cfg["checkpoint_location"])
    logger.info("Press Ctrl+C to stop.")

    query.awaitTermination()


if __name__ == "__main__":
    run_stream_scoring()
