"""
predict_batch.py — Step 8: Apply saved model to test data in batch mode.

Usage:  python -m src.ml.predict_batch
"""

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)


def batch_predict():
    spark = get_spark("predict_batch")
    cfg_ml = CONFIG["ml"]

    # Load saved model
    models_dir = CONFIG["data"]["models_dir"]
    model_path = f"{models_dir}/best_pipeline_model"
    model = PipelineModel.load(model_path)
    logger.info("Loaded model from %s", model_path)

    # Read gold features (test set only)
    gold_dir = CONFIG["data"]["gold_dir"]
    df = spark.read.parquet(f"{gold_dir}/gold_features")
    test_df = df.filter(F.col("step") >= cfg_ml["test_min_step"])
    # Add classWeight column (required by model but doesn't affect predictions)
    test_df = test_df.withColumn("classWeight", F.lit(1.0))
    logger.info("Test rows: %d", test_df.count())

    # Predict
    preds = model.transform(test_df)

    # Extract P(fraud)
    if "probability" in preds.columns:
        from pyspark.sql.types import DoubleType
        get_prob_udf = F.udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())
        preds = preds.withColumn(
            "prob_fraud", get_prob_udf(F.col("probability"))
        )

    # Alert flag
    threshold = cfg_ml["alert_threshold"]
    preds = preds.withColumn(
        "is_alert",
        (F.col("prob_fraud") >= threshold).cast("int"),
    )

    # Save
    preds_dir = CONFIG["data"]["predictions_dir"]
    output_cols = [
        "_row_id", "step", "type", "amount", "nameOrig", "nameDest",
        "isFraud", "prediction", "prob_fraud", "is_alert",
    ]
    existing = [c for c in output_cols if c in preds.columns]
    preds.select(existing).write.mode("overwrite").parquet(preds_dir)
    logger.info("Predictions written to %s", preds_dir)

    # Summary
    alert_count = preds.filter(F.col("is_alert") == 1).count()
    logger.info("Alerts (prob >= %.2f): %d", threshold, alert_count)

    spark.stop()
    logger.info("Batch prediction complete ✓")


if __name__ == "__main__":
    batch_predict()
