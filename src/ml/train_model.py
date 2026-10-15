"""
train_model.py — Step 7-8: Train ML models on Gold features, evaluate, save.

Pipeline:
  1. Read gold_features, split by TIME (not random).
  2. Apply class weights for imbalanced data.
  3. Build Spark ML Pipeline: StringIndexer + OneHotEncoder + VectorAssembler.
  4. Train LogisticRegression, RandomForest, GBTClassifier.
  5. Evaluate on PR-AUC (primary), ROC-AUC, precision, recall, F1.
  6. Save best model, predictions, metrics, PR curve points.

Leakage rules:
  - isFlaggedFraud is NOT used as a feature (rule-based, leaks label).
  - IDs (nameOrig, nameDest) are NOT used as features.
  - dest_fraud_inbound_count is NOT used (leaks label info).

Usage:  python -m src.ml.train_model
"""

import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.sql import functions as F

from src.common.config import CONFIG
from src.common.spark_session import get_spark
from src.common.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate():
    spark = get_spark("train_model")
    cfg_ml = CONFIG["ml"]

    # ---- Read Gold Features ----
    gold_dir = CONFIG["data"]["gold_dir"]
    df = spark.read.parquet(f"{gold_dir}/gold_features")
    total = df.count()
    logger.info("Gold features: %d rows", total)

    # ---- Time-based split (NO random split — avoids leakage) ----
    train_max = cfg_ml["train_max_step"]
    test_min = cfg_ml["test_min_step"]
    train_df = df.filter(F.col("step") <= train_max)
    test_df = df.filter(F.col("step") >= test_min)
    logger.info("Train (steps 1-%d): %d rows", train_max, train_df.count())
    logger.info("Test  (steps %d-743): %d rows", test_min, test_df.count())

    # ---- Class weights ----
    fraud_w = float(cfg_ml["fraud_weight"])
    non_fraud_w = float(cfg_ml["non_fraud_weight"])
    train_df = train_df.withColumn(
        "classWeight",
        F.when(F.col("isFraud") == 1, F.lit(fraud_w))
        .otherwise(F.lit(non_fraud_w)),
    )
    # Test set also needs the column for Pipeline compat but weights don't affect eval
    test_df = test_df.withColumn("classWeight", F.lit(1.0))

    logger.info("Class weights — fraud: %.0f, non-fraud: %.0f", fraud_w, non_fraud_w)

    # ---- Feature columns ----
    # Numeric features (must exist and not leak the label)
    numeric_features = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "hour_of_day", "day",
        "errorBalanceOrig", "errorBalanceDest",
        "txn_count_orig_last_24", "txn_amount_orig_last_24",
        "txn_count_dest_last_24", "txn_amount_dest_last_24",
        "steps_since_last_txn_orig",
        "dest_distinct_senders", "dest_total_inbound",
        # NOT using: isFlaggedFraud, nameOrig, nameDest, dest_fraud_inbound_count
    ]

    # ---- Build ML Pipeline stages ----
    # 1. StringIndexer for 'type'
    type_indexer = StringIndexer(
        inputCol="type", outputCol="type_index", handleInvalid="keep"
    )
    # 2. OneHotEncoder for the indexed type
    type_encoder = OneHotEncoder(
        inputCol="type_index", outputCol="type_vec"
    )
    # 3. VectorAssembler — combine all features into one vector
    assembler_inputs = numeric_features + ["type_vec"]
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="skip",
    )

    # ---- Define models ----
    label_col = cfg_ml["label_col"]

    models = {
        "LogisticRegression": LogisticRegression(
            labelCol=label_col,
            featuresCol="features",
            weightCol="classWeight",
            maxIter=30,
            regParam=0.01,
        ),
        "RandomForest": RandomForestClassifier(
            labelCol=label_col,
            featuresCol="features",
            weightCol="classWeight",
            numTrees=40,
            maxDepth=8,
            seed=42,
        ),
        "GBTClassifier": GBTClassifier(
            labelCol=label_col,
            featuresCol="features",
            weightCol="classWeight",
            maxIter=20,
            maxDepth=6,
            seed=42,
        ),
    }

    # ---- Evaluators ----
    pr_auc_eval = BinaryClassificationEvaluator(
        labelCol=label_col, metricName="areaUnderPR"
    )
    roc_auc_eval = BinaryClassificationEvaluator(
        labelCol=label_col, metricName="areaUnderROC"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, metricName="f1"
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, metricName="weightedRecall"
    )

    # ---- Train and evaluate each model ----
    results = {}
    best_model_name = None
    best_pr_auc = -1
    best_pipeline_model = None
    best_predictions = None

    for name, model in models.items():
        logger.info("--- Training %s ---", name)
        pipeline = Pipeline(stages=[type_indexer, type_encoder, assembler, model])
        pipeline_model = pipeline.fit(train_df)
        preds = pipeline_model.transform(test_df)

        pr_auc = pr_auc_eval.evaluate(preds)
        roc_auc = roc_auc_eval.evaluate(preds)
        f1 = f1_eval.evaluate(preds)
        precision = precision_eval.evaluate(preds)
        recall = recall_eval.evaluate(preds)

        results[name] = {
            "PR-AUC": round(pr_auc, 4),
            "ROC-AUC": round(roc_auc, 4),
            "F1": round(f1, 4),
            "Precision (weighted)": round(precision, 4),
            "Recall (weighted)": round(recall, 4),
        }
        logger.info("%s — PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f",
                    name, pr_auc, roc_auc, f1)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_pipeline_model = pipeline_model
            best_predictions = preds

    logger.info("Best model: %s (PR-AUC=%.4f)", best_model_name, best_pr_auc)

    # ---- Confusion matrix ----
    cm = (
        best_predictions
        .groupBy("isFraud", "prediction")
        .count()
        .orderBy("isFraud", "prediction")
        .collect()
    )
    cm_dict = {f"actual_{r['isFraud']}_pred_{int(r['prediction'])}": r["count"]
               for r in cm}
    logger.info("Confusion matrix: %s", cm_dict)

    # ---- Feature importances (for tree models) ----
    feature_names = numeric_features + ["type_vec"]
    importance_dict = {}
    model_stage = best_pipeline_model.stages[-1]
    if hasattr(model_stage, "featureImportances"):
        importances = model_stage.featureImportances.toArray()
        # Map back to feature names (type_vec may have multiple entries)
        for i, imp in enumerate(importances):
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            importance_dict[fname] = round(float(imp), 6)
        sorted_imp = sorted(importance_dict.items(), key=lambda x: -x[1])
        logger.info("Feature importances (top 10):")
        for fname, imp in sorted_imp[:10]:
            logger.info("  %s: %.6f", fname, imp)

    # ---- PR curve points (collect for dashboard) ----
    # Get probability of fraud class and actual label
    pr_curve_points = []
    from pyspark.sql.types import DoubleType
    get_prob_udf = F.udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())

    if "probability" in best_predictions.columns:
        # Extract P(fraud) — probability vector index 1
        prob_fraud = best_predictions.select(
            F.col("isFraud").alias("label"),
            get_prob_udf(F.col("probability")).alias("prob_fraud"),
        )
        # Sample if test set is large to avoid Py4J driver memory pressure
        test_size = prob_fraud.count()
        if test_size > 100_000:
            sample_frac = min(1.0, 100_000.0 / test_size)
            collected = prob_fraud.sample(fraction=sample_frac, seed=42).collect()
        else:
            collected = prob_fraud.collect()
        from sklearn.metrics import precision_recall_curve, roc_curve
        import numpy as np
        labels = [r["label"] for r in collected]
        probs = [r["prob_fraud"] for r in collected]
        p, r_vals, thresholds = precision_recall_curve(labels, probs)
        pr_curve_points = [
            {"threshold": float(t), "precision": float(p_), "recall": float(r_)}
            for t, p_, r_ in zip(thresholds, p, r_vals)
        ]
        # Also compute ROC curve for dashboard
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        roc_curve_points = [
            {"fpr": float(f), "tpr": float(t)}
            for f, t in zip(fpr, tpr)
        ]
    else:
        roc_curve_points = []

    # ---- Save outputs ----
    models_dir = CONFIG["data"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    # Save pipeline model
    model_path = f"{models_dir}/best_pipeline_model"
    best_pipeline_model.write().overwrite().save(model_path)
    logger.info("Model saved to %s", model_path)

    # Save predictions
    preds_dir = CONFIG["data"]["predictions_dir"]
    pred_cols = ["_row_id", "step", "isFraud", "prediction", "probability"]
    existing_cols = [c for c in pred_cols if c in best_predictions.columns]
    preds_out = best_predictions.select(existing_cols)
    if "probability" in preds_out.columns:
        preds_out = preds_out.withColumn(
            "prob_fraud",
            get_prob_udf(F.col("probability")),
        )
    preds_out.write.mode("overwrite").parquet(preds_dir)
    logger.info("Predictions saved to %s (%d rows)", preds_dir, preds_out.count())

    # Save metrics.json
    metrics = {
        "best_model": best_model_name,
        "all_results": results,
        "confusion_matrix": cm_dict,
        "feature_importances": importance_dict,
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "train_max_step": train_max,
        "test_min_step": test_min,
    }
    metrics_path = f"{models_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save PR curve + ROC curve points
    curves_path = f"{models_dir}/curves.json"
    with open(curves_path, "w") as f:
        json.dump({
            "pr_curve": pr_curve_points[::max(1, len(pr_curve_points)//500)],
            "roc_curve": roc_curve_points[::max(1, len(roc_curve_points)//500)],
        }, f)
    logger.info("Curve points saved to %s", curves_path)

    spark.stop()
    logger.info("Training and evaluation complete ✓")


if __name__ == "__main__":
    train_and_evaluate()
