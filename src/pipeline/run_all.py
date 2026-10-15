"""
run_all.py — Run the full FraudShield batch pipeline end-to-end.

Stages: Bronze → Silver → Features → Gold → Train → Predict

Usage:  python -m src.pipeline.run_all [--sample 0.05]
"""

import argparse
import time

from src.common.logger import get_logger

logger = get_logger(__name__)


def run_all(sample: float | None = None):
    start = time.time()

    logger.info("=" * 60)
    logger.info("  FraudShield — Full Pipeline Run")
    logger.info("=" * 60)

    # Step 3: Bronze
    logger.info("\n>>> STEP 3: Bronze Ingestion <<<")
    from src.ingestion.bronze_batch import ingest_bronze
    ingest_bronze(sample_fraction=sample)

    # Step 4: Silver
    logger.info("\n>>> STEP 4: Silver Cleaning <<<")
    from src.transform.silver_clean import run_silver
    run_silver()

    # Step 5: Feature Engineering
    logger.info("\n>>> STEP 5: Feature Engineering <<<")
    from src.transform.features_pyspark import run_features
    run_features()

    # Step 6: Gold
    logger.info("\n>>> STEP 6: Gold Build <<<")
    from src.transform.gold_build import build_gold
    build_gold()

    # Step 7-8: Train & Evaluate
    logger.info("\n>>> STEP 7-8: Train & Evaluate <<<")
    from src.ml.train_model import train_and_evaluate
    train_and_evaluate()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f seconds (%.1f minutes)",
                elapsed, elapsed / 60)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run full FraudShield pipeline")
    parser.add_argument("--sample", type=float, default=None)
    args = parser.parse_args()
    run_all(sample=args.sample)


if __name__ == "__main__":
    main()
