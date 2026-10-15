"""
test_pipeline.py — Unit and integration tests for FraudShield pipeline logic.
Run with: .venv/bin/python -m pytest tests/
"""

import json
import os
import pytest
from pathlib import Path

from src.common.config import CONFIG, PROJECT_ROOT
from src.ml.evaluate import load_metrics, load_curves


def test_config_paths():
    """Verify config paths are properly resolved."""
    assert os.path.exists(CONFIG["data"]["raw_dir"])
    assert "data/bronze" in CONFIG["data"]["bronze_dir"]
    assert "data/silver" in CONFIG["data"]["silver_dir"]
    assert "data/gold" in CONFIG["data"]["gold_dir"]


def test_gold_artifacts_exist():
    """Verify that all Gold-layer tables were generated."""
    gold_dir = Path(CONFIG["data"]["gold_dir"])
    assert (gold_dir / "gold_kpis").exists()
    assert (gold_dir / "gold_account_risk").exists()
    assert (gold_dir / "gold_daily_summary").exists()
    assert (gold_dir / "gold_features").exists()


def test_ml_artifacts_exist():
    """Verify that model, metrics, curves, and predictions exist."""
    models_dir = Path(CONFIG["data"]["models_dir"])
    preds_dir = Path(CONFIG["data"]["predictions_dir"])

    assert (models_dir / "best_pipeline_model").exists()
    assert (models_dir / "metrics.json").exists()
    assert (models_dir / "curves.json").exists()
    assert preds_dir.exists()


def test_metrics_content():
    """Verify metrics contain expected keys, high PR-AUC, and all models."""
    metrics = load_metrics()
    assert "best_model" in metrics
    assert "all_results" in metrics
    assert "RandomForest" in metrics["all_results"]
    assert "GBTClassifier" in metrics["all_results"]
    assert "LogisticRegression" in metrics["all_results"]

    # Check primary evaluation metric: PR-AUC > 0.90
    best_name = metrics["best_model"]
    best_pr_auc = metrics["all_results"][best_name]["PR-AUC"]
    assert best_pr_auc >= 0.90, f"Expected PR-AUC >= 0.90, got {best_pr_auc}"


def test_curves_content():
    """Verify curves.json contains valid PR curve points."""
    curves = load_curves()
    assert "pr_curve" in curves
    assert len(curves["pr_curve"]) > 0
    first_pt = curves["pr_curve"][0]
    assert "precision" in first_pt
    assert "recall" in first_pt


def test_balance_error_logic():
    """Test the mathematical identity of balance error derivation."""
    # Legitimate transaction: new = old - amount
    old_orig, amount, new_orig = 1000.0, 200.0, 800.0
    error_orig = new_orig + amount - old_orig
    assert error_orig == 0.0

    # Fraudulent balance wipe: new balance is 0 despite remaining balance
    old_orig, amount, new_orig = 5000.0, 200.0, 0.0
    error_orig = new_orig + amount - old_orig
    assert error_orig == -4800.0  # Clear discrepancy!
