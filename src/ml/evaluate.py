"""
evaluate.py — Utility functions for model evaluation.
Shared by train_model.py and the dashboard.
"""

import json
from pathlib import Path

from src.common.config import CONFIG
from src.common.logger import get_logger

logger = get_logger(__name__)


def load_metrics() -> dict:
    """Load metrics.json from the models directory."""
    metrics_path = Path(CONFIG["data"]["models_dir"]) / "metrics.json"
    if not metrics_path.exists():
        logger.warning("metrics.json not found at %s", metrics_path)
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def load_curves() -> dict:
    """Load PR/ROC curve points from curves.json."""
    curves_path = Path(CONFIG["data"]["models_dir"]) / "curves.json"
    if not curves_path.exists():
        logger.warning("curves.json not found at %s", curves_path)
        return {"pr_curve": [], "roc_curve": []}
    with open(curves_path) as f:
        return json.load(f)
