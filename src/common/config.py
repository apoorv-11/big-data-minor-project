"""
config.py — Central configuration loader.
Reads config/config.yaml and resolves all paths relative to the project root.
"""

import os
import yaml
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing config/config.yaml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Cannot find project root (config/config.yaml not found)")


# Resolved once at import time
PROJECT_ROOT = _find_project_root()


def load_config() -> dict:
    """Load and return the full config dict with paths resolved to absolute."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve all data paths to absolute
    for key, rel_path in cfg.get("data", {}).items():
        cfg["data"][key] = str(PROJECT_ROOT / rel_path)

    # Resolve streaming checkpoint
    if "streaming" in cfg and "checkpoint_location" in cfg["streaming"]:
        cfg["streaming"]["checkpoint_location"] = str(
            PROJECT_ROOT / cfg["streaming"]["checkpoint_location"]
        )

    cfg["project"]["root"] = str(PROJECT_ROOT)
    return cfg


# Module-level singleton so other modules can just import it
CONFIG = load_config()
