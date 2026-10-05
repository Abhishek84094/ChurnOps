"""
ChurnOps — Central Configuration Loader
Reads params.yaml once; all src modules import from here.
"""

import os
from pathlib import Path
import yaml

# Project root is three levels up from this file: src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH  = PROJECT_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Load and return the full params.yaml as a nested dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Singleton — loaded once on import
PARAMS = load_params()


def get_path(key: str) -> Path:
    """Return an absolute Path for a key in params.paths."""
    rel = PARAMS["paths"][key]
    return PROJECT_ROOT / rel


# Convenience aliases
RANDOM_STATE      = PARAMS["random_state"]
TARGET_COL        = PARAMS["dataset"]["target_column"]
ID_COL            = PARAMS["dataset"]["id_column"]
TEST_SIZE         = PARAMS["split"]["test_size"]
CV_SPLITS         = PARAMS["cv"]["n_splits"]
IMBALANCE_STRAT   = PARAMS["imbalance"]["strategy"]
MLFLOW_URI        = PARAMS["mlflow"]["tracking_uri"]
MLFLOW_EXPERIMENT = PARAMS["mlflow"]["experiment_name"]
