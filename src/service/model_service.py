"""
app/services/model_service.py
Handles loading and caching of the trained XGBoost model.

Design decision:
    ModelService is implemented as a singleton so the .pkl file is loaded
    once at startup and reused across all requests. Loading a 1.7MB XGBoost
    model on every request would be very slow (50–200ms each time).

The model predicts 'demand_qty' given:
    [price, competitor_price, inventory_level, price_diff_pct]

This aligns exactly with how train_executor.py trained the model
(feature_cols from price_prediction.yaml → training_data.feature_columns).
"""

from __future__ import annotations

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match how the model was trained
# See: config/jobs/price_prediction.yaml → training_data.feature_columns
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = ["price", "competitor_price", "inventory_level", "price_diff_pct"]
TARGET_COLUMN = "demand_qty"

# Feature importances from the trained model (for explanation generation)
# These were read directly from the pkl file:
# price: 0.037, competitor_price: 0.175, inventory_level: 0.443, price_diff_pct: 0.345
FEATURE_IMPORTANCES = {
    "price": 0.037,
    "competitor_price": 0.175,
    "inventory_level": 0.443,
    "price_diff_pct": 0.345,
}


class ModelService:
    """
    Singleton wrapper around the trained XGBoost model.
    Call ModelService.get_instance() to get the loaded model wrapper.
    """

    _instance: Optional["ModelService"] = None

    def __init__(self, model_path: str):
        logger.info("Loading model from: %s", model_path)
        self.model = joblib.load(model_path)
        self.model_version = self._extract_version_from_path(model_path)
        self.model_path = model_path
        logger.info(
            "Model loaded — version=%s, features=%s",
            self.model_version, FEATURE_COLUMNS,
        )

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Return the singleton instance, loading the model if needed."""
        if cls._instance is None:
            model_path = cls._resolve_model_path()
            cls._instance = cls(model_path)
        return cls._instance

    @classmethod
    def _resolve_model_path(cls) -> str:
        """
        Resolve the model file path in priority order:
        1. MODEL_PATH environment variable (recommended for production)
        2. Default local path used by train_executor.py when saving .pkl
        """
        # 1. Environment variable override — best for Docker / K8s
        env_path = os.getenv("MODEL_PATH")
        if env_path and os.path.exists(env_path):
            logger.info("Using model path from MODEL_PATH env: %s", env_path)
            return env_path

        # 2. Default path where train_executor.py saves the .pkl
        #    matches: DEFAULT_MODEL_LOCAL_PATH / {model_name}_{version}.pkl
        default_paths = [
            # Path used in this project based on actual saved pkl
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "models", "xgboost_v1.0.pkl",
            ),
            # Path from constants.py DEFAULT_MODEL_LOCAL_PATH (Windows path won't work on Linux)
            "src/data/raw/ml/models/xgboost_v1.0.pkl",
        ]

        for p in default_paths:
            p = os.path.normpath(p)
            if os.path.exists(p):
                logger.info("Using default model path: %s", p)
                return p

        raise FileNotFoundError(
            "Model file not found. Set the MODEL_PATH environment variable to "
            "point to your xgboost_v1.0.pkl file.\n"
            "Example: export MODEL_PATH=/path/to/xgboost_v1.0.pkl"
        )

    @staticmethod
    def _extract_version_from_path(path: str) -> str:
        """Extract version from filename like xgboost_v1.0.pkl → v1.0"""
        basename = os.path.basename(path)
        name_no_ext = os.path.splitext(basename)[0]  # xgboost_v1.0
        parts = name_no_ext.split("_")
        # Last part is the version if it starts with 'v'
        for part in reversed(parts):
            if part.startswith("v"):
                return part
        return "unknown"

    def predict_demand(
        self,
        price: float,
        competitor_price: float,
        inventory_level: float,
        price_diff_pct: float,
    ) -> float:
        """
        Predict demand quantity for a given set of features.

        This exactly mirrors predict_demand() in inference.py but without
        Ray Serve overhead — suitable for direct API use.
        """
        # Build a single-row DataFrame matching training feature order
        X = pd.DataFrame(
            [[price, competitor_price, inventory_level, price_diff_pct]],
            columns=FEATURE_COLUMNS,
        )
        prediction = self.model.predict(X)[0]
        # Demand cannot be negative
        return float(max(prediction, 0.0))

    def predict_demand_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict demand for a batch DataFrame.
        df must have columns matching FEATURE_COLUMNS.
        """
        predictions = self.model.predict(df[FEATURE_COLUMNS])
        return np.maximum(predictions, 0.0)