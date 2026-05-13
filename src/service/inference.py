"""
ml/inference.py
Loads the .pkl model and runs demand prediction.
No rule engine. No champion/challenger logic.
"""
from __future__ import annotations

import os
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config — sourced entirely from env vars
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "models")


def load_model(model_name: str, version: str) -> object:
    """
    Load a .pkl model from  {MODEL_DIR}/{model_name}_{version}.pkl
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    pkl_path = os.path.join(MODEL_DIR, f"{model_name}_{version}.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"Model file not found: '{pkl_path}'. "
            f"Check MODEL_DIR (currently '{MODEL_DIR}') and that training has completed."
        )
    model = joblib.load(pkl_path)
    logger.info("Model loaded: %s", pkl_path)
    return model


def predict_demand(
    model,
    price: float,
    competitor_price: float,
    inventory_level: float,
    price_diff_pct: float,
) -> float:
    """
    Run one demand prediction from the loaded model.
    Builds a single-row DataFrame in the feature order the model was trained on.
    """
    feature_cols = os.getenv(
        "FEATURE_COLUMNS",
        "price,competitor_price,total_inventory,price_diff_pct",
    ).split(",")

    row = {
        "price": price,
        "competitor_price": competitor_price,
        "total_inventory": inventory_level,
        "price_diff_pct": price_diff_pct,
    }

    # Build the row using only the columns the model expects
    X = pd.DataFrame([[row.get(c, 0) for c in feature_cols]], columns=feature_cols)
    pred = model.predict(X)
    return float(np.ravel(pred)[0])