"""
pipeline/pricing_executor.py
Core pricing pipeline executor.

Responsibilities
----------------
1. Load the golden dataset from Delta Lake          (no raw data access)
2. Load the trained model binary from Delta Lake    (via delta_model_reader)
3. Inject the model into the inference layer        (patches inference.py cache)
4. Dispatch to the correct pricing engine           (rule | ai | hybrid)
5. Run all simulators and log to MLflow dashboard
7. Persist output

This module is called exclusively by pricing_launcher.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from processors.custom.ingestion.delta_lake_reader import (
    read_golden_dataset, read_golden_dataset_from_csv)
import pandas as pd
from processors.custom.ingestion.delta_model_reader import (
    load_model_from_delta, load_champion_challenger_from_delta)
from processors.custom.services.pricing_service import run_pricing    # type: ignore
from utils.constants import (
    DELTA_LAKE, GOLDEN_DATASET_PATH, CFG_GOLDEN_CSV_FALLBACK,
    EXPECTED_COLUMNS, CFG_FEATURES, CFG_HYBRID,
    CFG_MIN_R2_FOR_AI, CFG_MAX_MAPE_FOR_AI,
    MODE_AI, MODE_HYBRID,
    XGBOOST, VERSION,
    DEFAULT_MIN_R2_FOR_AI, DEFAULT_MAX_MAPE_FOR_AI, DEFAULT_ENVIRONMENT,
    RESULT_STATUS, RESULT_MODE, RESULT_MODEL_NAME, RESULT_MODEL_VERSION,
    RESULT_ENVIRONMENT, RESULT_ROWS_PROCESSED, RESULT_ELAPSED_SECONDS,
    RESULT_STATUS_SUCCESS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Delta Lake data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_golden_dataset(config: dict, golden_fallback: str | None) -> pd.DataFrame:
    """
    Load the golden pricing dataset — same logic as the training pipeline.

    Uses read_golden_dataset (Delta Lake) or read_golden_dataset_from_csv
    (dev fallback) from delta_lake_reader.py — no duplicated logic here.
    """
    delta_cfg     = config.get(DELTA_LAKE, {})
    delta_path    = delta_cfg.get(GOLDEN_DATASET_PATH)
    fallback_path = golden_fallback or delta_cfg.get(CFG_GOLDEN_CSV_FALLBACK)
    expected_cols = delta_cfg.get(EXPECTED_COLUMNS) or None  # None → reader uses its default

    if delta_path:
        print(f"[Executor] Loading golden dataset from Delta Lake: {delta_path}")
        ds = read_golden_dataset(delta_path, expected_columns=expected_cols)
        df = ds.to_pandas()                 # read_golden_dataset returns a Ray Dataset
        logger.info("Golden dataset loaded from Delta: %s  rows=%d", delta_path, len(df))
        return df

    if fallback_path:
        print(f"[Executor] ⚠ Delta path not set — using CSV fallback: {fallback_path}")
        df = read_golden_dataset_from_csv(fallback_path, expected_columns=expected_cols)
        logger.warning("Golden dataset loaded from CSV fallback: %s", fallback_path)
        return df

    raise ValueError(
        "No data source available. Set [delta_lake.golden_dataset_path] "
        "or [delta_lake.golden_csv_fallback] in pricing_job.yaml."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model injection
# ─────────────────────────────────────────────────────────────────────────────

def _inject_model_into_inference(model_name: str, model_version: str,
                                  environment: str, config: dict) -> object:
    """
    Load the model from Delta Lake and inject it into the inference module cache.

    This means inference.load_model() will return the Delta-loaded model without
    trying MLflow registry or local .pkl paths.  A clean seam — the pricing
    engines don't need to know where the model came from.
    """
    import processors.custom.inference as inference_module   # type: ignore

    model = load_model_from_delta(
        model_name  = model_name,
        version     = model_version,
        environment = environment,
        config      = config,
    )

    # Patch the module-level cache so load_model() short-circuits immediately
    inference_module._LOCAL_MODEL_CACHE = model
    logger.info("Model injected into inference cache: %s@%s", model_name, model_version)
    print(f"[Executor] Model injected into inference layer: {model_name}@{model_version}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main executor
# ─────────────────────────────────────────────────────────────────────────────

def run_pricing_job(
    config:          dict,
    mode:            str   = MODE_HYBRID,
    model_name:      str   = XGBOOST,
    model_version:   str   = VERSION,
    environment:     str   = DEFAULT_ENVIRONMENT,
    tolerance:       float = 0.05,
    golden_fallback: str | None = None,
    orders_fallback: str | None = None,
) -> dict:
    """
    Full pricing pipeline execution.

    Parameters
    ----------
    config          : loaded pricing_job.yaml config dict
    mode            : "rule" | "ai" | "hybrid"
    model_name      : model identifier in Delta registry
    model_version   : specific version — None = latest active
    environment     : must match the env used during training
    tolerance       : minimum revenue-improvement threshold for AI acceptance
    golden_fallback : dev-only CSV path when Delta Lake is unreachable
    orders_fallback : dev-only CSV path when Delta Lake is unreachable

    Returns
    -------
    dict with run summary metrics
    """
    run_start = datetime.now(timezone.utc)

    # ── 1. Load data from Delta Lake (golden dataset only, no raw data) ───────
    print("\n[Executor] Step 1/5 — Loading datasets from Delta Lake...")

    master_df = _load_golden_dataset(config, golden_fallback)
    print(f"[Debug] master_df columns: {list(master_df.columns)}")
    print(f"[Debug] master_df shape  : {master_df.shape}")
    print(master_df.head(2))
    # ── 2. Load model from Delta Lake and inject into inference layer ─────────
    print("[Executor] Step 2/5 — Loading model from Delta Lake model registry...")
    if mode in (MODE_AI, MODE_HYBRID):
        _inject_model_into_inference(model_name, model_version, environment, config)

    # ── 3. Retrieve feature flags and guard-rails from config ─────────────────
    print(f"[Executor] Step 3/5 — Dispatching to '{mode}' pricing engine...")
    feature_flags = config.get(CFG_FEATURES, {})
    hybrid_cfg    = config.get(CFG_HYBRID, {})

    # Merge hybrid thresholds into config so hybrid_engine can read them
    config.setdefault(CFG_HYBRID, {})
    config[CFG_HYBRID][CFG_MIN_R2_FOR_AI]   = hybrid_cfg.get(CFG_MIN_R2_FOR_AI,   DEFAULT_MIN_R2_FOR_AI)
    config[CFG_HYBRID][CFG_MAX_MAPE_FOR_AI] = hybrid_cfg.get(CFG_MAX_MAPE_FOR_AI, DEFAULT_MAX_MAPE_FOR_AI)

    # ── 4. Run pricing service ────────────────────────────────────────────────
    run_pricing(
        master_df     = master_df,
        mode          = mode,
        model_name    = model_name,
        model_version = model_version,
        tolerance     = tolerance,
    )

    # ── 5. Build result summary ───────────────────────────────────────────────
    print("[Executor] Step 5/5 — Pricing job complete.")
    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
    result  = {
        RESULT_STATUS:          RESULT_STATUS_SUCCESS,
        RESULT_MODE:            mode,
        RESULT_MODEL_NAME:      model_name,
        RESULT_MODEL_VERSION:   model_version,
        RESULT_ENVIRONMENT:     environment,
        RESULT_ROWS_PROCESSED:  len(master_df),
        RESULT_ELAPSED_SECONDS: round(elapsed, 2),
    }
    logger.info("Pricing job completed: %s", result)
    return result