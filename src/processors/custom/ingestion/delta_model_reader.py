"""
delta_model_reader.py
Loads a trained model binary blob from the Delta Lake model registry table
and deserialises it back to a sklearn/xgboost model object.

This is the ONLY place in the pricing pipeline that touches model files.
All pricing engines (ai_engine, hybrid_engine, champion_challenger_engine)
import load_model_from_delta() instead of calling joblib.load() directly.

Public API
----------
    load_model_from_delta(model_name, version, environment, config)
        → fitted model object

    load_champion_challenger_from_delta(model_name, environment, config)
        → (champion_model, challenger_model)   or   (None, None)
"""

from __future__ import annotations

import io
import json
import logging
from functools import lru_cache

import joblib
import pandas as pd
from deltalake import DeltaTable

logger = logging.getLogger(__name__)

# Module-level cache: keyed by (model_name, version, environment)
_MODEL_CACHE: dict[tuple, object] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_registry_df(table_path: str) -> pd.DataFrame:
    """Read the Delta model-registry table as a DataFrame."""
    try:
        dt = DeltaTable(table_path)
        return dt.to_pandas()
    except Exception as exc:
        raise FileNotFoundError(
            f"Delta model registry not found at '{table_path}'. "
            f"Run training first to populate it.\nOriginal error: {exc}"
        ) from exc


def _deserialise(model_binary: bytes):
    """Joblib-deserialise raw bytes back to a model object."""
    buf = io.BytesIO(model_binary)
    return joblib.load(buf)


def _fetch_row(
    df: pd.DataFrame,
    model_name: str,
    version: str | None,
    environment: str,
) -> pd.Series:
    """
    Find the best matching row.
    Priority:
        1. Exact (model_name, version, environment, is_active=True)
        2. Latest active (model_name, environment, is_active=True)
        3. Latest row regardless of is_active
    """
    mask_base = (df["model_name"] == model_name) & (df["environment"] == environment)

    # 1. Exact version + active
    if version:
        exact = df[mask_base & (df["model_version"] == version) & df["is_active"]]
        if not exact.empty:
            return exact.iloc[-1]

    # 2. Latest active
    active = df[mask_base & df["is_active"]]
    if not active.empty:
        return active.iloc[-1]

    # 3. Any row (fallback — logs a warning)
    fallback = df[mask_base]
    if not fallback.empty:
        logger.warning(
            "No active model found for model=%s env=%s — using latest inactive row",
            model_name, environment,
        )
        return fallback.iloc[-1]

    raise ValueError(
        f"No model found in registry for model_name='{model_name}' "
        f"version='{version}' environment='{environment}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_delta(
    model_name:  str,
    version:     str  | None = None,
    environment: str         = "dev",
    config:      dict | None = None,
    use_cache:   bool        = True,
) -> object:
    """
    Load a model from the Delta Lake model registry.

    Parameters
    ----------
    model_name  : e.g. "xgboost"
    version     : e.g. "v1.0"  — if None, loads the latest active version
    environment : "dev" | "qa" | "prod"
    config      : pricing config dict (falls back to env-defaults)
    use_cache   : if True, returns cached model on repeated calls

    Returns
    -------
    Fitted sklearn / xgboost model object
    """
    cache_key = (model_name, version, environment)
    if use_cache and cache_key in _MODEL_CACHE:
        logger.debug("Model cache hit: %s", cache_key)
        return _MODEL_CACHE[cache_key]

    table_path = _resolve_table_path(config)
    df         = _get_registry_df(table_path)
    row        = _fetch_row(df, model_name, version, environment)
    model      = _deserialise(row["model_binary"])

    # Surface metadata so callers can log / audit
    resolved_version = row["model_version"]
    feature_cols     = json.loads(row.get("feature_columns", "[]"))
    created_at       = row.get("created_at", "unknown")

    logger.info(
        "Loaded model from Delta: model=%s version=%s env=%s features=%s created=%s",
        model_name, resolved_version, environment, feature_cols, created_at,
    )
    print(
        f"[ModelReader] ✓ Loaded model={model_name} | version={resolved_version} "
        f"| env={environment} | features={feature_cols}"
    )

    if use_cache:
        _MODEL_CACHE[cache_key] = model

    return model


def reload_model_cache() -> None:
    """Clear the in-process model cache so the next call re-reads from Delta."""
    _MODEL_CACHE.clear()
    logger.info("Delta model cache cleared")
    print("[ModelReader] Model cache cleared")


def load_champion_challenger_from_delta(
    model_name:  str,
    environment: str         = "dev",
    config:      dict | None = None,
) -> tuple:
    """
    Load the two most recent active versions of a model for
    champion-challenger evaluation.

    Returns
    -------
    (champion_model, challenger_model)
        champion   = the latest active version
        challenger = the second-latest active version
        Either can be None if fewer than 2 versions exist.
    """
    table_path = _resolve_table_path(config)
    df         = _get_registry_df(table_path)

    mask   = (df["model_name"] == model_name) & (df["environment"] == environment)
    active = df[mask].sort_values("created_at", ascending=False)

    if active.empty:
        logger.warning("No models found for champion-challenger: model=%s env=%s",
                       model_name, environment)
        return None, None

    champion   = _deserialise(active.iloc[0]["model_binary"])
    challenger = _deserialise(active.iloc[1]["model_binary"]) if len(active) >= 2 else None

    champ_ver = active.iloc[0]["model_version"]
    chall_ver = active.iloc[1]["model_version"] if challenger else "N/A"

    print(
        f"[ModelReader] Champion={model_name}@{champ_ver}  "
        f"Challenger={model_name}@{chall_ver}  env={environment}"
    )
    return champion, challenger


def get_model_metadata(
    model_name:  str,
    environment: str         = "dev",
    config:      dict | None = None,
) -> list[dict]:
    """
    Return metadata (without binary) for all versions of a model.
    Useful for audit / dashboard.
    """
    table_path = _resolve_table_path(config)
    df         = _get_registry_df(table_path)
    mask       = (df["model_name"] == model_name) & (df["environment"] == environment)
    rows       = df[mask].drop(columns=["model_binary"], errors="ignore")
    return rows.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Config helper (kept internal)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_table_path(config: dict | None) -> str:
    """Extract Delta model-registry path from config or return default."""
    if config:
        path = config.get("delta_lake", {}).get("model_registry_path")
        if path:
            return path
    # Lazy import to avoid circular dependency at module load time
    try:
        from bootstrap.train_config_loader import get_active_config
        cfg  = get_active_config()
        path = cfg.get("delta_lake", {}).get("model_registry_path")
        if path:
            return path
    except Exception:
        pass
    return "D:/delta_tables/model_registry"     # absolute fallback