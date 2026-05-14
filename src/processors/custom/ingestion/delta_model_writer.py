"""
delta_model_writer.py
Serialises a trained sklearn/xgboost model to binary and writes it as a
row in a Delta Lake table.

Schema written
--------------
model_name      : string
model_version   : string
environment     : string  (dev | qa | prod)
model_binary    : binary  (raw joblib bytes)
feature_columns : string  (JSON list)
created_at      : timestamp
is_active       : boolean

Usage (called from train_executor.py after model is fitted)
-----------------------------------------------------------
    from delta_model_writer import write_model_to_delta
    write_model_to_delta(model, "xgboost", "v1.0", "dev", config)
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
# from bootstrap.train_config_loader import get_active_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_model(model) -> bytes:
    """Dump any sklearn-compatible model to raw bytes via joblib."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.read()


def _build_record(model, model_name: str, model_version: str,
                  environment: str, feature_columns: list[str]) -> dict:
    return {
        "model_name":      model_name,
        "model_version":   model_version,
        "environment":     environment,
        "model_binary":    _serialise_model(model),
        "feature_columns": json.dumps(feature_columns),
        "created_at":      datetime.now(timezone.utc),
        "is_active":       True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def write_model_to_delta(
    model,
    model_name:    str,
    model_version: str,
    environment:   str  = "dev",
    config:        dict = None,
) -> None:
    """
    Persist *model* as a binary blob row in the Delta Lake model registry.

    Steps
    -----
    1. Mark every previous row for (model_name, environment) as inactive
       by overwriting only those rows (merge / overwrite-matching-predicate).
    2. Append the new active row.

    Parameters
    ----------
    model         : fitted sklearn / xgboost model
    model_name    : e.g. "xgboost"
    model_version : e.g. "v1.0"
    environment   : "dev" | "qa" | "prod"
    config        : loaded training config dict (falls back to get_active_config)
    """
    if config is None:
        config = get_active_config()

    delta_cfg = config.get("delta_lake", {})
    table_path = delta_cfg.get(
        "model_registry_path",
        "D:/delta_tables/model_registry"        # default — override in YAML
    )

    feature_columns: list[str] = (
        config.get("training_data", {}).get("feature_columns")
        or ["price", "competitor_price", "inventory_level", "price_diff_pct"]
    )

    record = _build_record(model, model_name, model_version,
                           environment, feature_columns)
    new_df = pd.DataFrame([record])

    # ── Define explicit PyArrow schema (required for binary column) ───────────
    arrow_schema = pa.schema([
        pa.field("model_name",      pa.string()),
        pa.field("model_version",   pa.string()),
        pa.field("environment",     pa.string()),
        pa.field("model_binary",    pa.large_binary()),
        pa.field("feature_columns", pa.string()),
        pa.field("created_at",      pa.timestamp("us", tz="UTC")),
        pa.field("is_active",       pa.bool_()),
    ])

    def _to_arrow(df: pd.DataFrame) -> pa.Table:
        """Convert DataFrame to PyArrow Table with explicit schema."""
        return pa.Table.from_pandas(df, schema=arrow_schema, safe=False)

    # ── Step 1: deactivate old versions ──────────────────────────────────────
    try:
        dt = DeltaTable(table_path)
        existing = dt.to_pandas()
        if not existing.empty:
            mask = (
                (existing["model_name"]  == model_name) &
                (existing["environment"] == environment)
            )
            existing.loc[mask, "is_active"] = False
            combined = pd.concat([existing, new_df], ignore_index=True)
            write_deltalake(table_path, _to_arrow(combined), mode="overwrite")
            logger.info(
                "Deactivated %d previous version(s) for model=%s env=%s",
                mask.sum(), model_name, environment,
            )
            print(f"[ModelWriter] Previous versions deactivated: {mask.sum()}")
            return

    except Exception as exc:
        logger.info("Delta table not found, creating new: %s (%s)", table_path, exc)

    # ── Step 2: append (or create) ───────────────────────────────────────────
    write_deltalake(table_path, _to_arrow(new_df), mode="append")
    logger.info(
        "Model written to Delta Lake: path=%s model=%s version=%s env=%s",
        table_path, model_name, model_version, environment,
    )
    print(
        f"[ModelWriter] ✓ Saved → {table_path} "
        f"| model={model_name} | version={model_version} | env={environment}"
    )