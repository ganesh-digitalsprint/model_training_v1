"""
delta_lake_reader.py
Reads the golden training dataset from Delta Lake using Ray Data.
"""
import ray
import ray.data
import pandas as pd
from processors.common.evaluation.schema_validator import validate
from deltalake import DeltaTable
from pathlib import Path
from urllib.parse import unquote

EXPECTED_COLUMNS = [
    "sku_id",
    "price",
    "competitor_price",
    "inventory_level",
    "price_diff_pct",
    "demand_qty",
]


def read_golden_dataset(delta_path: str, expected_columns=None):
    # Normalize path — fix spaces and slashes
    delta_path = str(Path(unquote(delta_path)).resolve())
    print(f"Reading Delta from: {delta_path}")

    cols = expected_columns or EXPECTED_COLUMNS

    # Read Delta → pandas (do ALL processing here, before Ray)
    dt  = DeltaTable(delta_path, storage_options={"allow_unsafe_rename": "true"})
    pdf = dt.to_pandas()
    print(f"✅ Loaded {len(pdf)} rows, {list(pdf.columns)}")

    # ── Column validation ──────────────────────────────────────────────────────
    missing = [c for c in cols if c not in pdf.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    pdf = pdf[cols]

    # ── Schema Validation ──────────────────────────────────────────────────────
    validate(pdf, cols, raise_on_error=True)

    # ── Basic Imputation (do on pandas, NOT Ray Dataset) ──────────────────────
    pdf["competitor_price"] = pdf["competitor_price"].fillna(pdf["price"])
    pdf["price_diff_pct"]   = pdf["price_diff_pct"].fillna(0)
    pdf["inventory_level"]  = pdf["inventory_level"].fillna(0)
    pdf = pdf.dropna(subset=cols)

    print(f"Delta Lake read complete. Rows: {len(pdf)}, Columns: {list(pdf.columns)}")

    # Convert to Ray dataset LAST, after all pandas operations are done
    ds = ray.data.from_pandas(pdf)
    return ds


def read_golden_dataset_from_csv(csv_path: str,
                                  expected_columns: list = None) -> pd.DataFrame:
    """
    Fallback: read golden dataset from a local CSV (dev / testing only).
    """
    cols = expected_columns or EXPECTED_COLUMNS
    print(f"[DEV FALLBACK] Reading golden dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    validate(df, cols, raise_on_error=True)
    df["competitor_price"] = df["competitor_price"].fillna(df["price"])
    df["price_diff_pct"]   = df["price_diff_pct"].fillna(0)
    df["inventory_level"]  = df["inventory_level"].fillna(0)
    df = df.dropna(subset=cols)
    return df