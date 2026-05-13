"""
load_raw_data.py
Orchestrates all raw source loaders using Ray Data for parallel ingestion.
Replaces: src/ingestion/load_raw_data.py
"""
import ray
import ray.data
import pandas as pd
import json
import os


@ray.remote
def _load_sku(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    for c in ["sku_id", "display_name"]:
        if c not in df.columns:
            raise ValueError(f"{c} missing in sku file")
    return df[["sku_id", "display_name"]]


@ray.remote
def _load_price(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    for alt in ["list_price", "price", "selling_price", "mrp"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "list_price"})
            break
    return df[["sku_id", "list_price"]]


@ray.remote
def _load_inventory(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    stock_col = "stock_level" if "stock_level" in df.columns else "quantity"
    df = df.rename(columns={"catalog_ref_id": "sku_id", stock_col: "stock"})
    return df[["sku_id", "stock"]]


@ray.remote
def _load_orders(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    qty_col = "quantity" if "quantity" in df.columns else "qty"
    df = df.rename(columns={"catalog_ref_id": "sku_id", qty_col: "quantity"})
    return df[["sku_id", "quantity"]]


@ray.remote
def _load_ga4(file_path: str):
    with open(file_path) as f:
        return json.load(f)


@ray.remote
def _load_competitor(file_path: str):
    with open(file_path) as f:
        return json.load(f)


def load_data(data_path: str) -> tuple:
    """
    Parallel raw data ingestion using Ray remote tasks.
    All loaders run concurrently — significant speedup on large files.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Launch all loads in parallel
    futures = {
        "sku":        _load_sku.remote(os.path.join(data_path, "sku.csv")),
        "price":      _load_price.remote(os.path.join(data_path, "price.csv")),
        "inventory":  _load_inventory.remote(os.path.join(data_path, "inventory.xlsx")),
        "orders":     _load_orders.remote(os.path.join(data_path, "purchased_order.xlsx")),
        "ga4":        _load_ga4.remote(os.path.join(data_path, "ga4_events.json")),
        "competitor": _load_competitor.remote(os.path.join(data_path, "competitor_pricing.json")),
    }

    results = {k: ray.get(v) for k, v in futures.items()}
    print("All raw sources loaded in parallel via Ray")
    return (results["sku"], results["price"], results["inventory"],
            results["orders"], results["ga4"], results["competitor"])
