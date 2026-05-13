"""
inventory_features.py
Builds inventory-level bucketing features.
Replaces: src/features/inventory_features.py
"""
import ray
import pandas as pd


@ray.remote
def _aggregate_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    inv = inv.copy()
    inv.columns = inv.columns.str.lower()
    stock_col = "stock_level" if "stock_level" in inv.columns else "stock"
    agg = (inv.groupby("sku_id")[stock_col]
           .sum().reset_index()
           .rename(columns={stock_col: "total_inventory"}))

    def bucket(q):
        if q > 1000: return "VERY_HIGH"
        if q > 400:  return "HIGH"
        if q > 100:  return "MED"
        return "LOW"

    agg["inventory_bucket"] = agg["total_inventory"].apply(bucket)
    return agg


def build_inventory_features(inv: pd.DataFrame) -> pd.DataFrame:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    return ray.get(_aggregate_inventory.remote(inv))
