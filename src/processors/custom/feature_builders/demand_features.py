"""
demand_features.py
Builds demand signals (sales, views, cart) using Ray Data for parallel processing.
Replaces: src/features/demand_features.py
"""
import ray
import ray.data
import pandas as pd
import json


@ray.remote
def _aggregate_sales(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders.columns = orders.columns.str.lower()
    sales = (orders.groupby("sku_id")["quantity"]
             .sum().reset_index()
             .rename(columns={"quantity": "total_sales"}))
    return sales


@ray.remote
def _parse_ga4(ga4: list) -> tuple:
    views, carts = [], []
    for e in ga4:
        for item in e.get("items", []):
            if e["event_name"] == "view_item":
                views.append(item["item_id"])
            elif e["event_name"] == "add_to_cart":
                carts.append(item["item_id"])
    views_df = (pd.Series(views).value_counts()
                .reset_index().rename(columns={"index": "sku_id", 0: "views"}))
    carts_df = (pd.Series(carts).value_counts()
                .reset_index().rename(columns={"index": "sku_id", 0: "add_to_cart"}))
    return views_df, carts_df


def build_demand_features(orders: pd.DataFrame, ga4: list) -> pd.DataFrame:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    sales_future = _aggregate_sales.remote(orders)
    ga4_future   = _parse_ga4.remote(ga4)

    sales_df            = ray.get(sales_future)
    views_df, carts_df  = ray.get(ga4_future)

    demand = (sales_df
              .merge(views_df, on="sku_id", how="left")
              .merge(carts_df, on="sku_id", how="left")
              .fillna(0))

    for c in ["total_sales", "views", "add_to_cart"]:
        if demand[c].max() > 0:
            demand[c] = demand[c] / demand[c].max()

    demand["score"] = (0.5 * demand["total_sales"] +
                       0.3 * demand["add_to_cart"] +
                       0.2 * demand["views"])

    def bucket(x):
        if x > 0.7: return "HIGH"
        if x > 0.4: return "MED"
        return "LOW"

    demand["demand_bucket"] = demand["score"].apply(bucket)
    return demand[["sku_id", "demand_bucket"]]
