"""
competitor_features.py
Builds competitor pricing features using Ray for parallel item processing.
Replaces: src/features/competitor_features.py
"""
import ray
import pandas as pd
import json


@ray.remote
def _parse_competitor_item(item: dict) -> dict:
    name, summary = item.get("name"), item.get("summary", {})
    min_price, min_site, comp_dict = None, None, {}
    for site, data in summary.items():
        if data and data.get("price"):
            try:
                price = float(data["price"])
                comp_dict[site] = str(round(price, 2))
                if min_price is None or price < min_price:
                    min_price, min_site = price, site
            except Exception:
                pass
    return {"display_name": name, "competitor_price": min_price,
            "competitor_name": min_site,
            "all_competitors_json": json.dumps(comp_dict)}


def build_competitor_features(price_df: pd.DataFrame,
                               sku_df: pd.DataFrame,
                               comp_data) -> pd.DataFrame:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    price_df = price_df.copy(); price_df.columns = price_df.columns.str.strip().str.lower()
    sku_df   = sku_df.copy();   sku_df.columns   = sku_df.columns.str.strip().str.lower()

    items = comp_data.get("results", comp_data) if isinstance(comp_data, dict) else comp_data
    rows  = ray.get([_parse_competitor_item.remote(item) for item in items])

    comp_df = pd.DataFrame(rows)
    comp_df = comp_df.merge(sku_df[["sku_id", "display_name"]], on="display_name", how="left")
    comp_df = comp_df.merge(price_df[["sku_id", "list_price"]], on="sku_id", how="left")

    def bucket(row):
        if pd.isna(row["competitor_price"]): return "NO_COMPETITOR"
        if row["competitor_price"] < row["list_price"]: return "COMPETITOR_CHEAPER"
        if row["competitor_price"] > row["list_price"]: return "WE_CHEAPER"
        return "SAME"

    comp_df["competitor_bucket"] = comp_df.apply(bucket, axis=1)
    comp_df["price_diff_pct"] = ((comp_df["list_price"] - comp_df["competitor_price"])
                                  / comp_df["competitor_price"] * 100).round(2)

    return comp_df[["sku_id", "competitor_price", "competitor_name",
                    "competitor_bucket", "price_diff_pct", "all_competitors_json"]]
