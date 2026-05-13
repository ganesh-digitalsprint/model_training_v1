"""
master_dataset_builder.py
Builds the master pricing dataset from raw sources.
Replaces: src/app/main.py :: build_master_dataset()
"""
import pandas as pd
from processors.custom.feature_builders.demand_features     import build_demand_features
from processors.custom.feature_builders.inventory_features  import build_inventory_features
from processors.custom.feature_builders.competitor_features import build_competitor_features
from processors.custom.feature_builders.sku_dedup           import remove_duplicate_skus


def build_master_dataset(sku_df, price_df, inventory_raw,
                          orders_df, ga4_data, comp_data) -> tuple:
    print("Building master dataset...")

    demand_df    = build_demand_features(orders_df, ga4_data)
    inventory_df = build_inventory_features(inventory_raw)
    comp_df      = build_competitor_features(price_df, sku_df, comp_data)

    master = price_df.merge(demand_df,    on="sku_id", how="left")
    master = master.merge(inventory_df,   on="sku_id", how="left")
    master = master.merge(comp_df,        on="sku_id", how="left")

    master["current_price"]     = master["list_price"]
    master = master[master["total_inventory"] > 0]
    master["demand_bucket"]     = master["demand_bucket"].fillna("LOW")
    master["inventory_bucket"]  = master["inventory_bucket"].fillna("LOW")
    master["competitor_bucket"] = master["competitor_bucket"].fillna("NO_COMPETITOR")
    master = remove_duplicate_skus(master, sku_df)

    print(f"Master dataset ready. Rows: {len(master)}")
    return master, orders_df
