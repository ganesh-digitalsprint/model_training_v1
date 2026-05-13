"""
sku_dedup.py
Removes duplicate SKUs based on display_name + current_price.
Replaces: src/features/sku_dedup.py
"""
import pandas as pd


def remove_duplicate_skus(master: pd.DataFrame, sku_df: pd.DataFrame) -> pd.DataFrame:
    sku_df = sku_df.copy(); sku_df.columns = sku_df.columns.str.lower()
    temp = master.merge(sku_df[["sku_id", "display_name"]], on="sku_id", how="left")
    temp = temp.sort_values(["display_name", "current_price", "sku_id"])
    temp = temp.drop_duplicates(subset=["display_name", "current_price"], keep="first")
    return temp
