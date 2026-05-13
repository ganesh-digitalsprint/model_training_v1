"""
rule_engine.py
Rule-based pricing engine. Dynamic-pricing specific.
Replaces: src/engine/rule_engine.py
"""
import pandas as pd
from utils.constants import (
    COL_CURRENT_PRICE, COL_COMPETITOR_PRICE, COL_COMPETITOR_NAME,
    COL_PRICE_DIFF_PCT, COL_DEMAND_BUCKET, COL_INVENTORY_BUCKET,
    COL_COMPETITOR_BUCKET, COL_NEW_PRICE, COL_DECISION_REASON, COL_ALERT_FLAG,
    BUCKET_INV_VERY_HIGH, BUCKET_INV_LOW, BUCKET_INV_MED,
    BUCKET_DEMAND_HIGH, BUCKET_DEMAND_LOW,
    BUCKET_COMP_NO_COMPETITOR, BUCKET_COMP_WE_CHEAPER,
    BUCKET_COMP_SAME, BUCKET_COMP_COMPETITOR_CHEAPER,
)


def pricing_rule(row) -> tuple:
    price      = row[COL_CURRENT_PRICE]
    comp_price = row.get(COL_COMPETITOR_PRICE)
    comp_name  = row.get(COL_COMPETITOR_NAME)
    price_diff = row.get(COL_PRICE_DIFF_PCT)
    demand     = row[COL_DEMAND_BUCKET]
    inv        = row[COL_INVENTORY_BUCKET]
    comp       = row[COL_COMPETITOR_BUCKET]
    alert_flag = False

    # Large price gap → manual review
    if price_diff is not None and (price_diff > 5 or price_diff < -5):
        return price, "Manual review required (large price gap)", True

    # Near price match → undercut by 0.1%
    if price_diff is not None and -2 <= price_diff <= 2 and comp_price:
        return comp_price * 0.999, "Auto match competitor (0.1% lower)", alert_flag

    # Stock clearance
    if inv == BUCKET_INV_VERY_HIGH and demand == BUCKET_DEMAND_LOW:
        return price * 0.70, "Stock clearance 30%", alert_flag

    # Monopoly premium
    if comp == BUCKET_COMP_NO_COMPETITOR and demand == BUCKET_DEMAND_HIGH and inv in [BUCKET_INV_LOW, BUCKET_INV_MED]:
        return price * 1.25, "Monopoly premium 25%", alert_flag

    # High demand, low stock, we are cheaper
    if demand == BUCKET_DEMAND_HIGH and inv == BUCKET_INV_LOW and comp == BUCKET_COMP_WE_CHEAPER:
        return price * 1.20, "Big increase", alert_flag

    if demand == BUCKET_DEMAND_HIGH and inv == BUCKET_INV_LOW and comp == BUCKET_COMP_SAME:
        return price * 1.10, "Increase", alert_flag

    # Match competitor
    if comp == BUCKET_COMP_COMPETITOR_CHEAPER and comp_price:
        return comp_price * 0.99, f"Match {comp_name}", alert_flag

    return price, "No change", alert_flag


def generate_rule_output(master_df: pd.DataFrame) -> pd.DataFrame:
    master = master_df.copy()
    master[[COL_NEW_PRICE, COL_DECISION_REASON, COL_ALERT_FLAG]] = master.apply(
        lambda x: pd.Series(pricing_rule(x)), axis=1)
    return master