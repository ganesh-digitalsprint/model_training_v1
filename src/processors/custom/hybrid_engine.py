"""
hybrid_engine.py
Hybrid pricing decision: AI when confident, rule-based as fallback.
Replaces: src/engine/hybrid_engine.py
"""
import pandas as pd
from processors.custom.rule_engine import generate_rule_output
from processors.custom.ai_engine   import generate_ai_output
from utils.constants import (
    COL_CURRENT_PRICE, COL_NEW_PRICE, COL_AI_OPTIMAL_PRICE,
    COL_ALERT_FLAG, COL_ELASTICITY_R2, COL_ELASTICITY_MAPE,
    COL_FINAL_PRICE, COL_PRICING_STRATEGY,
    REASON_MANUAL_REVIEW_ALERT, REASON_AI_PRICING, REASON_RULE_PRICING,
    LINEAR, VERSION,
)


def hybrid_decision(row) -> tuple:
    cp, rule_p, ai_p = row[COL_CURRENT_PRICE], row.get(COL_NEW_PRICE), row.get(COL_AI_OPTIMAL_PRICE)
    if row.get(COL_ALERT_FLAG, False):
        return cp, REASON_MANUAL_REVIEW_ALERT
    r2, mape = row.get(COL_ELASTICITY_R2), row.get(COL_ELASTICITY_MAPE)
    if r2 is not None and mape is not None and r2 >= 0.75 and mape <= 20:
        return ai_p, REASON_AI_PRICING
    return rule_p, REASON_RULE_PRICING


def generate_hybrid_output(master_df: pd.DataFrame, model_name: str = LINEAR,
                            version: str = VERSION, tolerance: float = 0.05) -> pd.DataFrame:
    master = generate_rule_output(master_df)
    master = generate_ai_output(master, model_name, version, tolerance)
    master[[COL_FINAL_PRICE, COL_PRICING_STRATEGY]] = master.apply(
        lambda x: pd.Series(hybrid_decision(x)), axis=1)
    return master