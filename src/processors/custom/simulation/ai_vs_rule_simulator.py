"""
ai_vs_rule_simulator.py
Compares AI pricing vs rule-based pricing revenue.
Replaces: src/simulation/ai_vs_rule_simulator.py
"""
import pandas as pd
from utils.constants import (
    COL_NEW_PRICE, COL_DEMAND_QTY, COL_AI_OPTIMAL_PRICE, COL_CURRENT_PRICE,
    COL_RULE_REVENUE, COL_AI_REVENUE, COL_REVENUE_LIFT, COL_REVENUE_LIFT_PCT,
    COL_CURRENT_REVENUE, COL_REVENUE_CHANGE, COL_REVENUE_CHANGE_PCT,
    SUMMARY_RULE_REVENUE, SUMMARY_AI_REVENUE, SUMMARY_REVENUE_LIFT,
    SUMMARY_LIFT_PCT, SUMMARY_CURRENT_REVENUE,
    SUMMARY_REVENUE_CHANGE, SUMMARY_REVENUE_CHANGE_PCT,
)


def compare_ai_vs_rule(master_df: pd.DataFrame) -> dict:
    sim = master_df.copy()
    sim[COL_RULE_REVENUE]    = sim[COL_NEW_PRICE]        * sim[COL_DEMAND_QTY]
    sim[COL_AI_REVENUE]      = sim[COL_AI_OPTIMAL_PRICE] * sim[COL_DEMAND_QTY]
    sim[COL_REVENUE_LIFT]    = sim[COL_AI_REVENUE] - sim[COL_RULE_REVENUE]
    sim[COL_REVENUE_LIFT_PCT] = sim.apply(
        lambda x: 0 if x[COL_RULE_REVENUE] == 0
        else (x[COL_REVENUE_LIFT] / x[COL_RULE_REVENUE]) * 100, axis=1).round(2)

    tr, ta = sim[COL_RULE_REVENUE].sum(), sim[COL_AI_REVENUE].sum()
    tl     = ta - tr
    summary = {
        SUMMARY_RULE_REVENUE:  round(float(tr), 2),
        SUMMARY_AI_REVENUE:    round(float(ta), 2),
        SUMMARY_REVENUE_LIFT:  round(float(tl), 2),
        SUMMARY_LIFT_PCT:      round(float(tl / tr * 100), 2) if tr else 0}
    print(f"AI vs Rule lift: {summary[SUMMARY_LIFT_PCT]}%")
    return {"data": sim, "summary": summary}


def compare_current_vs_ai(master_df: pd.DataFrame) -> dict:
    sim = master_df.copy()
    sim[COL_CURRENT_REVENUE] = sim[COL_CURRENT_PRICE]    * sim[COL_DEMAND_QTY]
    sim[COL_AI_REVENUE]      = sim[COL_AI_OPTIMAL_PRICE] * sim[COL_DEMAND_QTY]
    tc, ta = sim[COL_CURRENT_REVENUE].sum(), sim[COL_AI_REVENUE].sum()
    td     = ta - tc
    return {
        SUMMARY_CURRENT_REVENUE:    round(float(tc), 2),
        SUMMARY_AI_REVENUE:         round(float(ta), 2),
        SUMMARY_REVENUE_CHANGE:     round(float(td), 2),
        SUMMARY_REVENUE_CHANGE_PCT: round(float(td / tc * 100), 2) if tc else 0}