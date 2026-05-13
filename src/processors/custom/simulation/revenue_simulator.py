"""
revenue_simulator.py
Simulates revenue impact of new pricing vs current pricing.
Replaces: src/simulation/revenue_simulator.py
"""
import pandas as pd
from utils.constants import (
    COL_DEMAND_QTY, COL_CURRENT_PRICE,
    COL_FINAL_PRICE, COL_AI_OPTIMAL_PRICE, COL_NEW_PRICE,
    COL_CURRENT_REVENUE, COL_SIMULATED_REVENUE,
    COL_REVENUE_CHANGE, COL_REVENUE_CHANGE_PCT,
    SUMMARY_CURRENT_REVENUE, SUMMARY_NEW_REVENUE,
    SUMMARY_REVENUE_CHANGE, SUMMARY_REVENUE_CHANGE_PCT,
)


def run_revenue_simulator(master_df: pd.DataFrame) -> dict:
    print("Running Revenue Impact Simulator...")
    sim = master_df.copy()
    if COL_DEMAND_QTY not in sim.columns:
        raise ValueError("demand_qty missing — must be created in feature layer")

    price_col = next(
        (c for c in [COL_FINAL_PRICE, COL_AI_OPTIMAL_PRICE, COL_NEW_PRICE] if c in sim.columns),
        None)
    if not price_col:
        raise ValueError("No pricing column found for revenue simulation")

    sim[COL_CURRENT_REVENUE]   = sim[COL_CURRENT_PRICE] * sim[COL_DEMAND_QTY]
    sim[COL_SIMULATED_REVENUE] = sim[price_col]          * sim[COL_DEMAND_QTY]
    sim[COL_REVENUE_CHANGE]    = sim[COL_SIMULATED_REVENUE] - sim[COL_CURRENT_REVENUE]
    sim[COL_REVENUE_CHANGE_PCT] = sim.apply(
        lambda x: 0 if x[COL_CURRENT_REVENUE] == 0
        else (x[COL_REVENUE_CHANGE] / x[COL_CURRENT_REVENUE]) * 100, axis=1).round(2)

    tc, tn = sim[COL_CURRENT_REVENUE].sum(), sim[COL_SIMULATED_REVENUE].sum()
    td     = tn - tc
    summary = {
        SUMMARY_CURRENT_REVENUE:    round(float(tc), 2),
        SUMMARY_NEW_REVENUE:        round(float(tn), 2),
        SUMMARY_REVENUE_CHANGE:     round(float(td), 2),
        SUMMARY_REVENUE_CHANGE_PCT: round(float(td / tc * 100), 2) if tc else 0}
    print(f"Revenue Change: {summary[SUMMARY_REVENUE_CHANGE_PCT]}%")
    return {"data": sim, "summary": summary}