"""
champion_challenger_simulator.py
Compares champion vs challenger model revenue.
Replaces: src/simulation/champion_challenger_simulator.py
"""
from utils.constants import (
    COL_CHAMPION_REVENUE, COL_CHALLENGER_REVENUE,
    SUMMARY_REVENUE_LIFT, SUMMARY_LIFT_PCT,
    SUMMARY_CHAMPION_REVENUE, SUMMARY_CHALLENGER_REVENUE,
)


def compare_champion_challenger(df) -> dict:
    cr, clr = df[COL_CHAMPION_REVENUE].sum(), df[COL_CHALLENGER_REVENUE].sum()
    lift    = clr - cr
    pct     = (lift / cr * 100) if cr > 0 else 0
    return {SUMMARY_CHAMPION_REVENUE:  cr, SUMMARY_CHALLENGER_REVENUE: clr,
            SUMMARY_REVENUE_LIFT: lift, SUMMARY_LIFT_PCT: pct}