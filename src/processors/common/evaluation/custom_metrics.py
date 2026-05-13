"""
custom_metrics.py
Domain-specific metrics (e.g. revenue lift, pricing accuracy).
"""
import numpy as np


def revenue_lift(current_prices, new_prices, demand_qty) -> dict:
    current_rev = np.sum(np.array(current_prices) * np.array(demand_qty))
    new_rev     = np.sum(np.array(new_prices)     * np.array(demand_qty))
    lift        = new_rev - current_rev
    lift_pct    = (lift / current_rev * 100) if current_rev > 0 else 0.0
    return {"current_revenue": round(float(current_rev), 2),
            "new_revenue":     round(float(new_rev), 2),
            "revenue_lift":    round(float(lift), 2),
            "lift_pct":        round(float(lift_pct), 2)}
