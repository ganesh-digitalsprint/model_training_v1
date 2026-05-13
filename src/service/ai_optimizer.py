"""
app/services/ai_prediction_service.py
Prediction service for single SKU using trained model.

This mirrors the logic from your original run_ai_service() but works
with individual SKUs fetched from the feature store.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def compute_ai_price(
    current_price: float,
    competitor_price: Optional[float],
    inventory_level: float,
    price_diff_pct: float,
    demand_qty: float,
    model,
    tolerance: float = 0.05,
    n_steps: int = 15,
) -> Tuple[float, str]:
    """
    Revenue-optimisation for a single SKU.

    Sweep candidate prices between ±30% of current price.
    Pick the one that maximises predicted revenue (price × demand).
    Never exceed competitor price.
    Reject if best revenue drops more than `tolerance` below current.

    Parameters
    ----------
    current_price     : Current price of the SKU
    competitor_price  : Competitor's price (None → use current_price)
    inventory_level   : Available inventory
    price_diff_pct    : (current - competitor) / competitor × 100
    demand_qty        : Historical/baseline demand
    model             : Trained sklearn/xgboost model
    tolerance         : Max allowed revenue decline (0.05 = 5%)
    n_steps           : Number of candidate prices to sweep

    Returns
    -------
    (optimal_price, pricing_reason)
        optimal_price  : Recommended price (float)
        pricing_reason : One of:
                        - "AI_REVENUE_OPTIMIZED"  (found better price)
                        - "NO_IMPROVEMENT"        (no price beat current)
                        - "REJECTED_LOW_REVENUE"  (best dropped revenue too much)
    """
    
    # Normalize competitor_price
    if competitor_price is None or (
        isinstance(competitor_price, float) and np.isnan(competitor_price)
    ):
        competitor_price = current_price
    else:
        competitor_price = float(competitor_price)

    # Baseline: current price with historical demand
    baseline_demand = max(float(demand_qty), 0.1)  # avoid 0
    current_revenue = current_price * baseline_demand
    best_price      = current_price
    best_revenue    = current_revenue
    best_demand     = baseline_demand
    reason          = "NO_IMPROVEMENT"

    logger.debug(
        f"SKU ai_price sweep | current_price={current_price} | "
        f"competitor={competitor_price} | inventory={inventory_level} | "
        f"baseline_demand={baseline_demand} | current_revenue={current_revenue:.2f}"
    )

    # Sweep candidate prices
    candidates = np.linspace(
        current_price * 0.7,   # -30%
        current_price * 1.3,   # +30%
        n_steps,
    )

    for candidate_price in candidates:
        # Hard cap: never exceed competitor
        if competitor_price and candidate_price > competitor_price:
            continue

        # Predict demand at this candidate price
        demand_pred = _predict_demand_wrapper(
            model=model,
            price=candidate_price,
            competitor_price=competitor_price,
            inventory_level=inventory_level,
            price_diff_pct=price_diff_pct,
        )
        demand_pred = max(demand_pred, 0.0)

        # Revenue = price × predicted demand
        revenue = candidate_price * demand_pred

        if revenue > best_revenue:
            best_revenue = revenue
            best_price   = candidate_price
            best_demand  = demand_pred
            reason       = "AI_REVENUE_OPTIMIZED"

            logger.debug(
                f"  candidate={candidate_price:.2f} | demand={demand_pred:.4f} | "
                f"revenue={revenue:.2f} (new best)"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Tolerance gate: reject if revenue dropped too much
    # ─────────────────────────────────────────────────────────────────────────
    if current_revenue > 0:
        revenue_change_pct = (best_revenue - current_revenue) / current_revenue
        if revenue_change_pct < -tolerance:
            logger.debug(
                f"Rejected: revenue dropped {revenue_change_pct*100:.1f}% "
                f"(> -{tolerance*100:.1f}% tolerance)"
            )
            return round(current_price, 2), "REJECTED_LOW_REVENUE"

    logger.debug(
        f"Final: best_price={best_price:.2f} | best_demand={best_demand:.4f} | "
        f"best_revenue={best_revenue:.2f} | reason={reason}"
    )

    return round(best_price, 2), reason


def _predict_demand_wrapper(
    model,
    price: float,
    competitor_price: float,
    inventory_level: float,
    price_diff_pct: float,
) -> float:
    """
    Predict demand using the trained model.
    
    The model was trained on features:
      [price, competitor_price, total_inventory, price_diff_pct]
    
    This wrapper ensures they're passed in the correct order.
    """
    import pandas as pd

    # Build single-row DataFrame with features in training order
    X = pd.DataFrame(
        [[price, competitor_price, inventory_level, price_diff_pct]],
        columns=["price", "competitor_price", "total_inventory", "price_diff_pct"],
    )

    prediction = model.predict(X)[0]
    return float(max(prediction, 0.0))