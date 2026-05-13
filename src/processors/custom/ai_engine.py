"""
ai_engine.py
AI-based revenue-optimising pricing engine using Ray Data for parallel scoring.
Replaces: src/engine/ai_engine.py
"""
import numpy as np
import pandas as pd
import ray
import ray.data
from processors.custom.inference import load_model, predict_demand
from utils.constants import (
    COL_ALERT_FLAG, COL_NEW_PRICE, COL_CURRENT_PRICE,
    COL_COMPETITOR_PRICE, COL_TOTAL_INVENTORY, COL_PRICE_DIFF_PCT, COL_DEMAND_QTY,
    COL_AI_OPTIMAL_PRICE, COL_PRICING_REASON, COL_AI_STRATEGY, COL_PRICE_CHANGE_PCT,
    REASON_RULE_OVERRIDE_ALERT, REASON_NO_IMPROVEMENT,
    REASON_AI_REVENUE_OPTIMIZED, REASON_REJECTED_LOW_REVENUE,
    REASON_RULE_OVERRIDE, REASON_AI_OPTIMIZED,
    LINEAR, VERSION,
)


@ray.remote
def _compute_ai_price_batch(rows: pd.DataFrame, model_ref,
                             tolerance: float = 0.05) -> pd.DataFrame:
    """Process a batch of rows remotely — replaces row-by-row apply."""
    model = ray.get(model_ref)
    results = []
    for _, row in rows.iterrows():
        if row.get(COL_ALERT_FLAG, False):
            results.append((row.get(COL_NEW_PRICE, row[COL_CURRENT_PRICE]), REASON_RULE_OVERRIDE_ALERT))
            continue

        cp   = row[COL_CURRENT_PRICE]
        comp = row.get(COL_COMPETITOR_PRICE, cp) or cp
        if pd.isna(comp): comp = cp

        inv   = row.get(COL_TOTAL_INVENTORY, 0) or 0
        diff  = row.get(COL_PRICE_DIFF_PCT, 0)  or 0
        dqty  = row.get(COL_DEMAND_QTY, 1)       or 1

        best_price, best_rev, reason = cp, cp * dqty, REASON_NO_IMPROVEMENT
        for p in np.linspace(cp * 0.7, cp * 1.3, 15):
            if comp and p > comp: continue
            demand = max(predict_demand(model, p, comp, inv, diff), 0)
            rev = p * demand
            if rev > best_rev:
                best_rev, best_price, reason = rev, p, REASON_AI_REVENUE_OPTIMIZED

        curr_rev = cp * dqty
        if curr_rev > 0 and ((best_rev - curr_rev) / curr_rev) < -tolerance:
            results.append((cp, REASON_REJECTED_LOW_REVENUE))
        else:
            results.append((round(best_price, 2), reason))

    rows = rows.copy()
    rows[[COL_AI_OPTIMAL_PRICE, COL_PRICING_REASON]] = pd.DataFrame(
        results, index=rows.index)
    return rows


def generate_ai_output(master: pd.DataFrame, model_name: str = LINEAR,
                        version: str = VERSION, tolerance: float = 0.05) -> pd.DataFrame:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    model     = load_model(model_name, version)
    model_ref = ray.put(model)

    n_batches = max(1, len(master) // 50)
    batches   = np.array_split(master, n_batches)
    futures   = [_compute_ai_price_batch.remote(b, model_ref, tolerance)
                 for b in batches]
    parts     = ray.get(futures)
    result    = pd.concat(parts, ignore_index=True)

    result[COL_AI_STRATEGY] = np.where(result[COL_ALERT_FLAG], REASON_RULE_OVERRIDE, REASON_AI_OPTIMIZED)
    result[COL_PRICE_CHANGE_PCT] = ((result[COL_AI_OPTIMAL_PRICE] - result[COL_CURRENT_PRICE])
                                     / result[COL_CURRENT_PRICE] * 100).round(2)
    return result