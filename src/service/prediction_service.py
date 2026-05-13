# """
# app/services/prediction_service.py

# CORRECT APPROACH: Mirrors your original working code.

# Instead of passing individual features, we:
# 1. Fetch requested SKUs from feature store
# 2. Build a master_df (just like build_master_dataset)
# 3. Run compute_ai_price on the master_df
# 4. Return results

# This ensures feature columns match exactly what the model was trained on.
# """

# from __future__ import annotations

# import logging
# import pandas as pd
# import numpy as np
# from datetime import datetime, timezone
# from typing import Optional, Tuple, List

# from src.schemas.schemas import SKUPricingResult
# from src.service.feature_store_service import FeatureStoreService
# from src.service.model_service import ModelService

# logger = logging.getLogger(__name__)

# # ─────────────────────────────────────────────────────────────────────────────
# # Tunable thresholds
# # ─────────────────────────────────────────────────────────────────────────────
# REVENUE_IMPROVEMENT_THRESHOLD = 0.01
# PRICE_SEARCH_MIN_FACTOR       = 0.70
# PRICE_SEARCH_MAX_FACTOR       = 1.30
# N_PRICE_CANDIDATES            = 15
# COMPETITOR_PRICE_CAP_FACTOR   = 1.05
# PRICE_FLOOR_FACTOR            = 0.50
# ALERT_PRICE_GAP_THRESHOLD     = 20.0

# INV_VERY_HIGH_THRESHOLD = 500
# INV_HIGH_THRESHOLD      = 200
# INV_MED_THRESHOLD       = 50

# CONFIDENCE_HIGH_THRESHOLD   = 0.70
# CONFIDENCE_MEDIUM_THRESHOLD = 0.40


# # ─────────────────────────────────────────────────────────────────────────────
# # Bucket classifiers
# # ─────────────────────────────────────────────────────────────────────────────

# def _inventory_bucket(inventory: float) -> str:
#     if inventory >= INV_VERY_HIGH_THRESHOLD:
#         return "VERY_HIGH"
#     if inventory >= INV_HIGH_THRESHOLD:
#         return "HIGH"
#     if inventory >= INV_MED_THRESHOLD:
#         return "MED"
#     return "LOW"


# def _demand_bucket(demand_qty: float) -> str:
#     return "HIGH" if demand_qty > 5.0 else "LOW"


# def _competitor_bucket(our_price: float, comp_price: Optional[float]) -> str:
#     if comp_price is None or comp_price <= 0:
#         return "NO_COMPETITOR"
#     diff_pct = ((our_price - comp_price) / comp_price) * 100
#     if abs(diff_pct) <= 2:
#         return "SAME"
#     return "WE_CHEAPER" if diff_pct < -2 else "COMPETITOR_CHEAPER"


# def _confidence_label(score: float) -> str:
#     if score >= CONFIDENCE_HIGH_THRESHOLD:
#         return "HIGH"
#     if score >= CONFIDENCE_MEDIUM_THRESHOLD:
#         return "MEDIUM"
#     return "LOW"


# # ─────────────────────────────────────────────────────────────────────────────
# # Confidence scoring
# # ─────────────────────────────────────────────────────────────────────────────

# def _compute_confidence(row: pd.Series, price_change_pct: float) -> float:
#     """
#     Heuristic confidence 0–1 based on data quality.
#     """
#     score = 0.0
    
#     if pd.notna(row.get("competitor_price")) and row["competitor_price"] > 0:
#         score += 0.25
    
#     if row.get("total_inventory", 0) >= INV_MED_THRESHOLD:
#         score += 0.20
    
#     if row.get("demand_qty", 0) > 1.0:
#         score += 0.20
    
#     if row.get("ga4_views", 0) > 0 or row.get("ga4_add_to_cart", 0) > 0:
#         score += 0.15
    
#     if abs(price_change_pct) < 10.0:
#         score += 0.20
    
#     return round(min(score, 1.0), 4)


# # ─────────────────────────────────────────────────────────────────────────────
# # AI Price Computation (MATCHES YOUR ORIGINAL CODE)
# # ─────────────────────────────────────────────────────────────────────────────

# def _compute_ai_price(
#     row: dict,
#     model,
#     tolerance: float = 0.05,
#     n_steps: int = 15,
# ) -> Tuple[float, str]:
#     """
#     Revenue-optimisation for a single SKU row.
#     EXACT COPY of logic from your original ai_optimizer.py
    
#     Parameters
#     ----------
#     row       : dict with current_price, competitor_price, total_inventory, 
#                 price_diff_pct, demand_qty
#     model     : loaded sklearn/xgboost model
#     tolerance : max allowed revenue decline (0.05 = 5%)
#     n_steps   : number of candidate prices to sweep

#     Returns
#     -------
#     (optimal_price, pricing_reason)
#     """
#     current_price    = float(row["current_price"])
#     competitor_price = row.get("competitor_price")
#     inventory_level  = float(row.get("total_inventory", 0))
#     price_diff_pct   = float(row.get("price_diff_pct", 0))
#     demand_qty       = float(row.get("demand_qty", 1))

#     if competitor_price is None or (
#         isinstance(competitor_price, float) and np.isnan(competitor_price)
#     ):
#         competitor_price = current_price
#     else:
#         competitor_price = float(competitor_price)

#     best_price   = current_price
#     best_revenue = current_price * demand_qty
#     reason       = "NO_IMPROVEMENT"

#     for candidate_price in np.linspace(
#         current_price * 0.7,
#         current_price * 1.3,
#         n_steps,
#     ):
#         # Hard cap — never exceed competitor
#         if competitor_price and candidate_price > competitor_price:
#             continue

#         # CRITICAL: Create DataFrame with column names matching training
#         # Model was trained on: [price, competitor_price, inventory_level, price_diff_pct]
#         X = pd.DataFrame(
#             [[candidate_price, competitor_price, inventory_level, price_diff_pct]],
#             columns=["price", "competitor_price", "inventory_level", "price_diff_pct"],
#         )
        
#         demand_pred = model.predict(X)[0]
#         demand_pred = max(demand_pred, 0)
#         revenue     = candidate_price * demand_pred

#         if revenue > best_revenue:
#             best_revenue = revenue
#             best_price   = candidate_price
#             reason       = "AI_REVENUE_OPTIMIZED"

#     # ── Revenue-loss tolerance gate ───────────────────────────────────────
#     current_revenue = current_price * demand_qty
#     if current_revenue > 0:
#         revenue_change_pct = (best_revenue - current_revenue) / current_revenue
#         if revenue_change_pct < -tolerance:
#             return current_price, "REJECTED_LOW_REVENUE"

#     return round(best_price, 2), reason


# def _apply_business_rules(
#     ai_optimal_price: float,
#     row: dict,
#     has_competitor: bool,
#     price_diff_pct: float,
# ) -> Tuple[float, str, str, str, bool]:
#     """
#     Apply business guardrails to the AI-optimal price.
#     """
#     current_price = row["current_price"]
#     floor_price   = current_price * PRICE_FLOOR_FACTOR

#     # Rule 1: Price gap alert
#     if has_competitor and price_diff_pct > ALERT_PRICE_GAP_THRESHOLD:
#         return (
#             current_price,
#             "RULE_OVERRIDE_ALERT",
#             "RULE_OVERRIDE",
#             "Manual review required — our price significantly exceeds competitor.",
#             True,
#         )

#     # Rule 2: Safety floor
#     new_price = max(ai_optimal_price, floor_price)

#     # Rule 3: Decision explanation
#     price_change_pct = ((new_price - current_price) / current_price) * 100 if current_price > 0 else 0

#     if abs(price_change_pct) < 0.5:
#         return (
#             new_price,
#             "NO_CHANGE",
#             "AI_OPTIMISED",
#             "Current price is already optimal — no change required.",
#             False,
#         )
#     elif price_change_pct < 0:
#         decision = (
#             f"Price reduced by {abs(price_change_pct):.1f}% to maximise revenue"
#             + (" and remain competitive." if has_competitor else ".")
#         )
#         return new_price, "AI_OPTIMISED", "AI_OPTIMISED", decision, False
#     else:
#         decision = (
#             f"Price increased by {price_change_pct:.1f}% based on demand signals"
#             + (" with limited competitive pressure." if not has_competitor else ".")
#         )
#         return new_price, "AI_OPTIMISED", "AI_OPTIMISED", decision, False


# # ─────────────────────────────────────────────────────────────────────────────
# # Build master dataset for requested SKUs (like your original code)
# # ─────────────────────────────────────────────────────────────────────────────

# def _build_master_df_for_skus(sku_ids: List[str]) -> pd.DataFrame:
#     """
#     Build a master dataframe for requested SKUs.
#     This mirrors your original build_master_dataset() logic.
#     """
#     store = FeatureStoreService.get_instance()
    
#     rows = []
#     for sku_id in sku_ids:
#         feat = store.get(sku_id)
#         if feat is None:
#             continue
#         print("particular sku",feat)
#         rows.append({
#             "sku_id": feat.sku_id,
#             "display_name": feat.display_name,
#             "list_price": feat.list_price,
#             "current_price": feat.current_price,
#             "total_inventory": feat.total_inventory,
#             "demand_qty": feat.demand_qty,
#             "competitor_price": feat.competitor_price,
#             "competitor_name": feat.competitor_name,
#             "price_diff_pct": (
#                 ((feat.list_price - feat.competitor_price) / feat.competitor_price * 100)
#                 if feat.competitor_price and feat.competitor_price > 0
#                 else 0.0
#             ),
#             "all_competitors_json": feat.all_competitors_json,
#             "ga4_views": feat.ga4_views,
#             "ga4_add_to_cart": feat.ga4_add_to_cart,
#         })
    
#     if not rows:
#         return pd.DataFrame()
    
#     master = pd.DataFrame(rows)
    
#     # Fill NaNs like original code
#     master["demand_qty"] = master["demand_qty"].fillna(0)
#     master["competitor_price"] = master["competitor_price"].fillna(0)
    
#     logger.info(f"Built master_df with {len(master)} SKUs")
#     return master


# # ─────────────────────────────────────────────────────────────────────────────
# # Main prediction functions
# # ─────────────────────────────────────────────────────────────────────────────

# def predict_batch(
#     sku_ids: List[str],
#     run_id: str,
#     model_version: str,
#     tolerance: float = 0.05,
# ) -> Tuple[List[SKUPricingResult], List[dict]]:
#     """
#     Predict for a batch of SKU IDs.
    
#     Flow (matches your original run_pricing):
#       1. Build master_df for requested SKUs (from feature store)
#       2. Run AI optimization on master_df
#       3. Apply guardrails
#       4. Return SKUPricingResult list
#     """
#     results: List[SKUPricingResult] = []
#     errors: List[dict] = []

#     # Step 1: Build master dataframe
#     logger.info(f"Building master dataset for {len(sku_ids)} SKUs...")
#     master = _build_master_df_for_skus(sku_ids)
    
#     if master.empty:
#         for sku_id in sku_ids:
#             errors.append({"sku_id": sku_id, "error": "SKU not found in feature store"})
#         return results, errors

#     # Step 2: Load model
#     try:
#         model_svc = ModelService.get_instance()
#     except FileNotFoundError as exc:
#         logger.error(f"Model not found: {exc}")
#         for sku_id in sku_ids:
#             errors.append({"sku_id": sku_id, "error": f"Model unavailable: {exc}"})
#         return results, errors

#     # Step 3: Run AI optimization for each row
#     logger.info("Running AI price optimization...")
    
#     for idx, row in master.iterrows():
#         try:
#             sku_id = row["sku_id"]
            
#             # AI optimize
#             ai_optimal_price, ai_reason = _compute_ai_price(
#                 row.to_dict(),
#                 model_svc.model,
#                 tolerance=tolerance,
#                 n_steps=N_PRICE_CANDIDATES,
#             )
#             print("ai reason",ai_reason)
#             # Apply guardrails
#             has_competitor = row["competitor_price"] > 0 and pd.notna(row["competitor_price"])
#             price_diff_pct = row["price_diff_pct"]
            
#             (
#                 new_price,
#                 pricing_reason,
#                 ai_strategy,
#                 decision_reason,
#                 alert_flag,
#             ) = _apply_business_rules(
#                 ai_optimal_price=ai_optimal_price,
#                 row=row.to_dict(),
#                 has_competitor=has_competitor,
#                 price_diff_pct=price_diff_pct,
#             )
            
#             # Re-predict demand at final price if changed
#             if new_price != ai_optimal_price:
#                 rdiff = (
#                     ((new_price - row["competitor_price"]) / row["competitor_price"] * 100)
#                     if row["competitor_price"] > 0
#                     else 0.0
#                 )
#                 X = pd.DataFrame(
#                     [[new_price, row["competitor_price"], row["total_inventory"], rdiff]],
#                     columns=["price", "competitor_price", "inventory_level", "price_diff_pct"],
#                 )
#                 demand_qty = float(model_svc.model.predict(X)[0])
#             else:
#                 # Demand at AI-optimal price
#                 X = pd.DataFrame(
#                     [[ai_optimal_price, row["competitor_price"], row["total_inventory"], price_diff_pct]],
#                     columns=["price", "competitor_price", "inventory_level", "price_diff_pct"],
#                 )
#                 demand_qty = float(model_svc.model.predict(X)[0])
            
#             # Derived fields
#             current_price = row["current_price"]
#             price_change_pct = round(((new_price - current_price) / current_price) * 100, 2) if current_price > 0 else 0.0
#             confidence_score = _compute_confidence(row, price_change_pct)
#             run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            
#             # Assemble result
#             result = SKUPricingResult(
#                 sku_id=sku_id,
#                 display_name=row["display_name"],
#                 list_price=row["list_price"],
#                 current_price=current_price,
#                 competitor_price=row["competitor_price"] if row["competitor_price"] > 0 else None,
#                 competitor_name=row.get("competitor_name"),
#                 competitor_bucket=_competitor_bucket(current_price, row["competitor_price"] if row["competitor_price"] > 0 else None),
#                 price_diff_pct=round(price_diff_pct, 2),

#                 all_competitors_json=(
#                      row["all_competitors_json"]
#                         if isinstance(row.get("all_competitors_json"), dict)
#                         else None
#                     ),
#                 total_inventory=row["total_inventory"],
#                 inventory_bucket=_inventory_bucket(row["total_inventory"]),
#                 demand_qty=round(demand_qty, 4),
#                 demand_bucket=_demand_bucket(demand_qty),
#                 new_price=round(new_price, 2),
#                 ai_optimal_price=round(ai_optimal_price, 2),
#                 price_change_pct=price_change_pct,
#                 confidence_score=confidence_score,
#                 confidence_label=_confidence_label(confidence_score),
#                 pricing_reason=pricing_reason,
#                 ai_strategy=ai_strategy,
#                 decision_reason=decision_reason,
#                 alert_flag=alert_flag,
#                 run_date=run_date,
#                 run_id=run_id,
#                 model_version=model_version,
#             )
#             results.append(result)
            
#         except Exception as exc:
#             logger.exception(f"Prediction failed for SKU {row.get('sku_id')}")
#             errors.append({
#                 "sku_id": row.get("sku_id", "unknown"),
#                 "error": str(exc)
#             })
    
#     logger.info(f"Prediction complete: {len(results)} successful, {len(errors)} failed")
#     return results, errors





"""
app/services/prediction_service.py

Mirrors the original working code:
1. fetch SKUs from feature store
2. build master_df with real demand_qty from orders
3. apply rule engine (sets new_price + alert_flag)
4. run compute_ai_price (revenue optimization)
5. compare_current_vs_ai
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple, List

from src.schemas.schemas import SKUPricingResult
from src.service.feature_store_service import FeatureStoreService
from src.service.model_service import ModelService

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunable thresholds  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
REVENUE_IMPROVEMENT_THRESHOLD = 0.01
PRICE_SEARCH_MIN_FACTOR       = 0.70
PRICE_SEARCH_MAX_FACTOR       = 1.30
N_PRICE_CANDIDATES            = 15
COMPETITOR_PRICE_CAP_FACTOR   = 1.05
PRICE_FLOOR_FACTOR            = 0.50
ALERT_PRICE_GAP_THRESHOLD     = 20.0

INV_VERY_HIGH_THRESHOLD = 500
INV_HIGH_THRESHOLD      = 200
INV_MED_THRESHOLD       = 50

CONFIDENCE_HIGH_THRESHOLD   = 0.70
CONFIDENCE_MEDIUM_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# predict_demand — mirrors ml/inference.py predict_demand exactly
# ─────────────────────────────────────────────────────────────────────────────

def _predict_demand(model, price, competitor_price, inventory_level, price_diff_pct) -> float:
    """
    Mirrors predict_demand() from your original ml/inference.py.
    Model was trained on: [price, competitor_price, inventory_level, price_diff_pct]
    """
    X = pd.DataFrame(
        [[price, competitor_price, inventory_level, price_diff_pct]],
        columns=["price", "competitor_price", "inventory_level", "price_diff_pct"],
    )
    return float(model.predict(X)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Rule engine — mirrors engine/rule_engine.py generate_rule_output exactly
# ─────────────────────────────────────────────────────────────────────────────

def _generate_rule_output(master: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors generate_rule_output() from your original rule_engine.py.
    Sets new_price, alert_flag, pricing_reason on every row.
    """
    master = master.copy()

    def apply_rules(row):
        current_price    = row["current_price"]
        competitor_price = row.get("competitor_price", None)
        price_diff_pct   = row.get("price_diff_pct", 0)

        # Rule 1: Large price gap — flag for manual review
        if pd.notna(competitor_price) and competitor_price > 0:
            if abs(price_diff_pct) > ALERT_PRICE_GAP_THRESHOLD:
                return pd.Series({
                    "new_price":      current_price,
                    "alert_flag":     True,
                    "pricing_reason": "RULE_OVERRIDE_ALERT",
                    "decision_reason": "Manual review required (large price gap)",
                })

        # Rule 2: Auto-match competitor (within 0.1% tolerance)
        if pd.notna(competitor_price) and competitor_price > 0:
            diff_pct = ((current_price - competitor_price) / competitor_price) * 100
            if abs(diff_pct) <= 0.1:
                return pd.Series({
                    "new_price":       round(competitor_price, 2),
                    "alert_flag":      False,
                    "pricing_reason":  "RULE_AUTO_MATCH",
                    "decision_reason": f"Auto match competitor (0.1% lower)",
                })

            # Rule 3: Match lowest competitor
            match_name = row.get("competitor_name", "")
            return pd.Series({
                "new_price":       round(competitor_price * 0.999, 2),
                "alert_flag":      False,
                "pricing_reason":  "RULE_MATCH_COMPETITOR",
                "decision_reason": f"Match {match_name}",
            })

        # No competitor — keep current price
        return pd.Series({
            "new_price":       current_price,
            "alert_flag":      False,
            "pricing_reason":  "NO_COMPETITOR",
            "decision_reason": "No competitor data available",
        })

    rule_cols = master.apply(apply_rules, axis=1)
    master[["new_price", "alert_flag", "pricing_reason", "decision_reason"]] = rule_cols

    return master


# ─────────────────────────────────────────────────────────────────────────────
# compute_ai_price — EXACT copy of your working compute_ai_price
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ai_price(row: dict, model, tolerance: float = 0.05) -> Tuple[float, str]:
    """
    Exact copy of compute_ai_price from your working ai_engine.py.
    """
    # Alert override — don't touch the price
    if row.get("alert_flag", False):
        return row.get("new_price", row["current_price"]), "RULE_OVERRIDE_ALERT"

    current_price    = row["current_price"]
    competitor_price = row.get("competitor_price")
    inventory_level  = row.get("total_inventory", 0)
    price_diff_pct   = row.get("price_diff_pct", 0)
    demand_qty       = row.get("demand_qty", 1)

    if pd.isna(competitor_price):
        competitor_price = current_price

    best_price   = current_price
    best_revenue = current_price * demand_qty
    reason       = "NO_IMPROVEMENT"

    for p in np.linspace(current_price * 0.7, current_price * 1.3, 15):
        # Cannot exceed competitor
        if competitor_price and p > competitor_price:
            continue

        demand_pred = _predict_demand(
            model,
            p,
            competitor_price,
            inventory_level,
            price_diff_pct,
        )
        demand_pred = max(demand_pred, 0)
        revenue     = p * demand_pred

        if revenue > best_revenue:
            best_revenue = revenue
            best_price   = p
            reason       = "AI_REVENUE_OPTIMIZED"

    # 5% revenue loss tolerance gate
    current_revenue = current_price * demand_qty
    if current_revenue > 0:
        revenue_change_pct = (best_revenue - current_revenue) / current_revenue
        if revenue_change_pct < -tolerance:
            return current_price, "REJECTED_LOW_REVENUE"

    return round(best_price, 2), reason


# ─────────────────────────────────────────────────────────────────────────────
# generate_ai_output — EXACT copy of your working generate_ai_output
# ─────────────────────────────────────────────────────────────────────────────

def _generate_ai_output(master: pd.DataFrame, model, tolerance: float = 0.05) -> pd.DataFrame:
    """
    Exact copy of generate_ai_output from your working ai_engine.py.
    Runs rule engine first, then AI optimization on top.
    """
    # Step 1: Rule engine sets new_price + alert_flag
    master = _generate_rule_output(master)

    # Step 2: AI optimization
    results = master.apply(
        lambda x: _compute_ai_price(x.to_dict(), model, tolerance),
        axis=1,
    )
    master[["ai_optimal_price", "pricing_reason"]] = pd.DataFrame(
        results.tolist(), index=master.index
    )

    # Step 3: AI strategy flag
    master["ai_strategy"] = np.where(
        master["alert_flag"],
        "RULE_OVERRIDE",
        "AI_OPTIMISED",
    )

    # Step 4: Price change %
    master["price_change_pct"] = (
        (master["ai_optimal_price"] - master["current_price"])
        / master["current_price"] * 100
    ).round(2)

    return master


# ─────────────────────────────────────────────────────────────────────────────
# compare_current_vs_ai — EXACT copy of your working compare_current_vs_ai
# ─────────────────────────────────────────────────────────────────────────────

def compare_current_vs_ai(master_df: pd.DataFrame) -> dict:
    """
    Exact copy of compare_current_vs_ai from your working revenue simulator.
    """
    if "demand_qty" not in master_df.columns:
        raise ValueError("demand_qty missing for revenue comparison")

    sim = master_df.copy()
    sim["current_revenue"] = sim["current_price"] * sim["demand_qty"]
    sim["ai_revenue"]      = sim["ai_optimal_price"] * sim["demand_qty"]

    total_current = sim["current_revenue"].sum()
    total_ai      = sim["ai_revenue"].sum()
    total_diff    = total_ai - total_current

    return {
        "current_revenue":     round(float(total_current), 2),
        "ai_revenue":          round(float(total_ai), 2),
        "revenue_change":      round(float(total_diff), 2),
        "revenue_change_pct":  round(
            float((total_diff / total_current) * 100) if total_current != 0 else 0, 2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bucket helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inventory_bucket(inventory: float) -> str:
    if inventory >= INV_VERY_HIGH_THRESHOLD: return "VERY_HIGH"
    if inventory >= INV_HIGH_THRESHOLD:      return "HIGH"
    if inventory >= INV_MED_THRESHOLD:       return "MED"
    return "LOW"


def _demand_bucket(demand_qty: float) -> str:
    return "HIGH" if demand_qty > 5.0 else "LOW"


def _competitor_bucket(our_price: float, comp_price: Optional[float]) -> str:
    if comp_price is None or comp_price <= 0:
        return "NO_COMPETITOR"
    diff_pct = ((our_price - comp_price) / comp_price) * 100
    if abs(diff_pct) <= 2:  return "SAME"
    return "WE_CHEAPER" if diff_pct < -2 else "COMPETITOR_CHEAPER"


def _confidence_label(score: float) -> str:
    if score >= CONFIDENCE_HIGH_THRESHOLD:   return "HIGH"
    if score >= CONFIDENCE_MEDIUM_THRESHOLD: return "MEDIUM"
    return "LOW"


def _compute_confidence(row: pd.Series, price_change_pct: float) -> float:
    score = 0.0
    if pd.notna(row.get("competitor_price")) and row["competitor_price"] > 0: score += 0.25
    if row.get("total_inventory", 0) >= INV_MED_THRESHOLD:                    score += 0.20
    if row.get("demand_qty", 0) > 1.0:                                         score += 0.20
    if row.get("ga4_views", 0) > 0 or row.get("ga4_add_to_cart", 0) > 0:     score += 0.15
    if abs(price_change_pct) < 10.0:                                           score += 0.20
    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Build master_df for requested SKUs — with REAL demand_qty from orders
# ─────────────────────────────────────────────────────────────────────────────

def _build_master_df_for_skus(sku_ids: List[str]) -> pd.DataFrame:
    """
    Builds master_df from feature store.
    demand_qty comes from SKUFeatures.demand_qty which must be populated
    from real order quantities — NOT left as 0.0.
    """
    store = FeatureStoreService.get_instance()

    rows = []
    for sku_id in sku_ids:
        feat = store.get(sku_id)
        if feat is None:
            logger.warning("SKU not found in feature store: %s", sku_id)
            continue

        comp_price = feat.competitor_price if feat.competitor_price and feat.competitor_price > 0 else None
        price_diff_pct = (
            ((feat.list_price - comp_price) / comp_price * 100)
            if comp_price else 0.0
        )

        rows.append({
            "sku_id":               feat.sku_id,
            "display_name":         feat.display_name,
            "list_price":           feat.list_price,
            "current_price":        feat.current_price,
            "total_inventory":      feat.total_inventory,
            "demand_qty":           feat.demand_qty,      # ← real orders qty, not 0
            "competitor_price":     comp_price if comp_price else np.nan,
            "competitor_name":      feat.competitor_name,
            "price_diff_pct":       round(price_diff_pct, 2),
            "all_competitors_json": feat.all_competitors_json,
            "ga4_views":            feat.ga4_views,
            "ga4_add_to_cart":      feat.ga4_add_to_cart,
        })

    if not rows:
        return pd.DataFrame()

    master = pd.DataFrame(rows)
    
    master["demand_qty"] = master["demand_qty"].fillna(0)
    logger.info("Built master_df with %d SKUs", len(master))
    return master


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    sku_ids: List[str],
    run_id: str,
    model_version: str,
    tolerance: float = 0.05,
) -> Tuple[List[SKUPricingResult], List[dict]]:
    """
    Main entry point. Mirrors your original run_pricing() flow exactly:
      1. Build master_df from feature store
      2. Run rule engine + AI optimization via generate_ai_output
      3. compare_current_vs_ai (logged)
      4. Assemble SKUPricingResult list
    """
    results: List[SKUPricingResult] = []
    errors:  List[dict]             = []

    # ── Step 1: Build master_df ───────────────────────────────────────────
    master = _build_master_df_for_skus(sku_ids)

    if master.empty:
        for sku_id in sku_ids:
            errors.append({"sku_id": sku_id, "error": "SKU not found in feature store"})
        return results, errors

    # Track which SKUs were found
    found_ids   = set(master["sku_id"].tolist())
    missing_ids = [s for s in sku_ids if s not in found_ids]
    for sku_id in missing_ids:
        errors.append({"sku_id": sku_id, "error": "SKU not found in feature store"})

    # ── Step 2: Load model ────────────────────────────────────────────────
    try:
        model_svc = ModelService.get_instance()
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        for sku_id in sku_ids:
            errors.append({"sku_id": sku_id, "error": f"Model unavailable: {exc}"})
        return results, errors

    # ── Step 3: Rule engine + AI optimization (exact working flow) ────────
    master = _generate_ai_output(master, model_svc.model, tolerance=tolerance)

    # ── Step 4: Revenue comparison (logged) ───────────────────────────────
    rev_summary = compare_current_vs_ai(master)
    logger.info(
        "Revenue summary | current=%.2f | ai=%.2f | change=%.2f (%.2f%%)",
        rev_summary["current_revenue"],
        rev_summary["ai_revenue"],
        rev_summary["revenue_change"],
        rev_summary["revenue_change_pct"],
    )

    # ── Step 5: Assemble SKUPricingResult per row ─────────────────────────
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    for _, row in master.iterrows():
        try:
            sku_id        = row["sku_id"]
            current_price = float(row["current_price"])
            new_price     = float(row["ai_optimal_price"])   # AI wins
            comp_price    = float(row["competitor_price"]) if pd.notna(row["competitor_price"]) else None

            price_change_pct = round(
                ((new_price - current_price) / current_price * 100)
                if current_price > 0 else 0.0, 2
            )
            confidence_score = _compute_confidence(row, price_change_pct)

            # decision_reason comes from rule engine; override with AI reason if AI changed price
            decision_reason = row.get("decision_reason", "")
            pricing_reason  = row["pricing_reason"]
            ai_strategy     = row["ai_strategy"]

            # all_competitors_json — pass dict directly
            comp_json = row.get("all_competitors_json")
            if isinstance(comp_json, str):
                try:
                    comp_json = json.loads(comp_json)
                except Exception:
                    comp_json = None

            result = SKUPricingResult(
                sku_id=sku_id,
                display_name=str(row["display_name"]),
                list_price=float(row["list_price"]),
                current_price=current_price,
                competitor_price=comp_price,
                competitor_name=row.get("competitor_name"),
                competitor_bucket=_competitor_bucket(current_price, comp_price),
                price_diff_pct=round(float(row["price_diff_pct"]), 2),
                all_competitors_json=comp_json,
                total_inventory=float(row["total_inventory"]),
                inventory_bucket=_inventory_bucket(float(row["total_inventory"])),
                demand_qty=round(float(row["demand_qty"]), 4),
                demand_bucket=_demand_bucket(float(row["demand_qty"])),
                new_price=round(new_price, 2),
                ai_optimal_price=round(float(row["ai_optimal_price"]), 2),
                price_change_pct=price_change_pct,
                confidence_score=confidence_score,
                confidence_label=_confidence_label(confidence_score),
                pricing_reason=pricing_reason,
                ai_strategy=ai_strategy,
                decision_reason=decision_reason,
                alert_flag=bool(row["alert_flag"]),
                run_date=run_date,
                run_id=run_id,
                model_version=model_version,
            )
            results.append(result)

        except Exception as exc:
            logger.exception("Prediction failed for SKU %s", row.get("sku_id"))
            errors.append({"sku_id": row.get("sku_id", "unknown"), "error": str(exc)})

    logger.info("Prediction complete: %d successful, %d failed", len(results), len(errors))
    return results, errors