"""
elasticity_trainer.py
Per-SKU price elasticity model using Ray Data for parallel processing.
Reusable for any demand-elasticity use case.
Replaces: src/ml/elasticity_model.py
"""
import numpy as np
import pandas as pd
import ray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from processors.base.train_base_stage import TrainBaseStage


class ElasticityTrainer(TrainBaseStage):
    """Trains a per-SKU LinearRegression elasticity model using Ray."""

    def __init__(self):
        self.models  = {}
        self.metrics = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def prepare_data(self, master_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        orders_df = orders_df.copy()
        orders_df.columns = orders_df.columns.str.lower()
        demand = (orders_df.groupby("sku_id")["quantity"]
                  .sum().reset_index()
                  .rename(columns={"quantity": "demand_qty"}))
        df = master_df.merge(demand, on="sku_id", how="left")
        df["demand_qty"] = df["demand_qty"].fillna(0)
        return df[df["demand_qty"] > 0][
            ["sku_id", "current_price", "demand_qty",
             "competitor_price", "inventory_bucket"]]

    def run(self, df: pd.DataFrame):
        """Train elasticity models in parallel via Ray remote tasks."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        groups  = [(sku, g) for sku, g in df.groupby("sku_id")]
        futures = [_train_sku_remote.remote(sku, g) for sku, g in groups]
        results = ray.get(futures)

        for sku, model, metric in results:
            if model is not None:
                self.models[sku]  = model
                self.metrics[sku] = metric

        print(f"Elasticity models trained for {len(self.models)} SKUs")

    def predict_optimal_price(self, row: pd.Series) -> float:
        sku = row["sku_id"]
        if sku not in self.models:
            return row["current_price"]
        model = self.models[sku]
        cp    = row["current_price"]
        best_price, best_rev = cp, 0.0
        for p in np.linspace(cp * 0.7, cp * 1.3, 20):
            demand = max(model.predict([[p]])[0], 0)
            rev    = p * demand
            if rev > best_rev:
                best_rev, best_price = rev, p
        return round(best_price, 2)

    def get_metrics(self, sku):
        m = self.metrics.get(sku)
        return (m["r2"], m["mae"], m["mape"]) if m else (None, None, None)


# ── Ray remote task (module-level so it can be pickled) ──────────────────────

@ray.remote
def _train_sku_remote(sku, g: pd.DataFrame):
    try:
        bp, bd = g["current_price"].iloc[0], g["demand_qty"].iloc[0]
        if bd <= 0:
            return sku, None, None
        prices  = np.linspace(bp * 0.7, bp * 1.3, 12)
        demands = [max(bd * (bp / p), 0) for p in prices]
        X, y    = np.array(prices).reshape(-1, 1), np.array(demands)
        model   = LinearRegression().fit(X, y)
        preds   = model.predict(X)
        metric  = {
            "r2":   round(r2_score(y, preds), 4),
            "mae":  round(mean_absolute_error(y, preds), 2),
            "mape": round(np.mean(np.abs((y - preds) / (y + 1))) * 100, 2)
        }
        return sku, model, metric
    except Exception:
        return sku, None, None
