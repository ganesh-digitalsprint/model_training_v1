"""
inference.py
Demand prediction with Ray Serve for scalable model serving.
Replaces: src/ml/inference.py
"""
import pandas as pd
import joblib
import mlflow.pyfunc
import ray
from ray import serve

FEATURE_COLUMNS = ["price", "competitor_price", "inventory_level", "price_diff_pct"]
_LOCAL_MODEL_CACHE = None


# ── Standard (non-Serve) load ─────────────────────────────────────────────────

def load_model(model_name: str = "linear", version: str = "v1.0",
               use_registry: bool = True, model_dir: str = "data/raw/ml/models"):
    global _LOCAL_MODEL_CACHE
    if _LOCAL_MODEL_CACHE is not None:
        return _LOCAL_MODEL_CACHE

    if use_registry:
        try:
            uri = f"models:/dynamic_pricing_{model_name}/Production"
            print(f"Loading Production model from MLflow Registry: {uri}")
            _LOCAL_MODEL_CACHE = mlflow.pyfunc.load_model(uri)
            return _LOCAL_MODEL_CACHE
        except Exception as e:
            print(f"Registry load failed, falling back to local: {e}")

    import os
    path = os.path.join(model_dir, f"{model_name}_{version}.pkl")
    print(f"Loading local model: {path}")
    _LOCAL_MODEL_CACHE = joblib.load(path)
    return _LOCAL_MODEL_CACHE


def reload_model():
    global _LOCAL_MODEL_CACHE
    _LOCAL_MODEL_CACHE = None
    print("Model cache cleared")


def predict_demand(model, price, competitor_price, inventory_level,
                   price_diff_pct) -> float:
    competitor_price  = price if (competitor_price is None or pd.isna(competitor_price)) else competitor_price
    inventory_level   = 0    if (inventory_level is None   or pd.isna(inventory_level))  else inventory_level
    price_diff_pct    = 0    if (price_diff_pct is None    or pd.isna(price_diff_pct))   else price_diff_pct

    X = pd.DataFrame([{"price": price, "competitor_price": competitor_price,
                        "inventory_level": inventory_level,
                        "price_diff_pct": price_diff_pct}])
    return model.predict(X)[0]


# ── Ray Serve deployment ──────────────────────────────────────────────────────

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
class DemandPredictorServe:
    """
    Ray Serve deployment for real-time demand inference.
    Replaces in-process model cache with a scalable HTTP-accessible service.
    """

    def __init__(self, model_name: str = "xgboost", version: str = "v1.0"):
        self.model = load_model(model_name, version)
        print(f"DemandPredictorServe ready — model={model_name} v{version}")

    async def __call__(self, request):
        body  = await request.json()
        price = body["price"]
        pred  = predict_demand(
            self.model,
            price,
            body.get("competitor_price"),
            body.get("inventory_level", 0),
            body.get("price_diff_pct", 0))
        return {"predicted_demand": float(pred)}


def deploy_serve(model_name: str = "xgboost", version: str = "v1.0"):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    serve.start(detached=True)
    DemandPredictorServe.deploy(model_name, version)
    print("Ray Serve deployment started at http://localhost:8000/DemandPredictorServe")
