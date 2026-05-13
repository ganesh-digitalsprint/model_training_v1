"""
model_factory.py
Creates sklearn / XGBoost / LightGBM models from name string.
Reusable across ANY training job.
Replaces: src/ml/model_factory.py
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


class ModelFactory:

    @staticmethod
    def create(model_name: str, params: dict = None):
        params = params or {}
        if model_name == "linear":
            return LinearRegression()
        elif model_name == "rf":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                random_state=42)
        elif model_name == "xgboost":
            return XGBRegressor(
                n_estimators=params.get("n_estimators", 300),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 5),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                random_state=42)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def load_production_model(model_name: str):
    uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(uri)


def load_champion_challenger_models(model_name: str = "xgboost"):
    registered = f"dynamic_pricing_{model_name}"
    try:
        champion   = mlflow.pyfunc.load_model(f"models:/{registered}/Production")
        challenger = mlflow.pyfunc.load_model(f"models:/{registered}/Staging")
        return champion, challenger
    except Exception as e:
        print(f"Champion/Challenger models not available: {e}")
        return None, None
