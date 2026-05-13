"""
app/main.py
FastAPI application entry point for the Price Prediction API.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes import routes as predict_router_module
import logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ecommerce Price Prediction API",
    description=(
        "Predicts optimal prices for SKUs using a trained XGBoost model. "
        "Pass only SKU IDs — all features (price, inventory, competitors, GA4) "
        "are resolved automatically from the server-side feature store."
    ),
    version="3.0.0",
)

# Allow cross-origin requests (lock down allowed_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register prediction + feature-store routes under /api/v1
app.include_router(predict_router_module.router, prefix="/api/v1", tags=["Predictions"])


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe for load balancers / k8s."""
    return {"status": "ok", "service": "price-prediction-api", "version": "3.0.0"}


@app.on_event("startup")
def startup_event():
    """
    Pre-load both the ML model and the feature store at startup so the
    first request is instant and any missing-file errors surface immediately.
    """
    from src.service.model_service import ModelService
    from src.service.feature_store_service import FeatureStoreService

    logger.info("=== Price Prediction API startup ===")

    logger.info("Loading prediction model...")
    try:
        svc = ModelService.get_instance()
        logger.info("Model loaded: %s", svc.model_version)
    except FileNotFoundError as exc:
        logger.error("Model file not found — /api/v1/predict will return 503 until fixed. %s", exc)

    logger.info("Loading feature store from raw data files...")
    try:
        store = FeatureStoreService.get_instance()
        logger.info("Feature store ready: %d SKUs loaded.", len(store))
    except Exception as exc:
        logger.error("Feature store failed to load — predictions will fail. %s", exc)

    logger.info("=== Startup complete ===")