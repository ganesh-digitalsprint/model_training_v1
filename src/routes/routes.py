"""
app/routers/predict.py
REST API endpoints for the Price Prediction service.

Endpoints
---------
POST /api/v1/predict
    Accepts sku_ids in execution_params.sku_ids.
    All features are resolved server-side from the feature store.

GET  /api/v1/model/info
    Returns model metadata (version, features, importances).

POST /api/v1/feature-store/reload
    Hot-reloads all raw data files without restarting the server.

GET  /api/v1/feature-store/info
    Returns summary stats about the loaded feature store.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from src.schemas.schemas import PricingRequest, PricingResponse, ResultEnvelope, ResultData
from src.service.feature_store_service import FeatureStoreService
from src.service.prediction_service import predict_batch
from src.service.model_service import ModelService, FEATURE_COLUMNS, FEATURE_IMPORTANCES

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PricingResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict optimal prices for one or more SKUs",
    description=(
        "Pass SKU IDs in execution_params.sku_ids (1–500 per request). "
        "All pricing, inventory, competitor, and GA4 features are resolved "
        "automatically from the server-side feature store. "
        "Returns the standard enterprise response envelope with "
        "result.data.sku_list containing one pricing result per SKU."
    ),
)
def predict_prices(request: PricingRequest) -> PricingResponse:
    """
    Enterprise price prediction endpoint.

    **Minimal request (single SKU):**
    ```json
    {
      "correlation_id": "abc-123",
      "execution_params": {
        "sku_ids": ["SKU-001"]
      }
    }
    ```

    **Batch request:**
    ```json
    {
      "correlation_id": "abc-456",
      "execution_params": {
        "sku_ids": ["SKU-001", "SKU-042", "SKU-099"]
      }
    }
    ```
    """
    correlation_id = request.correlation_id or str(uuid.uuid4())
    response_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id         = request.execution_params.run_id or str(uuid.uuid4())[:8]

    sku_ids = request.execution_params.sku_ids
    logger.info(
        "predict | correlation_id=%s | skus=%d | tenant=%s",
        correlation_id,
        len(sku_ids),
        request.user_info.identity.tenant_id or "unknown",
    )

    # Load model — raises if model file is missing
    try:
        model_svc = ModelService.get_instance()
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction model unavailable: {exc}",
        )

    # Run batch prediction
    try:
        results, errors = predict_batch(
            sku_ids=sku_ids,
            run_id=run_id,
            model_version=model_svc.model_version,
        )
    except Exception as exc:
        logger.exception("Unexpected error during batch prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction engine error: {exc}",
        )

    # Overall status
    n_requested  = len(sku_ids)
    n_successful = len(results)
    n_failed     = len(errors)
    failed_ids   = [e["sku_id"] for e in errors]

    if n_failed == 0:
        overall_status = "SUCCESS"
    elif n_successful > 0:
        overall_status = "PARTIAL_SUCCESS"
    else:
        overall_status = "FAILURE"

    # Summary line
    n_overrides = sum(1 for r in results if r.ai_strategy == "RULE_OVERRIDE")
    n_ai_opt    = sum(1 for r in results if r.ai_strategy == "AI_OPTIMISED")
    n_alerts    = sum(1 for r in results if r.alert_flag)
    summary_line = (
        f"{n_successful} of {n_requested} SKUs priced. "
        f"{n_ai_opt} AI optimised, {n_overrides} rule overrides, "
        f"{n_alerts} flagged for review."
    )

    # return PricingResponse(
    #     correlation_id=correlation_id,
    #     response_timestamp=response_ts,
    #     status=overall_status,
    #     result_type="json",
    #     user_info=request.user_info,
    #     result=ResultEnvelope(
    #         message=(
    #             "Pricing prediction completed successfully"
    #             if n_failed == 0
    #             else f"Pricing completed with {n_failed} failure(s)"
    #         ),
    #         action="PRICE_PREDICTED",
    #         summary=summary_line,
    #         data=ResultData(
    #             requested=n_requested,
    #             found=n_successful,
    #             missing_skus=failed_ids,
    #             sku_list=results,
    #         ),
    #     ),

    #     errors=[{"sku_id": e["sku_id"], "error": e["error"]} for e in errors],
    #     files=[],
    #     source_citations=[],
    # )
    try:
        return PricingResponse(
        correlation_id=correlation_id,
        response_timestamp=response_ts,
        status=overall_status,
        result_type="json",
        user_info=request.user_info,
        result=ResultEnvelope(
            message=(
                "Pricing prediction completed successfully"
                if n_failed == 0
                else f"Pricing completed with {n_failed} failure(s)"
            ),
            action="PRICE_PREDICTED",
            summary=summary_line,
            data=ResultData(
                requested=n_requested,
                found=n_successful,
                missing_skus=failed_ids,
                sku_list=results,
            ),
        ),
        errors=[{"sku_id": e["sku_id"], "error": e["error"]} for e in errors],
        files=[],
        source_citations=[],
    )
    except Exception as exc:
        logger.exception(f"❌ Response serialization failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Response building failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/model/info",
    status_code=status.HTTP_200_OK,
    summary="Get model metadata",
)
def get_model_info() -> dict:
    """Return metadata about the loaded prediction model."""
    try:
        svc = ModelService.get_instance()
        return {
            "model_version":       svc.model_version,
            "model_type":          type(svc.model).__name__,
            "model_path":          svc.model_path,
            "feature_columns":     FEATURE_COLUMNS,
            "target_column":       "demand_qty",
            "feature_importances": FEATURE_IMPORTANCES,
            "description": (
                "XGBoost regressor predicting demand from price, competitor_price, "
                "inventory_level, and price_diff_pct. "
                "Candidate prices are scanned to maximise estimated revenue (price × demand)."
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model info unavailable: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature store endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/feature-store/reload",
    status_code=status.HTTP_200_OK,
    summary="Hot-reload feature store from raw data files",
    description="Re-reads all raw CSVs/XLSXs/JSONs without restarting the server.",
)
def reload_feature_store() -> dict:
    """Trigger a hot-reload of the feature store."""
    try:
        store = FeatureStoreService.reload()
        return {
            "status": "reloaded",
            "sku_count": len(store),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as exc:
        logger.exception("Feature store reload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature store reload failed: {exc}",
        )


@router.get(
    "/feature-store/info",
    status_code=status.HTTP_200_OK,
    summary="Feature store summary",
    description="Returns the number of SKUs loaded and the first 20 SKU IDs.",
)
def feature_store_info() -> dict:
    """Return summary stats about the in-memory feature store."""
    try:
        store   = FeatureStoreService.get_instance()
        all_ids = store.all_sku_ids()
        return {
            "sku_count":    len(all_ids),
            "sample_skus":  all_ids[:20],
            "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Feature store info unavailable: {exc}",
        )