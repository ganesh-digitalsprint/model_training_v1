"""
app/models/schemas.py
Pydantic request/response models.

REQUEST  →  PricingRequest   (sku_ids list — all features fetched from feature store)
RESPONSE →  PricingResponse  (enterprise envelope with result.data.sku_list)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SUB-MODELS
# ══════════════════════════════════════════════════════════════════════════════

class UserIdentity(BaseModel):
    user_id:   str = ""
    email:     str = ""
    name:      str = ""
    tenant_id: str = ""


class UserAuth(BaseModel):
    api_key:        str = ""
    use_idp:        bool = False
    idp_type:       str = ""
    idp_server:     Dict[str, Any] = {}
    id_token:       str = ""
    access_token:   str = ""
    internal_token: str = ""


class UserLicense(BaseModel):
    tenant_id:          str = ""
    plan:               str = ""
    subscription_level: str = ""
    rate_limit:         int = 0


class UserRBAC(BaseModel):
    roles:       List[str] = []
    permissions: List[str] = []


class UserABAC(BaseModel):
    department:     str = ""
    country:        str = ""
    groups:         List[str] = []
    security_level: str = ""


class UserInfo(BaseModel):
    identity: UserIdentity = Field(default_factory=UserIdentity)
    auth:     UserAuth     = Field(default_factory=UserAuth)
    license:  UserLicense  = Field(default_factory=UserLicense)
    rbac:     UserRBAC     = Field(default_factory=UserRBAC)
    abac:     UserABAC     = Field(default_factory=UserABAC)


class ActionInfo(BaseModel):
    actionPerformed: str = ""
    payload:         Dict[str, Any] = {}


class FileInfo(BaseModel):
    file_name:        str = ""
    file_type:        str = ""
    size:             int = 0
    content:          str = ""
    access_level:     str = "PRIVATE"
    storage_provider: str = "LOCAL"
    file_path:        str = ""


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST  — callers only supply SKU IDs; features are fetched from the store
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionParams(BaseModel):
    """
    Callers pass a list of sku_ids (1–500).
    All pricing/inventory/competitor features are resolved server-side
    from the pre-loaded feature store.
    """
    sku_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="1 to 500 SKU IDs to generate pricing predictions for",
        example=["SKU-001", "SKU-042"],
    )
    run_id: Optional[str] = Field(default=None, description="Caller-supplied run identifier")

    @field_validator("sku_ids")
    @classmethod
    def _no_blank_ids(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("sku_ids must contain at least one non-empty ID")
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped = []
        for s in cleaned:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    model_config = {"extra": "allow"}


class PricingRequest(BaseModel):
    """
    Top-level enterprise request envelope.
    Only execution_params.sku_ids is required.
    """
    correlation_id: str = Field(
        default="",
        description="Caller-supplied trace ID — echoed back verbatim in the response.",
        example="f47ac10b-58cc-4372-a567-0e02b2c3d479",
    )
    raw_text_query:   str = Field(default="", description="Optional free-text query for audit logs")
    user_info:        UserInfo    = Field(default_factory=UserInfo)
    action:           ActionInfo  = Field(default_factory=ActionInfo)
    files:            List[FileInfo] = Field(default=[], description="Attached files (pass-through)")
    execution_params: ExecutionParams = Field(
        ...,
        description="Must contain execution_params.sku_ids with at least one SKU ID",
    )
    channel_type:      str = Field(default="WEB",           example="WEB")
    source_system:     str = Field(default="ADMIN_CONSOLE", example="ADMIN_CONSOLE")
    locale:            str = Field(default="en-IN",         example="en-IN")
    request_timestamp: str = Field(default="",              example="2026-01-26T22:10:00Z")
    session_id:        str = Field(default="",              example="sess_abc123")
    client_ip:         str = Field(default="",              example="203.0.113.45")
    status:            str = Field(default="SUCCESS")


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE — flat SKU result + enterprise envelope
# ══════════════════════════════════════════════════════════════════════════════

class SKUPricingResult(BaseModel):
    """
    Flat pricing result for one SKU — all fields a pricing admin needs.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    sku_id:       str
    display_name: str

    # ── Input prices (echoed for traceability) ────────────────────────────────
    list_price:    float
    current_price: float

    # ── Competitor intelligence ───────────────────────────────────────────────
    competitor_price:     Optional[float]
    competitor_name:      Optional[str]
    competitor_bucket:    str            # WE_CHEAPER | SAME | COMPETITOR_CHEAPER | NO_COMPETITOR
    price_diff_pct:       float          # (our_price - comp_price) / comp_price × 100
    all_competitors_json: Optional[Dict[str, str]]

    # ── Inventory ─────────────────────────────────────────────────────────────
    total_inventory: float
    inventory_bucket: str               # VERY_HIGH | HIGH | MED | LOW

    # ── Demand / engagement ───────────────────────────────────────────────────
    demand_qty:     float               # predicted or historical demand
    demand_bucket:  str                 # HIGH | LOW
    # ── Pricing decision ─────────────────────────────────────────────────────
    new_price:         float            # final recommended price (guardrails applied)
    ai_optimal_price:  float            # raw model-optimal price (before guardrails)
    price_change_pct:  float            # % change from current to new_price
    confidence_score:  float            # 0.0 – 1.0 confidence in the recommendation
    confidence_label:  str              # HIGH | MEDIUM | LOW
    pricing_reason:    str              # RULE_OVERRIDE_ALERT | AI_OPTIMISED | NO_CHANGE
    ai_strategy:       str              # RULE_OVERRIDE | AI_OPTIMISED
    decision_reason:   str              # human-readable explanation for admin users
    alert_flag:        bool             # True = manual review recommended

    # ── Run metadata ──────────────────────────────────────────────────────────
    run_date:      str                  # "2026-05-07_103200"
    run_id:        str
    model_version: str


class ResultData(BaseModel):
    """Wraps the SKU list plus summary counts."""
    requested:    int
    found:        int
    missing_skus: List[str]
    sku_list:     List[SKUPricingResult]


class ResultEnvelope(BaseModel):
    """Top-level result block inside the response."""
    message: str
    action:  str = "PRICE_PREDICTED"
    summary: str
    data:    ResultData


class PricingResponse(BaseModel):
    """
    Top-level enterprise response envelope.
    """
    correlation_id:     str
    response_timestamp: str
    status:             str             # SUCCESS | PARTIAL_SUCCESS | FAILURE
    result_type:        str = "json"
    user_info:          UserInfo
    result:             ResultEnvelope
    errors:             List[Dict[str, Any]] = []
    files:              List[Any] = []
    source_citations:   List[Any] = []