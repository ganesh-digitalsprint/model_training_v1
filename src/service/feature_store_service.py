"""
app/services/feature_store_service.py

Loads all raw data files at startup, runs feature engineering
(demand / inventory / competitor), then builds the master dataset
and exposes a per-SKU lookup via FeatureStoreService.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── data directory ────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "src/data"))
logger.info("DATA_DIR set to: %s", DATA_DIR.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Feature row stored for each SKU
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SKUFeatures:
    sku_id: str
    display_name: str = ""
    list_price: float = 0.0
    current_price: float = 0.0
    total_inventory: float = 0.0
    inventory_bucket: str = "LOW"
    demand_qty: float = 0.0
    demand_bucket: str = "LOW"
    competitor_price: Optional[float] = None
    competitor_name: Optional[str] = None
    competitor_bucket: str = "NO_COMPETITOR"
    price_diff_pct: Optional[float] = None
    all_competitors_json: Optional[Dict[str, str]] = field(default=None)
    ga4_views: int = 0
    ga4_add_to_cart: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _is_numeric(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


_SKU_ID_ALIASES = {
    "sku": "sku_id",
    "skuid": "sku_id",
    "sku_code": "sku_id",
    "product_id": "sku_id",
    "productid": "sku_id",
    "item_id": "sku_id",
    "itemid": "sku_id",
    "product_sku": "sku_id",
    "article_id": "sku_id",
    "style_id": "sku_id",
    "variant_id": "sku_id",
    "catalog_ref_id": "sku_id",
    "id": "sku_id",
}


def _normalize_sku_id_col(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "sku_id" in df.columns:
        return df
    df = df.rename(columns=_SKU_ID_ALIASES)
    if "sku_id" not in df.columns:
        logger.warning(
            "%s: could not find a sku_id column. Columns present: %s",
            source, list(df.columns),
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Raw-data loaders  (unchanged from original — they return raw DataFrames /
# dicts so the feature builders can work on them)
# ─────────────────────────────────────────────────────────────────────────────

def _load_sku_master() -> pd.DataFrame:
    path = DATA_DIR / "sku.csv"
    logger.info("Loading SKU master from: %s", path)
    if not path.exists():
        logger.warning("sku.csv not found — display_name will be sku_id")
        return pd.DataFrame(columns=["sku_id", "display_name"])
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    df = _normalize_sku_id_col(df, "sku.csv")
    df = df.rename(columns={"name": "display_name", "product_name": "display_name", "title": "display_name"})
    if "display_name" not in df.columns:
        df["display_name"] = df.get("sku_id", df.iloc[:, 0])
    logger.info("✓ Loaded %d SKUs from sku.csv", len(df))
    return df[["sku_id", "display_name"]].dropna(subset=["sku_id"])


def _load_prices() -> pd.DataFrame:
    path = DATA_DIR / "price.csv"
    logger.info("Loading prices from: %s", path)
    if not path.exists():
        logger.warning("price.csv not found — prices will be 0")
        return pd.DataFrame(columns=["sku_id", "list_price", "current_price"])
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = _normalize_sku_id_col(df, "price.csv")
    df = df.rename(columns={"price": "list_price", "mrp": "list_price",
                             "selling_price": "current_price", "sale_price": "current_price"})
    for col in ["list_price", "current_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "current_price" not in df.columns:
        df["current_price"] = df.get("list_price", 0.0)
    logger.info("✓ Loaded %d price records", len(df))
    cols = ["sku_id"] + [c for c in ["list_price", "current_price"] if c in df.columns]
    return df[cols]


def _load_inventory_raw() -> pd.DataFrame:
    path = DATA_DIR / "inventory.xlsx"
    if not path.exists():
        return pd.DataFrame(columns=["sku_id", "stock_level"])
    
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]       # dedup before rename
    df = _normalize_sku_id_col(df, "inventory.xlsx")
    df = df.loc[:, ~df.columns.duplicated()]       # ← dedup AGAIN after rename
    
    logger.info("inventory.xlsx columns after normalize: %s", list(df.columns))
    return df


def _load_orders_raw() -> pd.DataFrame:
    path = DATA_DIR / "purchased_order.xlsx"
    if not path.exists():
        return pd.DataFrame(columns=["sku_id", "quantity"])
    
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]       # dedup before rename
    df = _normalize_sku_id_col(df, "purchased_order.xlsx")
    df = df.loc[:, ~df.columns.duplicated()]       # ← dedup AGAIN after rename
    
    logger.info("purchased_order.xlsx columns after normalize: %s", list(df.columns))
    return df


def _load_competitor_pricing_raw():
    """Return raw JSON (list or dict) from competitor_pricing.json."""
    path = DATA_DIR / "competitor_pricing.json"
    logger.info("Loading competitor pricing from: %s", path)
    if not path.exists():
        logger.warning("competitor_pricing.json not found")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    logger.info("✓ Loaded competitor pricing JSON")
    return raw


def _load_ga4_events_raw():
    """Return raw GA4 JSON (list of events or pre-aggregated dict)."""
    path = DATA_DIR / "ga4_events.json"
    logger.info("Loading GA4 events from: %s", path)
    if not path.exists():
        logger.warning("ga4_events.json not found")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    logger.info("✓ Loaded GA4 events JSON")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Feature builders  (your original logic, column-mismatch fixes applied)
# ─────────────────────────────────────────────────────────────────────────────

def build_demand_features(orders: pd.DataFrame, ga4) -> pd.DataFrame:
    """
    Returns DataFrame with columns: sku_id, demand_bucket

    FIX — orders column: your file may call it 'quantity' or any alias;
    we detect whichever is present instead of hard-coding 'quantity'.
    """
    orders = orders.copy()
    orders.columns = orders.columns.str.lower()

    # ── detect the quantity column ────────────────────────────────────────────
    qty_aliases = ["quantity", "qty", "order_qty", "units_sold", "quantity_ordered", "demand_qty"]
    qty_col = next((c for c in qty_aliases if c in orders.columns), None)

    if qty_col is None:
        logger.warning("build_demand_features: no quantity column found in orders. "
                       "Columns present: %s", list(orders.columns))
        # Fall back to the first numeric column that isn't sku_id
        num_cols = [c for c in orders.select_dtypes(include="number").columns if c != "sku_id"]
        if num_cols:
            qty_col = num_cols[0]
            logger.info("build_demand_features: using '%s' as quantity column", qty_col)
        else:
            # Can't compute sales; return empty demand
            return pd.DataFrame(columns=["sku_id", "demand_bucket"])

    # ── total sales per SKU ───────────────────────────────────────────────────
    sales = (
        orders.groupby("sku_id")[qty_col]
        .sum()
        .reset_index()
    )
    sales.columns = ["sku_id", "total_sales"]

    # ── GA4: handle both pre-aggregated dict and raw event list ──────────────
    views, carts = [], []

    if isinstance(ga4, dict):
        # Pre-aggregated  { sku_id: { "view_item": N, "add_to_cart": M } }
        for sku, counts in ga4.items():
            views.extend([sku] * int(counts.get("view_item", 0)))
            carts.extend([sku] * int(counts.get("add_to_cart", 0)))
    else:
        # Raw event list  [{ "event_name": "view_item", "items": [{ "item_id": "..." }] }]
        for e in ga4:
            items = e.get("items", [])
            # items might be missing (some events store item_id at top level)
            if not items:
                item_id = (
                    e.get("item_id")
                    or e.get("sku_id")
                    or (e.get("event_params") or {}).get("item_id")
                )
                if item_id:
                    items = [{"item_id": item_id}]

            for item in items:
                sku = item.get("item_id", item.get("sku_id", ""))
                if not sku:
                    continue
                if e["event_name"] == "view_item":
                    views.append(sku)
                elif e["event_name"] == "add_to_cart":
                    carts.append(sku)

    views_df = pd.Series(views).value_counts().rename_axis("sku_id").reset_index(name="views")
    carts_df = pd.Series(carts).value_counts().rename_axis("sku_id").reset_index(name="add_to_cart")

    demand = (
        sales
        .merge(views_df, on="sku_id", how="left")
        .merge(carts_df, on="sku_id", how="left")
        .fillna(0)
    )

    # Normalise 0-1
    for c in ["total_sales", "views", "add_to_cart"]:
        if demand[c].max() > 0:
            demand[c] = demand[c] / demand[c].max()

    demand["score"] = (
        0.5 * demand["total_sales"]
        + 0.3 * demand["add_to_cart"]
        + 0.2 * demand["views"]
    )

    def bucket(x):
        if x > 0.7:
            return "HIGH"
        if x > 0.4:
            return "MED"
        return "LOW"

    demand["demand_bucket"] = demand["score"].apply(bucket)

    logger.info("✓ build_demand_features: %d SKUs", len(demand))
    return demand[["sku_id", "demand_bucket"]]


def build_inventory_features(inv: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns: sku_id, total_inventory, inventory_bucket

    FIX — stock column detection: checks 'stock_level' then 'stock' then any
    numeric column, so it never silently returns zeros.
    """
    inv = inv.copy()
    inv.columns = inv.columns.str.lower()

    # ── detect the stock column ───────────────────────────────────────────────
    stock_aliases = ["stock_level", "stock", "stock_qty", "quantity_on_hand",
                     "inventory", "qty", "total_inventory"]
    stock_col = next((c for c in stock_aliases if c in inv.columns), None)

    if stock_col is None:
        num_cols = [c for c in inv.select_dtypes(include="number").columns if c != "sku_id"]
        if num_cols:
            stock_col = num_cols[0]
            logger.info("build_inventory_features: using '%s' as stock column", stock_col)
        else:
            logger.warning("build_inventory_features: no stock column found. Columns: %s",
                           list(inv.columns))
            return pd.DataFrame(columns=["sku_id", "total_inventory", "inventory_bucket"])

    inv[stock_col] = pd.to_numeric(inv[stock_col], errors="coerce").fillna(0)

    inv_agg = (
        inv.groupby("sku_id")[stock_col]
        .sum()
        .reset_index()
    )
    inv_agg.columns = ["sku_id", "total_inventory"]

    def bucket(q):
        if q > 1000:
            return "VERY_HIGH"
        if q > 400:
            return "HIGH"
        if q > 100:
            return "MED"
        return "LOW"

    inv_agg["inventory_bucket"] = inv_agg["total_inventory"].apply(bucket)

    logger.info("✓ build_inventory_features: %d SKUs", len(inv_agg))
    return inv_agg  # sku_id, total_inventory, inventory_bucket


def build_competitor_features(price_df,sku_df,comp_data):

    price_df = price_df.copy()
    price_df.columns = price_df.columns.str.strip().str.lower()

    #sku_df = pd.read_csv("data/sku.csv")
    sku_df.columns = sku_df.columns.str.strip().str.lower()

    #with open("data/competitor_pricing.json") as f:
    #    comp_data = json.load(f)

    if isinstance(comp_data, dict) and "results" in comp_data:
        comp_items = comp_data["results"]
    else:
        comp_items = comp_data

    rows = []

    for item in comp_items:

        name = item.get("name")
        summary = item.get("summary", {})

        min_price = None
        min_site = None

        # competitor dict
        comp_dict = {}

        for site, data in summary.items():
            if data and data.get("price"):
                try:
                    price = float(data["price"])

                    # store competitor price map
                    comp_dict[site] = str(round(price, 2))

                    # find lowest competitor
                    if min_price is None or price < min_price:
                        min_price = price
                        min_site = site
                except:
                    pass

        rows.append({
            "display_name": name,
            "competitor_price": min_price,
            "competitor_name": min_site,
            "all_competitors_json": json.dumps(comp_dict)
        })

    comp_df = pd.DataFrame(rows)

    # map sku
    comp_df = comp_df.merge(
        sku_df[["sku_id","display_name"]],
        on="display_name",
        how="left"
    )

    # merge our price
    comp_df = comp_df.merge(
        price_df[["sku_id","list_price"]],
        on="sku_id",
        how="left"
    )

    # bucket logic
    def bucket(row):
        if pd.isna(row["competitor_price"]):
            return "NO_COMPETITOR"
        if row["competitor_price"] < row["list_price"]:
            return "COMPETITOR_CHEAPER"
        if row["competitor_price"] > row["list_price"]:
            return "WE_CHEAPER"
        return "SAME"

    comp_df["competitor_bucket"] = comp_df.apply(bucket, axis=1)

    comp_df["price_diff_pct"] = ((
        (comp_df["list_price"] - comp_df["competitor_price"])
        / comp_df["competitor_price"]
    ) * 100).round(2)

    return comp_df[[
        "sku_id",
        "competitor_price",
        "competitor_name",
        "competitor_bucket",
        "price_diff_pct",
        "all_competitors_json"
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# Master dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_master_dataset(
    sku_df: pd.DataFrame,
    price_df: pd.DataFrame,
    inventory_raw: pd.DataFrame,
    orders_raw: pd.DataFrame,
    ga4_data,
    comp_data,
) -> pd.DataFrame:
    """
    Runs all feature builders and joins them into a single master DataFrame.

    Columns in the result
    ─────────────────────
    sku_id, display_name, list_price, current_price,
    total_inventory, inventory_bucket,
    demand_bucket,
    competitor_price, competitor_name, competitor_bucket,
    price_diff_pct, all_competitors_json
    """
    logger.info("=" * 60)
    logger.info("build_master_dataset: starting feature engineering")

    demand_df    = build_demand_features(orders_raw, ga4_data)
    inventory_df = build_inventory_features(inventory_raw)
    comp_df      = build_competitor_features(price_df, sku_df, comp_data)

    logger.info("Merging into master…")

    # MASTER DATASET
    master = price_df.merge(sku_df[["sku_id", "display_name"]], on="sku_id", how="left")
    master = master.merge(demand_df,    on="sku_id", how="left")
    master = master.merge(inventory_df, on="sku_id", how="left")
    master = master.merge(comp_df,      on="sku_id", how="left")

    # current_price defaults to list_price if not set separately
    if "current_price" not in master.columns:
        master["current_price"] = master["list_price"]
    else:
        mask = master["current_price"].isna() | (master["current_price"] == 0)
        master.loc[mask, "current_price"] = master.loc[mask, "list_price"]

    # Remove SKUs with no stock (can't sell them)
    master = master[master["total_inventory"].fillna(0) > 0]

    # Fill categorical buckets
    master["demand_bucket"]     = master["demand_bucket"].fillna("LOW")
    master["inventory_bucket"]  = master["inventory_bucket"].fillna("LOW")
    master["competitor_bucket"] = master["competitor_bucket"].fillna("NO_COMPETITOR")

    logger.info("build_master_dataset: %d SKUs in final master", len(master))
    logger.info("Columns: %s", list(master.columns))
    logger.info("=" * 60)
    return master


# ─────────────────────────────────────────────────────────────────────────────
# SKUFeatures dataclass — populated from master dataset
# ─────────────────────────────────────────────────────────────────────────────

class FeatureStoreService:
    """
    Singleton that owns the in-memory per-SKU feature lookup.
    Call  FeatureStoreService.get_instance()  anywhere in the app.
    """

    _instance: Optional["FeatureStoreService"] = None

    def __init__(self) -> None:
        self._store: Dict[str, SKUFeatures] = {}
        self._load()

    @classmethod
    def get_instance(cls) -> "FeatureStoreService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reload(cls) -> "FeatureStoreService":
        """Force a fresh load from disk."""
        cls._instance = cls()
        return cls._instance

    def get(self, sku_id: str) -> Optional[SKUFeatures]:
        return self._store.get(sku_id) or self._store.get(sku_id.lower())

    def all_sku_ids(self) -> List[str]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        logger.info("=" * 80)
        logger.info("FeatureStoreService: loading from %s", DATA_DIR.resolve())
        logger.info("=" * 80)

        # 1. Load raw data
        sku_df        = _load_sku_master()
        price_df      = _load_prices()
        inventory_raw = _load_inventory_raw()
        orders_raw    = _load_orders_raw()
        comp_data     = _load_competitor_pricing_raw()
        ga4_data      = _load_ga4_events_raw()

        # 2. Build master dataset via feature engineering
        master = build_master_dataset(
            sku_df, price_df, inventory_raw, orders_raw, ga4_data, comp_data
        )

        # 3. Populate the store from master rows
        self._store = {}
        for _, row in master.iterrows():
            sid = str(row.get("sku_id", "")).strip()
            if not sid:
                continue

            # all_competitors_json arrives as a JSON string from comp_df
            raw_comp = row.get("all_competitors_json")
            if isinstance(raw_comp, str):
                try:
                    comp_dict = json.loads(raw_comp)
                except (json.JSONDecodeError, TypeError):
                    comp_dict = None
            else:
                comp_dict = raw_comp  # already dict or None

            self._store[sid] = SKUFeatures(
                sku_id=sid,
                display_name=str(row.get("display_name", sid)).strip() or sid,
                list_price=float(row.get("list_price", 0.0) or 0.0),
                current_price=float(row.get("current_price", row.get("list_price", 0.0)) or 0.0),
                total_inventory=float(row.get("total_inventory", 0.0) or 0.0),
                inventory_bucket=str(row.get("inventory_bucket", "LOW")),
                demand_qty=0.0,          # raw qty not in master; use demand_bucket instead
                demand_bucket=str(row.get("demand_bucket", "LOW")),
                competitor_price=(
                    float(row["competitor_price"])
                    if pd.notna(row.get("competitor_price")) else None
                ),
                competitor_name=(
                    str(row["competitor_name"])
                    if pd.notna(row.get("competitor_name")) else None
                ),
                competitor_bucket=str(row.get("competitor_bucket", "NO_COMPETITOR")),
                price_diff_pct=(
                    float(row["price_diff_pct"])
                    if pd.notna(row.get("price_diff_pct")) else None
                ),
                all_competitors_json=comp_dict,
                ga4_views=0,       # GA4 signals folded into demand_bucket score
                ga4_add_to_cart=0,
            )

        logger.info(
            "FeatureStoreService: ready — %d SKUs loaded",
            len(self._store),
        )
        logger.info("=" * 80)