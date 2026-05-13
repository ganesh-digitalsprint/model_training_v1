"""
constants.py
Framework-wide model tags and metric names.
"""

# FEATURE_COLS     = ["price", "competitor_price", "inventory_level", "price_diff_pct"]
# TARGET_COL       = "demand_qty"

MODEL_TAG_DOMAIN = "dynamic_pricing"
METRIC_R2 = "r2"
METRIC_MAE = "mae"
METRIC_MAPE = "mape"

STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"

XGBOOST = "xgboost"
VERSION = "v1.0"
# RAY_DIR="D:/ray_tmp"
DELTA_LAKE = "delta_lake"
GOLDEN_DATASET_PATH = "golden_dataset_path"
EXPECTED_COLUMNS = "expected_columns"
# GOLDEN_DATA_FALLBACK_PATH="data/raw/data/golden_training_data.csv"
# TEST_SIZE=0.2
# RANDOM_STATE_SIZE=42
ML_TRAINING = "ml_training"
CFG_TRAINING_DATA = "training_data"
CFG_RAY = "ray"
CFG_RAY_TEMP_DIR = "temp_dir"
CFG_CSV_FALLBACK_PATH = "csv_fallback_path"
CFG_TEST_SIZE = "test_size"
CFG_RANDOM_STATE = "random_state"

CFG_FEATURE_COLUMNS = "feature_columns"
CFG_TARGET_COLUMN = "target_column"
ENABLE_HYPERPARAMETER_TUNING = "enable_hyperparameter_tuning"
TUNING_METHOD = "tuning_method"
RAY_TUNE = "ray_tune"
TUNING_TRIALS = "tuning_trials"
RANDOM_FOREST = "rf"
SKLEARN = "sklearn"
# ── train_executor — local model storage ──────────────────────────────────────
CFG_MODEL_LOCAL_PATH = "model_local_path"
DEFAULT_MODEL_LOCAL_PATH = "D:/delta_tables/model_registry/ml"
CFG_TOLERANCE = "tolerance"
# ── train_executor — MLflow param log keys ────────────────────────────────────
PARAM_MODEL_NAME = "model_name"
PARAM_MODEL_VERSION = "model_version"
PARAM_DATASET_ROWS = "dataset_rows"
PARAM_FEATURES = "features"
PARAM_TARGET = "target"
PARAM_TRAIN_SIZE = "train_size"
PARAM_TEST_SIZE = "test_size"
PARAM_TUNING_METHOD = "tuning_method"
PARAM_DATASET_SOURCE = "dataset_source"
PARAM_TUNING_NONE = "none"

# ── train_executor — artifact filenames ───────────────────────────────────────
ARTIFACT_ACTUAL_VS_PRED = "actual_vs_pred.png"
ARTIFACT_FEATURE_IMPORTANCE = "feature_importance.png"

# ── train_executor — chart labels ────────────────────────────────────────────
CHART_LABEL_ACTUAL = "Actual"
CHART_LABEL_PREDICTED = "Predicted"
CHART_TITLE_ACTUAL_VS_PRED = "Actual vs Predicted"
CHART_TITLE_FEATURE_IMPORTANCE = "Feature Importance"

# ── train_executor — metric keys ──────────────────────────────────────────────
METRIC_RMSE = "rmse"
METRIC_PSEUDO_ACC = "pseudo_accuracy"

# ── train_executor — result dict keys ────────────────────────────────────────
RESULT_RMSE = "rmse"
RESULT_PSEUDO_ACCURACY = "pseudo_accuracy"
RESULT_REGISTERED_VERSION = "registered_version"
RESULT_MLFLOW_REGISTRY = "mlflow_registry"
RESULT_DELTA_WRITE = "delta_write"
RESULT_DELTA_WRITE_SUCCESS = "success"

# ── train_launcher — job name literals ────────────────────────────────────────
JOB_TRAINING = "gap_analysis_training"
JOB_PRICING = "price_prediction"

# ── train_launcher — environment choices ──────────────────────────────────────
ENV_DEV = "dev"
ENV_QA = "qa"
ENV_PROD = "prod"

# ── train_launcher — config keys used at dispatch level ───────────────────────
CFG_MODEL = "model"
CFG_MODEL_VERSION = "model_version"
CFG_PRICING_MODE = "pricing_mode"
CFG_ENVIRONMENT = "environment"
CFG_PRICING = "pricing"
CFG_ML = "ml"
# ── pricing_executor — config section keys ────────────────────────────────────
CFG_GOLDEN_CSV_FALLBACK = "golden_csv_fallback"
CFG_HYBRID = "hybrid"
CFG_MIN_R2_FOR_AI = "min_r2_for_ai"
CFG_MAX_MAPE_FOR_AI = "max_mape_for_ai"

# ── pricing_executor — hybrid threshold defaults ───────────────────────────────
DEFAULT_MIN_R2_FOR_AI = 0.75
DEFAULT_MAX_MAPE_FOR_AI = 20.0

# ── pricing_executor — result summary keys ────────────────────────────────────
RESULT_STATUS = "status"
RESULT_MODE = "mode"
RESULT_MODEL_NAME = "model_name"
RESULT_MODEL_VERSION = "model_version"
RESULT_ENVIRONMENT = "environment"
RESULT_ROWS_PROCESSED = "rows_processed"
RESULT_ELAPSED_SECONDS = "elapsed_seconds"
RESULT_STATUS_SUCCESS = "success"

# ── pricing_executor — environment default ────────────────────────────────────
DEFAULT_ENVIRONMENT = "dev"

# ── Fallback defaults ─────────────────────────────────────────────────────────
DEFAULT_VERSION = "v1.0"
DEFAULT_TUNING_TRIALS = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_NAME = "unknown"
DEFAULT_MODEL_VERSION = "unknown"
MODE_RULE = "rule"
MODE_HYBRID = "hybrid"
MODE_AI = "ai"
# ── DataFrame column names — input ────────────────────────────────────────────
COL_SKU_ID = "sku_id"
COL_CURRENT_PRICE = "current_price"
COL_COMPETITOR_PRICE = "competitor_price"
COL_COMPETITOR_NAME = "competitor_name"
COL_PRICE_DIFF_PCT = "price_diff_pct"
COL_DEMAND_QTY = "demand_qty"
COL_TOTAL_INVENTORY = "total_inventory"
COL_INVENTORY_LEVEL = "inventory_level"
COL_ALERT_FLAG = "alert_flag"

# bucket columns
COL_DEMAND_BUCKET = "demand_bucket"
COL_INVENTORY_BUCKET = "inventory_bucket"
COL_COMPETITOR_BUCKET = "competitor_bucket"

# elasticity columns
COL_ELASTICITY_R2 = "elasticity_r2"
COL_ELASTICITY_MAPE = "elasticity_mape"

# ── DataFrame column names — output (pricing engines) ─────────────────────────
COL_NEW_PRICE = "new_price"
COL_DECISION_REASON = "decision_reason"
COL_AI_OPTIMAL_PRICE = "ai_optimal_price"
COL_PRICING_REASON = "pricing_reason"
COL_AI_STRATEGY = "ai_strategy"
COL_PRICE_CHANGE_PCT = "price_change_pct"
COL_FINAL_PRICE = "final_price"
COL_PRICING_STRATEGY = "pricing_strategy"

# champion-challenger output columns
COL_CHAMPION_DEMAND = "champion_demand"
COL_CHALLENGER_DEMAND = "challenger_demand"
COL_CHAMPION_REVENUE = "champion_revenue"
COL_CHALLENGER_REVENUE = "challenger_revenue"

# ── DataFrame column names — metadata ─────────────────────────────────────────
COL_RUN_DATE = "run_date"
COL_MODEL_VERSION = "model_version"
COL_RUN_ID = "run_id"

# ── DataFrame column names — simulator outputs ────────────────────────────────
COL_CURRENT_REVENUE = "current_revenue"
COL_SIMULATED_REVENUE = "simulated_revenue"
COL_RULE_REVENUE = "rule_revenue"
COL_AI_REVENUE = "ai_revenue"
COL_REVENUE_CHANGE = "revenue_change"
COL_REVENUE_CHANGE_PCT = "revenue_change_pct"
COL_REVENUE_LIFT = "revenue_lift"
COL_REVENUE_LIFT_PCT = "revenue_lift_pct"
COL_NEW_REVENUE = "new_revenue"

# ── Bucket value literals — inventory ─────────────────────────────────────────
BUCKET_INV_VERY_HIGH = "VERY_HIGH"
BUCKET_INV_HIGH = "HIGH"
BUCKET_INV_MED = "MED"
BUCKET_INV_LOW = "LOW"

# ── Bucket value literals — demand ────────────────────────────────────────────
BUCKET_DEMAND_HIGH = "HIGH"
BUCKET_DEMAND_LOW = "LOW"

# ── Bucket value literals — competitor ───────────────────────────────────────
BUCKET_COMP_NO_COMPETITOR = "NO_COMPETITOR"
BUCKET_COMP_WE_CHEAPER = "WE_CHEAPER"
BUCKET_COMP_SAME = "SAME"
BUCKET_COMP_COMPETITOR_CHEAPER = "COMPETITOR_CHEAPER"

# ── Pricing reason literals ───────────────────────────────────────────────────
REASON_RULE_OVERRIDE_ALERT = "RULE_OVERRIDE_ALERT"
REASON_NO_IMPROVEMENT = "NO_IMPROVEMENT"
REASON_AI_REVENUE_OPTIMIZED = "AI_REVENUE_OPTIMIZED"
REASON_REJECTED_LOW_REVENUE = "REJECTED_LOW_REVENUE"
REASON_RULE_OVERRIDE = "RULE_OVERRIDE"
REASON_AI_OPTIMIZED = "AI_OPTIMIZED"
REASON_MANUAL_REVIEW_ALERT = "Manual review (alert triggered)"
REASON_AI_PRICING = "AI optimized pricing"
REASON_RULE_PRICING = "Rule-based pricing"

# ── Simulator summary dict keys ───────────────────────────────────────────────
SUMMARY_CURRENT_REVENUE = "current_revenue"
SUMMARY_NEW_REVENUE = "new_revenue"
SUMMARY_RULE_REVENUE = "rule_revenue"
SUMMARY_AI_REVENUE = "ai_revenue"
SUMMARY_REVENUE_CHANGE = "revenue_change"
SUMMARY_REVENUE_CHANGE_PCT = "revenue_change_pct"
SUMMARY_REVENUE_LIFT = "revenue_lift"
SUMMARY_LIFT_PCT = "lift_pct"
SUMMARY_CHAMPION_REVENUE = "champion_revenue"
SUMMARY_CHALLENGER_REVENUE = "challenger_revenue"

# ── MLflow dashboard literals ─────────────────────────────────────────────────
MLFLOW_EXPERIMENT_PRICING = "pricing_strategy_comparison"
MLFLOW_ARTIFACT_PRICING_CHART = "pricing_comparison.png"
DASHBOARD_LABEL_CURRENT = "Current"
DASHBOARD_LABEL_RULE = "Rule"
DASHBOARD_LABEL_AI = "AI"

# ── Run metadata keys ─────────────────────────────────────────────────────────
META_TIMESTAMP = "timestamp"
META_MODEL_VERSION = "model_version"
META_RUN_ID = "run_id"
META_TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"

# ── Output writer defaults ────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "data/output"

# ── Training pipeline constants ───────────────────────────────────────────────
FEATURE_COLS = ["price", "competitor_price", "inventory_level", "price_diff_pct"]
TARGET_COL = "demand_qty"
RAY_DIR = "D:/ray_tmp"
DELTA_LAKE = "delta_lake"
GOLDEN_DATASET_PATH = "golden_dataset_path"
EXPECTED_COLUMNS = "expected_columns"
GOLDEN_DATA_FALLBACK_PATH = "data/raw/data/golden_training_data.csv"
TEST_SIZE = 0.2
RANDOM_STATE_SIZE = 42
ML_TRAINING = "ml_training"

# ── train_executor — mlflow setup keys ───────────────────────────────────────
CFG_MLFLOW_TRACKING_URI = "tracking_uri"
CFG_EXPERIMENT_NAME = "experiment_name"

# ── train_executor — extra mlflow param log keys ──────────────────────────────
PARAM_PRICING_STRATEGY = "pricing_strategy"
PARAM_TOLERANCE = "tolerance"
PARAM_DATASET_NAME = "dataset_name"
PARAM_DATASET_VERSION = "dataset_version"
PARAM_FEATURES_USED = "features_used"

# ── train_executor — dataset defaults ─────────────────────────────────────────
DEFAULT_DATASET_NAME = "golden_training_data"
DEFAULT_DATASET_VERSION = "v1"

# ── train_executor — mlflow artifact path ─────────────────────────────────────
MLFLOW_ARTIFACT_MODEL = "model"
CFG_MLFLOW = "mlflow"
LINEAR = "linear"
CFG_DEFAULT_MODEL = "default_model"
CFG_ACTIVE_MODEL_VERSION = "active_model_version"
CFG_FEATURES = "features"
CFG_ENABLE_RULE_PRICING = "enable_rule_pricing"
CFG_ENABLE_HYBRID_MODE = "enable_hybrid_mode"
CFG_ENABLE_AI_PRICING = "enable_ai_pricing"
