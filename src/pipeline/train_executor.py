"""
train_executor.py
Orchestrates the full model fit → eval → register cycle.
All paths, column names, and numeric hyper-defaults are read from YAML config.

Pipeline
--------
1.  Load golden dataset from Delta Lake (or CSV fallback)
2.  Optionally tune hyperparameters with Ray Tune
3.  Train model
4.  Evaluate and log metrics + artifacts to MLflow
5.  Register model in MLflow registry → transition to Staging
6.  Write model binary (.pkl) to Delta Lake model registry
"""
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import ray
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from bootstrap.train_config_loader                 import get_active_config
from processors.common.trainers.model_factory      import ModelFactory
from processors.common.tuning.ray_tune_adapter     import tune_xgboost, tune_random_forest
from processors.common.evaluation.regression_eval  import evaluate
from processors.custom.ingestion.delta_lake_reader import (
    read_golden_dataset, read_golden_dataset_from_csv)
from orchestration.experiment_tracker              import ExperimentTracker
from orchestration.model_metadata_store            import ModelMetadataStore
from processors.custom.ingestion.delta_model_writer import write_model_to_delta
from utils.constants import (
    XGBOOST, VERSION, RANDOM_FOREST, SKLEARN, RAY_TUNE,
    METRIC_R2, METRIC_MAE, METRIC_RMSE, METRIC_PSEUDO_ACC,
    STAGE_STAGING, MODEL_TAG_DOMAIN,
    CFG_TRAINING_DATA, CFG_FEATURE_COLUMNS, CFG_TARGET_COLUMN,
    CFG_RAY, CFG_RAY_TEMP_DIR,
    CFG_TEST_SIZE, CFG_RANDOM_STATE, CFG_CSV_FALLBACK_PATH,
    CFG_ML, CFG_ENVIRONMENT, CFG_MLFLOW, CFG_MLFLOW_TRACKING_URI, CFG_EXPERIMENT_NAME,
    CFG_PRICING, CFG_TOLERANCE,
    ML_TRAINING, ENABLE_HYPERPARAMETER_TUNING, TUNING_METHOD, TUNING_TRIALS,
    DELTA_LAKE, GOLDEN_DATASET_PATH, EXPECTED_COLUMNS,
    CFG_MODEL_LOCAL_PATH, DEFAULT_MODEL_LOCAL_PATH,
    PARAM_MODEL_NAME, PARAM_MODEL_VERSION, PARAM_DATASET_ROWS,
    PARAM_FEATURES, PARAM_FEATURES_USED, PARAM_TARGET,
    PARAM_TRAIN_SIZE, PARAM_TEST_SIZE,
    PARAM_TUNING_METHOD, PARAM_DATASET_SOURCE, PARAM_TUNING_NONE,
    PARAM_PRICING_STRATEGY, PARAM_TOLERANCE, PARAM_DATASET_NAME, PARAM_DATASET_VERSION,
    ARTIFACT_ACTUAL_VS_PRED, ARTIFACT_FEATURE_IMPORTANCE, MLFLOW_ARTIFACT_MODEL,
    CHART_LABEL_ACTUAL, CHART_LABEL_PREDICTED,
    CHART_TITLE_ACTUAL_VS_PRED, CHART_TITLE_FEATURE_IMPORTANCE,
    RESULT_RMSE, RESULT_PSEUDO_ACCURACY, RESULT_REGISTERED_VERSION,
    RESULT_MLFLOW_REGISTRY, RESULT_DELTA_WRITE, RESULT_DELTA_WRITE_SUCCESS,
    RESULT_ENVIRONMENT, DEFAULT_ENVIRONMENT,
    MODE_HYBRID, DEFAULT_DATASET_NAME, DEFAULT_DATASET_VERSION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_training_schema(config: dict) -> tuple[list[str], list[str], str]:
    """
    Returns (feature_cols, expected_cols, target_col) from config.
    """
    data_cfg     = config.get(CFG_TRAINING_DATA, {})
    feature_cols = data_cfg.get(CFG_FEATURE_COLUMNS)
    target_col   = data_cfg.get(CFG_TARGET_COLUMN)

    if not feature_cols or not target_col:
        raise ValueError(
            "YAML is missing [training_data.feature_columns] or "
            "[training_data.target_column]. Please add them."
        )

    delta_cfg     = config.get(DELTA_LAKE, {})
    expected_cols = delta_cfg.get(EXPECTED_COLUMNS) or (feature_cols + [target_col])
    return feature_cols, expected_cols, target_col


def _resolve_ray_dir(config: dict) -> str:
    """Returns Ray temp dir from config[ray][temp_dir]."""
    ray_cfg = config.get(CFG_RAY, {})
    ray_dir = ray_cfg.get(CFG_RAY_TEMP_DIR)
    if not ray_dir:
        raise ValueError(
            "YAML is missing [ray.temp_dir]. Add it to avoid Ray using a path with spaces."
        )
    return ray_dir


def _setup_mlflow(config: dict) -> None:
    """
    Configures MLflow tracking URI and experiment from YAML config.
    Raises clearly if either key is missing — catches the most common
    reason MLflow silently logs nowhere.
    """
    mlflow_cfg    = config.get(CFG_MLFLOW, {})
    tracking_uri  = mlflow_cfg.get(CFG_MLFLOW_TRACKING_URI)
    experiment    = mlflow_cfg.get(CFG_EXPERIMENT_NAME)

    if not tracking_uri:
        raise ValueError(
            "YAML is missing [mlflow.tracking_uri]. "
            "Add it so MLflow knows where to log runs."
        )
    if not experiment:
        raise ValueError(
            "YAML is missing [mlflow.experiment_name]. "
            "Add it so runs are grouped under the correct experiment."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"MLflow experiment   : {experiment}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model_name:        str  = XGBOOST,
    version:           str  = VERSION,
    csv_fallback_path: str  = None,
) -> dict:
    """
    Full training pipeline.

    Returns
    -------
    dict with model_name, version, metrics, mlflow registered_version,
    and delta_write status.
    """
    config = get_active_config()

    # ── MLflow setup — must happen before any mlflow call ────────────────────
    _setup_mlflow(config)

    # ── Ray init ──────────────────────────────────────────────────────────────
    ray_dir = _resolve_ray_dir(config)
    os.makedirs(ray_dir, exist_ok=True)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, _temp_dir=ray_dir)

    # ── 1. Resolve schema from config ─────────────────────────────────────────
    feature_cols, expected_cols, target_col = _resolve_training_schema(config)

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    delta_cfg  = config.get(DELTA_LAKE, {})
    delta_path = delta_cfg.get(GOLDEN_DATASET_PATH)
    fallback_path = csv_fallback_path or delta_cfg.get(CFG_CSV_FALLBACK_PATH)

    if delta_path:
        print(f"Loading dataset from Delta Lake: {delta_path}")
        ds = read_golden_dataset(delta_path, expected_columns=expected_cols)
        df = ds.to_pandas()
    elif fallback_path:
        print(f"Delta path not set — using CSV fallback: {fallback_path}")
        df = read_golden_dataset_from_csv(fallback_path, expected_columns=expected_cols)
    else:
        raise ValueError(
            "No data source available. Set [delta_lake.golden_dataset_path] "
            "or [delta_lake.csv_fallback_path] in your YAML config."
        )

    print(f"Dataset loaded — rows: {len(df)}, columns: {list(df.columns)}")

    # ── 3. Feature / target split ─────────────────────────────────────────────
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from dataset: {missing}")

    X = df[feature_cols]
    y = df[target_col]

    ml_train     = config.get(ML_TRAINING, {})
    test_size    = float(ml_train.get(CFG_TEST_SIZE))
    random_state = int(ml_train.get(CFG_RANDOM_STATE))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── 4. Hyperparameter tuning ──────────────────────────────────────────────
    enable_tune   = json.loads(str(ml_train.get(ENABLE_HYPERPARAMETER_TUNING, "false")).lower())
    tuning_method = ml_train.get(TUNING_METHOD, RAY_TUNE)
    n_trials      = int(ml_train.get(TUNING_TRIALS))

    best_params = {}
    if enable_tune and tuning_method == RAY_TUNE:
        print(f"Ray Tune search — model={model_name}, trials={n_trials}")
        if model_name == XGBOOST:
            model, best_params = tune_xgboost(X_train, y_train, num_samples=n_trials)
        elif model_name == RANDOM_FOREST:
            model, best_params = tune_random_forest(X_train, y_train, num_samples=n_trials)
        else:
            model = ModelFactory.create(model_name)
    else:
        model = ModelFactory.create(model_name)

    # ── 5. Train & log to MLflow ──────────────────────────────────────────────
    # Close any stale run left over from a previous failed execution
    if mlflow.active_run():
        mlflow.end_run()

    tracker = ExperimentTracker(model_name, version)
    run_id  = None
    metrics = {}

    with tracker.start_run():

        # ── Log all params in one place — inside the active run ───────────────
        tracker.log_params({
            PARAM_MODEL_NAME:      model_name,
            PARAM_MODEL_VERSION:   version,
            PARAM_DATASET_ROWS:    len(df),
            PARAM_FEATURES:        ",".join(feature_cols),
            PARAM_FEATURES_USED:   ",".join(feature_cols),
            PARAM_TARGET:          target_col,
            PARAM_TRAIN_SIZE:      len(X_train),
            PARAM_TEST_SIZE:       len(X_test),
            PARAM_TUNING_METHOD:   tuning_method if enable_tune else PARAM_TUNING_NONE,
            PARAM_DATASET_SOURCE:  delta_path or fallback_path,
            PARAM_PRICING_STRATEGY: MODE_HYBRID,
            PARAM_TOLERANCE:       config.get(CFG_PRICING, {}).get(CFG_TOLERANCE, 0.05),
            PARAM_DATASET_NAME:    DEFAULT_DATASET_NAME,
            PARAM_DATASET_VERSION: DEFAULT_DATASET_VERSION,
            **best_params,
        })

        # Skip re-fit if Ray Tune already trained the model
        if not (enable_tune and tuning_method == RAY_TUNE):
            model.fit(X_train, y_train)

        # ── Save .pkl locally ─────────────────────────────────────────────────
        local_model_dir = (
            config.get(DELTA_LAKE, {}).get(CFG_MODEL_LOCAL_PATH)
            or DEFAULT_MODEL_LOCAL_PATH
        )
        os.makedirs(local_model_dir, exist_ok=True)
        local_pkl_path = os.path.join(local_model_dir, f"{model_name}_{version}.pkl")
        joblib.dump(model, local_pkl_path)
        print(f"Model saved locally: {local_pkl_path}")

        preds      = model.predict(X_test)
        metrics    = evaluate(y_test, preds)
        rmse       = np.sqrt(mean_squared_error(y_test, preds))
        pseudo_acc = 100 - (np.mean(np.abs((y_test - preds) / y_test)) * 100)
        metrics[METRIC_RMSE]       = rmse
        metrics[METRIC_PSEUDO_ACC] = pseudo_acc

        tracker.log_metrics(metrics)

        # ── Log model under a consistent artifact path ────────────────────────
        if model_name == XGBOOST:
            mlflow.xgboost.log_model(model, MLFLOW_ARTIFACT_MODEL)
        else:
            mlflow.sklearn.log_model(model, MLFLOW_ARTIFACT_MODEL)

        run_id = tracker.run_id   # grabbed inside the with block — always valid here

        print(
            f"R2: {round(metrics[METRIC_R2], 3)}  "
            f"MAE: {round(metrics[METRIC_MAE], 3)}  "
            f"RMSE: {round(rmse, 3)}  "
            f"PseudoAcc: {round(pseudo_acc, 2)}%"
        )

        # Actual vs Predicted scatter
        plt.figure()
        plt.scatter(y_test, preds, alpha=0.5)
        plt.xlabel(CHART_LABEL_ACTUAL)
        plt.ylabel(CHART_LABEL_PREDICTED)
        plt.title(CHART_TITLE_ACTUAL_VS_PRED)
        plt.tight_layout()
        plt.savefig(ARTIFACT_ACTUAL_VS_PRED)
        tracker.log_artifact(ARTIFACT_ACTUAL_VS_PRED)
        plt.close()

        # Feature importance bar chart
        if hasattr(model, "feature_importances_"):
            plt.figure()
            plt.bar(feature_cols, model.feature_importances_)
            plt.xticks(rotation=45)
            plt.title(CHART_TITLE_FEATURE_IMPORTANCE)
            plt.tight_layout()
            plt.savefig(ARTIFACT_FEATURE_IMPORTANCE)
            tracker.log_artifact(ARTIFACT_FEATURE_IMPORTANCE)
            plt.close()

    # run_id is captured inside the with block — safe to use here
    if run_id is None:
        raise RuntimeError("run_id is None — MLflow run did not start correctly.")

    # ── 6. Register in MLflow registry → transition to Staging ───────────────
    registry_name    = f"{MODEL_TAG_DOMAIN}_{model_name}"
    model_uri        = f"runs:/{run_id}/{MLFLOW_ARTIFACT_MODEL}"

    store       = ModelMetadataStore()
    reg_version = store.register(run_id, registry_name, artifact_name=MLFLOW_ARTIFACT_MODEL)
    store.transition(registry_name, reg_version, STAGE_STAGING)

    # ── 7. Write model binary to Delta Lake model registry ────────────────────
    environment = config.get(CFG_ML, {}).get(CFG_ENVIRONMENT, DEFAULT_ENVIRONMENT)
    write_model_to_delta(
        model         = model,
        model_name    = model_name,
        model_version = version,
        environment   = environment,
        config        = config,
    )

    return {
        PARAM_MODEL_NAME:          model_name,
        PARAM_MODEL_VERSION:       version,
        METRIC_R2:                 metrics[METRIC_R2],
        METRIC_MAE:                metrics[METRIC_MAE],
        RESULT_RMSE:               round(metrics[METRIC_RMSE], 4),
        RESULT_PSEUDO_ACCURACY:    round(metrics[METRIC_PSEUDO_ACC], 2),
        RESULT_REGISTERED_VERSION: reg_version,
        RESULT_MLFLOW_REGISTRY:    registry_name,
        RESULT_DELTA_WRITE:        RESULT_DELTA_WRITE_SUCCESS,
        RESULT_ENVIRONMENT:        environment,
    }