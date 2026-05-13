"""
ray_tune_adapter.py
Replaces RandomizedSearchCV with Ray Tune for distributed hyperparameter search.
"""
import os
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# Short paths — avoids Windows 260 char limit
RAY_STORAGE = "D:/ray_results"
os.makedirs(RAY_STORAGE, exist_ok=True)


def short_trial_dirname(trial):
    """Keep trial folder names short to avoid Windows path length limit."""
    return f"trial_{trial.trial_id}"


def _make_objective(model_cls, model_kwargs_fn, X_train, y_train, cv=3):
    """Return a Ray Tune trainable function."""

    def trainable(config):
        estimator = model_cls(**model_kwargs_fn(config))
        scores    = cross_val_score(
            estimator, X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error")
        tune.report({"mae": -scores.mean()})

    return trainable


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 num_samples: int = 20) -> tuple:
    from xgboost import XGBRegressor

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, _temp_dir="D:/ray_tmp")

    search_space = {
        "n_estimators":     tune.choice([200, 300, 400]),
        "max_depth":        tune.choice([3, 5, 7]),
        "learning_rate":    tune.loguniform(0.01, 0.1),
        "subsample":        tune.uniform(0.7, 1.0),
        "colsample_bytree": tune.uniform(0.7, 1.0),
    }

    def model_kwargs(cfg):
        return {**cfg, "random_state": 42, "verbosity": 0}

    objective = _make_objective(XGBRegressor, model_kwargs, X_train, y_train)

    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=num_samples,
        metric="mae", mode="min",
        search_alg=OptunaSearch(),
        scheduler=ASHAScheduler(),
        storage_path=RAY_STORAGE,           # ← short path on D:
        trial_dirname_creator=short_trial_dirname,  # ← short folder names
        verbose=1,
    )

    best_params = analysis.best_config
    best_model  = XGBRegressor(**{**best_params, "random_state": 42}).fit(X_train, y_train)
    print(f"Best XGBoost params: {best_params}")
    return best_model, best_params


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       num_samples: int = 20) -> tuple:
    from sklearn.ensemble import RandomForestRegressor

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, _temp_dir="D:/ray_tmp")

    search_space = {
        "n_estimators":      tune.choice([100, 200, 300, 400]),
        "max_depth":         tune.choice([4, 6, 8, 10]),
        "min_samples_split": tune.choice([2, 5, 10]),
        "min_samples_leaf":  tune.choice([1, 2, 4]),
        "max_features":      tune.choice(["sqrt", "log2"]),
    }

    def model_kwargs(cfg):
        return {**cfg, "random_state": 42}

    objective = _make_objective(RandomForestRegressor, model_kwargs, X_train, y_train)

    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=num_samples,
        metric="mae", mode="min",
        search_alg=OptunaSearch(),
        scheduler=ASHAScheduler(),
        storage_path=RAY_STORAGE,           # ← short path on D:
        trial_dirname_creator=short_trial_dirname,  # ← short folder names
        verbose=1,
    )

    best_params = analysis.best_config
    best_model  = RandomForestRegressor(**{**best_params, "random_state": 42}).fit(X_train, y_train)
    print(f"Best RF params: {best_params}")
    return best_model, best_params