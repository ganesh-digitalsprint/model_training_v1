"""
optuna_tuner.py
Optuna-based tuner (fallback when Ray Tune is not available).
"""
import optuna
from sklearn.model_selection import cross_val_score


def run_optuna(model_cls, param_fn, X_train, y_train,
               n_trials: int = 50, cv: int = 3) -> tuple:
    def objective(trial):
        params = param_fn(trial)
        model  = model_cls(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="neg_mean_absolute_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model  = model_cls(**best_params).fit(X_train, y_train)
    return best_model, best_params
