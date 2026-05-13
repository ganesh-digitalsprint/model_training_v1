"""
regression_eval.py
Standard regression metrics. Reusable across all regression training jobs.
Replaces: metrics logic spread across train_model.py
"""
import math
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


def _clean(x) -> float:
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return 0.0


def evaluate(y_true, y_pred) -> dict:
    r2  = _clean(r2_score(y_true, y_pred))
    mae = _clean(mean_absolute_error(y_true, y_pred))
    mape = _clean(
        np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                       / (np.array(y_true) + 1))) * 100)
    return {"r2": r2, "mae": mae, "mape": mape}
