"""
grid_search_tuner.py
Vanilla sklearn GridSearch (no Ray) — used when cluster is unavailable.
"""
from sklearn.model_selection import GridSearchCV


def run_grid_search(estimator, param_grid: dict, X_train, y_train,
                    cv: int = 3, scoring: str = "neg_mean_absolute_error") -> tuple:
    search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                          n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
