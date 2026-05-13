"""
classification_eval.py
Standard classification metrics placeholder.
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, average="weighted")}
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics
