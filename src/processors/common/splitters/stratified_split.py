"""stratified_split.py — Stratified train/test splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split


def split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=None)
