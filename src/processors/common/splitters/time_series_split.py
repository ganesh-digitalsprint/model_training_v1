"""time_series_split.py — Time-aware train/test splitting."""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split(df: pd.DataFrame, date_col: str, n_splits: int = 5):
    df   = df.sort_values(date_col)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(df))
    train_idx, test_idx = splits[-1]
    return df.iloc[train_idx], df.iloc[test_idx]
