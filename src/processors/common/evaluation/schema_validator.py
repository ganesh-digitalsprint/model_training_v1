"""
schema_validator.py
Validates a Delta Lake / pandas DataFrame against an expected schema.
Used in the golden dataset read step before training.
"""
import pandas as pd


class SchemaValidationError(Exception):
    pass


def validate(df: pd.DataFrame, expected_columns: list, raise_on_error: bool = True) -> bool:
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        msg = f"Schema validation failed. Missing columns: {missing}"
        if raise_on_error:
            raise SchemaValidationError(msg)
        print(f"WARNING: {msg}")
        return False

    null_report = {c: int(df[c].isna().sum()) for c in expected_columns if df[c].isna().any()}
    if null_report:
        print(f"Null value report: {null_report}")

    print(f"Schema validation passed. Rows: {len(df)}, Columns: {list(df.columns)}")
    return True
