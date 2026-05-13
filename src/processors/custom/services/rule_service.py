"""rule_service.py — Thin wrapper that executes rule-based pricing."""
from processors.custom.rule_engine import generate_rule_output
import pandas as pd

def run_rule_service(master_df: pd.DataFrame) -> pd.DataFrame:
    return generate_rule_output(master_df)
