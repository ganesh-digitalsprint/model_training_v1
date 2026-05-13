"""ai_service.py — Thin wrapper that executes AI-based pricing."""
from processors.custom.ai_engine import generate_ai_output
import pandas as pd

def run_ai_service(master_df: pd.DataFrame, model_name: str = "linear",
                   model_version: str = "v1.0", tolerance: float = 0.05) -> pd.DataFrame:
    return generate_ai_output(master_df, model_name=model_name,
                              version=model_version, tolerance=tolerance)
