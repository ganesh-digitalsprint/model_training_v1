"""
output_writer.py
Saves pricing output CSVs with versioned filenames.
Replaces: src/output/save_output.py
"""
import os
import pandas as pd
from utils.constants import (
    DEFAULT_OUTPUT_DIR,
    META_TIMESTAMP, META_MODEL_VERSION, META_RUN_ID,
)


def save_output(df: pd.DataFrame, mode: str, metadata: dict,
                output_dir: str = DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    filename  = (f"{mode}_pricing_output_"
                 f"{metadata[META_TIMESTAMP]}_"
                 f"{metadata[META_MODEL_VERSION]}_"
                 f"{metadata[META_RUN_ID]}.csv")
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Output saved: {file_path}")