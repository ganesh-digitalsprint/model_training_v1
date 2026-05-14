import os
import sys

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"]           = r"D:\hadoop"
os.environ["PATH"]                  = r"D:\hadoop\bin;" + os.environ["PATH"]

import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


def get_spark():
    builder = (
        SparkSession.builder
        .master("local[*]")
        .appName("dsai_delta_writer")
        .config("spark.sql.extensions",            "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions",    "1")
        .config("spark.default.parallelism",       "1")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

import re

def sanitize_columns(sdf):
    """Replace invalid Delta column name characters with underscores."""
    for col_name in sdf.columns:
        clean = re.sub(r'[ ,;{}()\n\t=?]', '_', col_name).strip('_')
        if clean != col_name:
            sdf = sdf.withColumnRenamed(col_name, clean)
    return sdf


def _read_file(file_path: Path) -> pd.DataFrame:
    """
    Read CSV or XLSX into a DataFrame.
    Raises clearly if the extension is unsupported.
    """
    ext = file_path.suffix.lower()

    if ext == ".csv":
        print(f"Reading CSV  : {file_path}")
        return pd.read_csv(file_path)

    if ext in (".xlsx", ".xls"):
        print(f"Reading Excel: {file_path}")
        # openpyxl is required for .xlsx — install with: pip install openpyxl
        # xlrd is required for .xls   — install with: pip install xlrd
        return pd.read_excel(file_path, engine="openpyxl" if ext == ".xlsx" else "xlrd")

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: .csv, .xlsx, .xls\n"
        f"File: {file_path}"
    )


def file_to_delta(file_path: str, delta_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    spark = get_spark()

    pdf = _read_file(path)
    print(f"Loaded {len(pdf):,} rows, {len(pdf.columns)} columns")

    # Convert via Arrow — no Python worker spawned
    sdf = spark.createDataFrame(pdf)
    sdf = sanitize_columns(sdf)
    sdf.coalesce(1).write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(str(delta_path))
    # Coalesce to 1 partition — avoids multi-worker issues on Windows
    sdf.coalesce(1).write.format("delta").mode("overwrite").save(str(delta_path))

    print(f"Written {len(pdf):,} rows → {delta_path}")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: uv run python delta_writer.py <file_path> <delta_path>")
        print("       Supported file types: .csv, .xlsx, .xls")
        sys.exit(1)

    file_to_delta(sys.argv[1], sys.argv[2])