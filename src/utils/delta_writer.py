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
        .config("spark.sql.extensions",       "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Use Arrow for pandas→Spark conversion (avoids Python worker crash)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Single partition — no worker processes needed for small CSVs
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism",    "1")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def csv_to_delta(csv_path: str, delta_path: str):
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv.resolve()}")

    spark = get_spark()

    # Read with pandas first
    pdf = pd.read_csv(csv)
    print(f"📄 Loaded {len(pdf)} rows, {len(pdf.columns)} columns")

    # Convert via Arrow (no Python worker spawned)
    sdf = spark.createDataFrame(pdf)

    # Coalesce to 1 partition — avoids multi-worker issues on Windows
    sdf.coalesce(1).write.format("delta").mode("overwrite").save(str(delta_path))

    print(f"✅ Written {len(pdf)} rows → {delta_path}")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: uv run python delta_writer.py <csv_path> <delta_path>")
        sys.exit(1)

    csv_to_delta(sys.argv[1], sys.argv[2])