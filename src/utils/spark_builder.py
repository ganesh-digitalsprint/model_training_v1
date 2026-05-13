"""
spark_builder.py
Spark session builder for Databricks / EMR environments.
"""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "DynamicPricingTraining") -> SparkSession:
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())
