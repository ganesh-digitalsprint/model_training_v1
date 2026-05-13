"""
db.py
Database connection helper. Replaces src/utils/db.py
Credentials should be injected via environment variables, not hardcoded.
"""
import os
from sqlalchemy import create_engine


def get_connection():
    user     = os.getenv("DB_USER",     "root")
    password = os.getenv("DB_PASSWORD", "")
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "3306")
    db       = os.getenv("DB_NAME",     "dynamic_pricing")
    return create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")
