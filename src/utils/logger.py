"""
logger.py
Structured JSON logger for all training jobs.
"""
import logging
import json
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "level":   record.levelname,
            "name":    record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger  = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
