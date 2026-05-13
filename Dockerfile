FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

ENV PYTHONPATH=/app/src
ENV RAY_ADDRESS=""

ENTRYPOINT ["python", "src/train_launcher.py"]
CMD ["--job", "price_prediction_training", "--env", "dev"]
