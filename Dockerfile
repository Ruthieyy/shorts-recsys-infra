FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY certs /app/certs
COPY spark /app/spark

EXPOSE 8000
CMD ["python", "-m", "shorts_recsys.serve"]
