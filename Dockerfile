# Zelvryn API — production image
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api ./api
COPY coldstart ./coldstart
COPY config ./config
COPY core ./core
COPY db ./db
COPY detectors ./detectors
COPY ingestion ./ingestion
COPY ml ./ml
COPY profiles ./profiles
COPY scripts ./scripts
COPY simulation ./simulation
COPY models ./models
COPY data/processed ./data/processed

RUN useradd --create-home --uid 10001 zelvryn \
 && mkdir -p /app/logs /app/data/shadow_exports \
 && chown -R zelvryn:zelvryn /app/logs /app/data /app/models
USER zelvryn

ENV ZELVRYN_ENV=production \
    API_WORKERS=4

# Hosts like Render set PORT; default to 8000 elsewhere
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -sf http://localhost:${PORT}/health/live || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${API_WORKERS} --proxy-headers --forwarded-allow-ips='*'"]
