# Container image for the URL shortener service (the orchestrator is a development-time tool).
# NOTE: not built in the authoring environment (no Docker available); review before use.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 shortener
WORKDIR /app

COPY pyproject.toml README.md ./
COPY shortener ./shortener
COPY orchestrator ./orchestrator
RUN pip install .

USER shortener
# Secrets (SHORTENER_API_KEYS, SHORTENER_ANALYTICS_SECRET) are injected at runtime, never baked in.
ENV SHORTENER_DB_PATH=/data/shortener.db
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
CMD ["python", "-m", "shortener", "--host", "0.0.0.0", "--port", "8000"]
