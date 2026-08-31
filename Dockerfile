# Reference image for the daemon and its HTTP surface.
#
# The package lives under src/ and needs no build step, so the image is a slim
# Python base plus the pinned dependencies. It runs as a non-root user and keeps
# all mutable state — sqlite database, checkpoints, JSONL feeds — under /data,
# which is the only path that needs to be a volume.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PME_LOG_LEVEL=INFO \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY __main__.py pyproject.toml ./

RUN useradd --create-home --uid 10001 pme \
    && mkdir -p /data \
    && chown -R pme:pme /app /data
USER pme

VOLUME ["/data"]
EXPOSE 8080

# Liveness is the cheap probe; readiness additionally opens the database.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "pme", "serve", "--host", "0.0.0.0", "--port", "8080", \
     "--output", "/data", "--log-format", "json"]
