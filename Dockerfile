# Multi-stage build for minimal runtime image
FROM python:3.13-slim AS base

# System deps: ffmpeg for video, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY assets/ ./assets/
COPY data/ ./data/
COPY output/ ./output/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck: database accessible, ffmpeg present
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD python -c "import sqlite3; sqlite3.connect('data/app.db').execute('SELECT 1'); import shutil; assert shutil.which('ffmpeg')" || exit 1

# Default: run scheduler loop (1 hour interval)
ENTRYPOINT ["python", "-m", "app.scheduler", "run-loop", "--interval", "3600"]


# ---- Development stage (includes test deps) ----
FROM base AS dev
USER root
RUN pip install --no-cache-dir pytest pytest-mock
USER appuser
ENTRYPOINT ["python", "-m", "pytest", "tests/", "-q"]