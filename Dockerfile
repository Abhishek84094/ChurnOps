# ChurnOps — Production Dockerfile
# ============================================================
# Multi-stage build: builder installs deps, runtime is minimal
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer cache optimization)
COPY requirements.txt .

# Install to a prefix directory for copy
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ChurnOps Team"
LABEL description="ChurnOps Churn Prediction API — Tuned XGBoost | AUC: 0.8491"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code and model artifacts
COPY api/ ./api/
COPY src/ ./src/
COPY models/champion_model.joblib ./models/champion_model.joblib
COPY models/model_card.json ./models/model_card.json
COPY params.yaml .

# Create non-root user for security
RUN useradd --no-create-home --shell /bin/false churnops \
    && chown -R churnops:churnops /app
USER churnops

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
