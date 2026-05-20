# ==========================================
# Stage 1: Build Dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed to build packages (e.g. LightGBM, compiler tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for rapid package installation
COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv venv /opt/venv && \
    /opt/venv/bin/uv pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Minimal Production Image
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Install runtime system libraries (LightGBM requires libgomp1, PySpark requires Java)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy application source code and artifacts
COPY src/ /app/src/
COPY config/ /app/config/
COPY dashboard.py /app/dashboard.py
# Create placeholder directory for trained models
RUN mkdir -p /app/artifacts

EXPOSE 8000
EXPOSE 8501

# Default command to run FastAPI api
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
