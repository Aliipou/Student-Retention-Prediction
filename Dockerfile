# Builder stage
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target /build/deps

# Runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages

# Create non-root user
RUN useradd -m -u 1000 appuser

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY assets/ ./assets/

RUN mkdir -p data models assets && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501 8000

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health 2>/dev/null || \
        curl --fail http://localhost:8000/health 2>/dev/null || exit 1

CMD ["streamlit", "run", "src/dashboard.py", "--server.address", "0.0.0.0"]
