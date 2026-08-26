# Build frontend
FROM node:20-alpine AS frontend-builder

ARG VITE_API_TOKEN=""
ENV VITE_API_TOKEN=${VITE_API_TOKEN}

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Python backend with frontend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (postgresql-client for pg_dump backups)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy ML pipeline
COPY ml/ ./ml/

# Copy analysis modules
COPY canslim_scorer.py ./
COPY data_fetcher.py ./
COPY growth_projector.py ./
COPY sp500_tickers.py ./
COPY redis_cache.py ./
COPY config_loader.py ./
COPY config/ ./config/
COPY async_data_fetcher.py ./
COPY async_scanner.py ./
COPY fmp_rate_limiter.py ./

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Build stamp: UTC epoch seconds captured at image-build time, surfaced by
# /health as the deploy stamp (backend/build_info.py formats it to Central
# time). Sits after the last COPY so any code change refreshes this layer —
# replaces the manually-bumped BUILD_VERSION that went stale within weeks.
RUN date -u +%s > /app/build_stamp.txt

# Create data + backup directories
RUN mkdir -p /app/data /app/data/backups

# Environment variables
ENV PYTHONPATH=/app
ENV PORT=8001

EXPOSE 8001

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
