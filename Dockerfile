# syntax=docker/dockerfile:1.4
# Multi-stage build for optimal size and speed
# Optimized for faster builds with layer caching

FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Use BuildKit cache for pip to speed up rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Build arg to control Playwright installation (set to 0 to skip)
ARG INSTALL_PLAYWRIGHT=1

# Install runtime dependencies in one layer
# Playwright deps are always installed (small overhead) but browser download is optional
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    cron \
    # Playwright browser dependencies (needed for UI testing)
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libwayland-client0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure Python can find the installed packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.11/site-packages:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Install Playwright browser conditionally (skip with --build-arg INSTALL_PLAYWRIGHT=0)
RUN if [ "$INSTALL_PLAYWRIGHT" = "1" ]; then \
        /root/.local/bin/playwright install chromium 2>/dev/null || echo "Playwright browser install skipped"; \
    else \
        echo "Playwright browser install skipped (INSTALL_PLAYWRIGHT=0)"; \
    fi

# Copy application code LAST for maximum cache benefit
COPY . .

# Create necessary directories
RUN mkdir -p logs /var/lib/ai-memory /var/log

# Setup cron for memory sync
COPY crontab /etc/cron.d/memory-sync
RUN chmod 0644 /etc/cron.d/memory-sync && \
    crontab /etc/cron.d/memory-sync

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE 10000

# Start cron and application
# Python is at /usr/local/bin/python (from base image), packages are in /root/.local (from builder)
# PATH includes /root/.local/bin for console_scripts installed by pip
CMD service cron start && python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}
