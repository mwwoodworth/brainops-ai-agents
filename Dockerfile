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

# Create non-root user early so we can reference it throughout
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home --shell /bin/bash appuser

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

# Copy installed packages from builder into a shared location (not /root),
# preserving appuser ownership to avoid expensive recursive chown layers.
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Ensure Python can find the installed packages under the non-root user's home
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/home/appuser/.local/lib/python3.11/site-packages:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Install Playwright browser conditionally (skip with --build-arg INSTALL_PLAYWRIGHT=0)
# Set PLAYWRIGHT_BROWSERS_PATH so browsers are stored in appuser's home
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright
RUN if [ "$INSTALL_PLAYWRIGHT" = "1" ]; then \
        /home/appuser/.local/bin/playwright install chromium 2>/dev/null || echo "Playwright browser install skipped"; \
    else \
        echo "Playwright browser install skipped (INSTALL_PLAYWRIGHT=0)"; \
    fi

# Copy application code LAST for maximum cache benefit and preserve appuser ownership.
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p logs /var/lib/ai-memory /var/log

# Setup cron for memory sync
COPY crontab /etc/cron.d/memory-sync
RUN chmod 0644 /etc/cron.d/memory-sync && \
    crontab -u appuser /etc/cron.d/memory-sync

# Ensure runtime data/cache directories are writable by appuser.
RUN chown -R appuser:appuser /var/lib/ai-memory && \
    chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Expose port
EXPOSE 10000

# Switch to non-root user
USER appuser

# Start application (cron requires root, so use a Python-based scheduler or skip cron)
# Note: 'service cron start' requires root. If cron is essential, use supercronic or
# an entrypoint script with gosu/su-exec. For now, run the app directly.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
