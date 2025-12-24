# Multi-stage build for optimal size and speed
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install runtime dependencies + Playwright browser deps for UI testing
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    cron \
    # Playwright browser dependencies
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Install Playwright browser for UI testing
RUN /root/.local/bin/playwright install chromium --with-deps 2>/dev/null || echo "Playwright browser install skipped"

# Copy application code
COPY . .

# Ensure Python can find the installed packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

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
CMD service cron start && python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}
