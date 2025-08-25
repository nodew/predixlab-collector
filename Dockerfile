# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set metadata
LABEL maintainer="qstock-collector"
LABEL description="Independent stock market data service using yfinance and Yahoo Query"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install uv for faster package management
RUN pip install uv

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY . .

# Set appropriate permissions
RUN chmod +x main.py

# Create a non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Change ownership of app directory to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "main.py", "--help"]
