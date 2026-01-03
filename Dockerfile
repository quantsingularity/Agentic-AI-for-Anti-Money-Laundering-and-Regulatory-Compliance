FROM python:3.10-slim

# Metadata
LABEL maintainer="AML Agentic System"
LABEL description="Multi-agent system for AML compliance and SAR generation"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    graphviz \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy application code
COPY code/ ./code/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY run_quick.sh run_full.sh ./

# Make scripts executable
RUN chmod +x run_quick.sh run_full.sh

# Create necessary directories
RUN mkdir -p results figures logs

# Expose port for web UI
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "-m", "code.ui.web_app"]
