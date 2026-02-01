# Use PyTorch official image with CUDA support
# This image already includes Python and PyTorch with CUDA
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

# Install additional system dependencies (Python is already installed)
RUN apt-get update -qq && apt-get install -y -qq \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy server code
COPY booknlp_server.py /app/
COPY requirements.txt /app/

# Install Python dependencies
# --break-system-packages is safe in Docker containers (PEP 668)
RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages && \
    pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Download SpaCy model
RUN python3 -m spacy download en_core_web_sm

# Create directories for models and data
RUN mkdir -p /models /data /tmp/booknlp

# Expose API port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run the server
CMD ["python3", "booknlp_server.py"]
