# Use PyTorch official image with CUDA support and Python 3.11
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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
RUN python3 -m spacy download en_core_web_sm --break-system-packages

# Create directories for models and data
RUN mkdir -p /models /data /tmp/booknlp /models/.cache/transformers /models/.cache/datasets

# Expose API port
EXPOSE 9999

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9999/health || exit 1

# Run the server
CMD ["python3", "booknlp_server.py"]
