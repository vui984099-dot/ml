# Multi-stage Docker build for Amazon Product Q&A system

FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/parquet data/indexes models

# API stage
FROM base as api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# UI stage  
FROM base as ui
EXPOSE 8501
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Data processing stage
FROM base as data-processor
CMD ["python", "src/etl/ingest_data.py"]

# Index builder stage
FROM base as index-builder
CMD ["python", "src/indexing/build_index.py"]