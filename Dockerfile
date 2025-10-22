# Customer Segmentation with Auto Encoders - Docker Image
# This image contains all dependencies and code needed to run the customer segmentation pipeline

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY train_pipeline.py .
COPY predict.py .
COPY download_data.py .
COPY README.md .
COPY SETUP_INSTRUCTIONS.md .

# Create necessary directories
RUN mkdir -p data/raw data/processed models results results/plots

# Copy data if available (optional - can be mounted as volume)
COPY Kaggle/*.csv data/raw/ 2>/dev/null || true

# Set proper permissions
RUN chmod +x train_pipeline.py predict.py

# Expose port for Jupyter notebook (if needed)
EXPOSE 8888

# Default command - run the training pipeline
CMD ["python", "train_pipeline.py"]

# Alternative commands:
# Run prediction: docker run customer-segmentation python predict.py
# Run Jupyter: docker run -p 8888:8888 customer-segmentation jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
# Interactive shell: docker run -it customer-segmentation bash
