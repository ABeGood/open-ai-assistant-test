# Use Python 3.11 alpine for smaller size and lower memory usage
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies in one layer
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    gfortran \
    musl-dev \
    libffi-dev \
    postgresql-dev \
    freetype-dev \
    libpng-dev \
    openblas-dev \
    lapack-dev \
    && apk add --no-cache \
    ca-certificates \
    libpq \
    freetype \
    libpng \
    openblas \
    lapack

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps

# Copy only necessary application files
COPY main.py .
COPY telegram_bot/ telegram_bot/
COPY agents/ agents/
COPY classes/ classes/
COPY config/ config/
COPY db_utils/ db_utils/
COPY data/files_processed/ data/files_processed/

# Create non-root user
RUN adduser -D -s /bin/sh app && \
    chown -R app:app /app
USER app

# Command to run the application
CMD ["python", "main.py"]