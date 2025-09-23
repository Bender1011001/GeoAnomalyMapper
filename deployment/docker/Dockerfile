# Multi-stage Dockerfile for GeoAnomalyMapper (GAM)
# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libnetcdf-dev \
    libhdf5-dev \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
WORKDIR /tmp/build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .[geophysics,visualization]

# Copy source code and install in editable mode
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.12-slim-bookworm AS runtime

# Install runtime system dependencies (no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal31 \
    libgeos-c1v5 \
    libproj19 \
    libspatialindex6v5 \
    libnetcdf19 \
    libhdf5-103 \
    libpq5 \
    libssl3 \
    libffi8 \
    ca-certificates \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /opt/venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 gam \
    && useradd --uid 1000 --gid 1000 -m gam \
    && chown -R gam:gam /app
USER gam

# Set working directory
WORKDIR /app

# Copy source code (non-sensitive)
COPY --from=builder --chown=gam:gam /app .

# Expose port for potential API/web interface (adjust if needed)
EXPOSE 8000

# Health check using the CLI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gam --help || exit 1

# Default entrypoint: GAM CLI
ENTRYPOINT ["gam"]
CMD ["--help"]