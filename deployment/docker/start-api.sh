#!/bin/bash
# Startup script for GAM FastAPI backend
# Handles environment setup, mode detection, and uvicorn launch

set -euo pipefail

# Default environment variables
export PYTHONPATH="/app:${PYTHONPATH:-}"
export GAM_ENV="${GAM_ENV:-production}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export UVICORN_WORKERS="${UVICORN_WORKERS:-4}"
export UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
export UVICORN_PORT="${UVICORN_PORT:-8000}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start-api] $1" >&2
}

# Ensure directories exist and are writable (non-root user)
mkdir -p /app/data /app/results
chmod 755 /app/data /app/results

# Mode-specific configuration
if [ "$GAM_ENV" = "development" ] || [ "$GAM_ENV" = "dev" ]; then
    log "Starting in development mode"
    UVICORN_ARGS="--reload --log-level debug"
    UVICORN_WORKERS=1  # Single worker for hot reload
else
    log "Starting in production mode"
    UVICORN_ARGS="--log-level $LOG_LEVEL --access-log"
fi

# Health check endpoint addition (if not present, but per task no core mod; assume exists or use root)
log "Configuring uvicorn with workers=$UVICORN_WORKERS, host=$UVICORN_HOST, port=$UVICORN_PORT"

# Graceful shutdown trap
trap 'log "Received shutdown signal, stopping uvicorn..."; kill -TERM $UVICORN_PID || true; wait $UVICORN_PID || true; log "Shutdown complete"; exit 0' TERM INT

# Launch uvicorn
log "Launching FastAPI server: uvicorn gam.api.main:app --host $UVICORN_HOST --port $UVICORN_PORT --workers $UVICORN_WORKERS $UVICORN_ARGS"
exec uvicorn gam.api.main:app \
    --host "$UVICORN_HOST" \
    --port "$UVICORN_PORT" \
    --workers "$UVICORN_WORKERS" \
    $UVICORN_ARGS

# If exec fails
log "Failed to start uvicorn"
exit 1