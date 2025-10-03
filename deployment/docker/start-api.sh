#!/usr/bin/env bash
# Startup script for GAM FastAPI backend
# Handles environment setup, mode detection, and uvicorn launch

set -euo pipefail

# ---------- logging ----------
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start-api] $1" >&2
}

# ---------- defaults ----------
export PYTHONPATH="/app:${PYTHONPATH:-}"
export GAM_ENV="${GAM_ENV:-production}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
# Prefer WEB_CONCURRENCY if present (Heroku/container convention)
export UVICORN_WORKERS="${UVICORN_WORKERS:-${WEB_CONCURRENCY:-4}}"
export UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
export UVICORN_PORT="${UVICORN_PORT:-8000}"
export PYTHONUNBUFFERED=1

# ---------- fs prep ----------
mkdir -p /app/data /app/results
# 775 so a non-root user in the group can write; 755 blocks group writes
chmod 775 /app/data /app/results || true

# ---------- mode ----------
UVICORN_ARGS=""
if [[ "$GAM_ENV" == "development" || "$GAM_ENV" == "dev" ]]; then
  log "Starting in development mode"
  UVICORN_ARGS="--reload --log-level debug"
  UVICORN_WORKERS=1
else
    log "Starting in production mode"
    UVICORN_WORKERS=1  # Add this line
    UVICORN_ARGS="--log-level $LOG_LEVEL --access-log"
fi

log "Configuring uvicorn with workers=$UVICORN_WORKERS, host=$UVICORN_HOST, port=$UVICORN_PORT"
log "Launching: uvicorn gam.api.main:app --host $UVICORN_HOST --port $UVICORN_PORT --workers $UVICORN_WORKERS $UVICORN_ARGS"

# Use exec so uvicorn becomes PID 1 (clean signals/shutdown). No trap needed.
exec uvicorn gam.api.main:app \
  --host "$UVICORN_HOST" \
  --port "$UVICORN_PORT" \
  --workers "$UVICORN_WORKERS" \
  $UVICORN_ARGS

# If exec fails:
log "Failed to start uvicorn"
exit 1
