#!/usr/bin/env bash
# Startup script for GAM Streamlit Dashboard
# Handles environment setup, API URL patching, and streamlit launch

set -euo pipefail

# ---------- logging ----------
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start-dashboard] $1" >&2
}

# ---------- defaults ----------
export PYTHONPATH="/app:${PYTHONPATH:-}"
export GAM_ENV="${GAM_ENV:-production}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export GAM_API_URL="${GAM_API_URL:-http://localhost:8000}"
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8501}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

log "Starting dashboard with GAM_API_URL=$GAM_API_URL"

# ---------- fs prep ----------
mkdir -p /app/results
chmod 775 /app/results || true

# ---------- mode ----------
STREAMLIT_ARGS=""
if [[ "$GAM_ENV" == "development" || "$GAM_ENV" == "dev" ]]; then
  log "Starting in development mode"
  STREAMLIT_ARGS="--server.runOnSave true --logger.level debug --theme.base light"
else
  log "Starting in production mode"
  STREAMLIT_ARGS="--server.headless true --logger.level $LOG_LEVEL --theme.base dark --server.enableCORS false"
fi

log "Launching: streamlit run dashboard/app.py --server.port $STREAMLIT_SERVER_PORT --server.address $STREAMLIT_SERVER_ADDRESS $STREAMLIT_ARGS"

# Use exec so Streamlit takes PID 1; no trap needed.
exec streamlit run dashboard/app.py \
  --server.port "$STREAMLIT_SERVER_PORT" \
  --server.address "$STREAMLIT_SERVER_ADDRESS" \
  $STREAMLIT_ARGS

# If exec fails:
log "Failed to start streamlit"
exit 1
