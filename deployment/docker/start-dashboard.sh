#!/bin/bash
# Startup script for GAM Streamlit Dashboard
# Handles environment setup, API URL patching, and streamlit launch

set -euo pipefail

# Default environment variables
export PYTHONPATH="/app:${PYTHONPATH:-}"
export GAM_ENV="${GAM_ENV:-production}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export GAM_API_URL="${GAM_API_URL:-http://localhost:8000}"
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8501}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
log "Starting dashboard with API_URL from environment: $GAM_API_URL"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start-dashboard] $1" >&2
}

# Ensure directories exist and are writable
mkdir -p /app/results
chmod 755 /app/results


# Mode-specific configuration
if [ "$GAM_ENV" = "development" ] || [ "$GAM_ENV" = "dev" ]; then
    log "Starting in development mode"
    STREAMLIT_ARGS="--server.runOnSave true --logger.level debug --theme.base light"
else
    log "Starting in production mode"
    STREAMLIT_ARGS="--server.headless true --logger.level $LOG_LEVEL --theme.base dark --server.enableCORS false"
fi

# Graceful shutdown trap
trap 'log "Received shutdown signal, stopping streamlit..."; pkill -TERM -f "streamlit run" || true; log "Shutdown complete"; exit 0' TERM INT

# Launch Streamlit
log "Launching Streamlit dashboard: streamlit run dashboard/app.py --server.port $STREAMLIT_SERVER_PORT --server.address $STREAMLIT_SERVER_ADDRESS $STREAMLIT_ARGS"
exec streamlit run dashboard/app.py \
    --server.port "$STREAMLIT_SERVER_PORT" \
    --server.address "$STREAMLIT_SERVER_ADDRESS" \
    $STREAMLIT_ARGS

# If exec fails
log "Failed to start streamlit"
exit 1