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

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start-dashboard] $1" >&2
}

# Ensure directories exist and are writable
mkdir -p /app/results
chmod 755 /app/results

# Patch API_BASE_URL in dashboard/app.py to use environment variable
# Replace hardcoded "http://localhost:8000" with dynamic value
if grep -q "http://localhost:8000" /app/dashboard/app.py; then
    log "Patching API_BASE_URL in dashboard/app.py to use GAM_API_URL: $GAM_API_URL"
    sed -i "s|API_BASE_URL = \"http://localhost:8000\"|API_BASE_URL = os.getenv('GAM_API_URL', 'http://localhost:8000')|" /app/dashboard/app.py
    # Also import os at top if not present (check and add)
    if ! grep -q "import os" /app/dashboard/app.py; then
        sed -i '13i import os' /app/dashboard/app.py
    fi
else
    log "API_BASE_URL already configured or not hardcoded"
fi

# Mode-specific configuration
if [ "$GAM_ENV" = "development" ] || [ "$GAM_ENV" = "dev" ]; then
    log "Starting in development mode"
    STREAMLIT_ARGS="--server.runOnSave true --logger.level debug --theme.base light"
else
    log "Starting in production mode"
    STREAMLIT_ARGS="--server.headless true --logger.level $LOG_LEVEL --theme.base dark --server.enableCORS false --server.enableXsrfProtection false"
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