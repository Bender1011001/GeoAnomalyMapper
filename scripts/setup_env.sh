#!/bin/bash
# setup_env.sh - Environment setup script for GeoAnomalyMapper (GAM)
# Installs system dependencies, Python environment, and GAM package.
# Usage: ./scripts/setup_env.sh [OPTIONS]
# Options:
#   --dry-run: Show commands without executing
#   --no-deps: Skip system dependency installation
#   --help: Show this help

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DRY_RUN=false
NO_DEPS=false
PYTHON_VERSION="3.12"

# Help function
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --dry-run    Show commands without executing"
  echo "  --no-deps    Skip system dependency installation"
  echo "  --help       Show this help message"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-deps)
      NO_DEPS=true
      shift
      ;;
    --help)
      show_help
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      show_help
      ;;
  esac
done

# Function to run commands (with dry-run support)
run_cmd() {
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: $@"
  else
    "$@"
  fi
}

echo -e "${YELLOW}Starting GAM environment setup...${NC}"

# Detect OS
OS=$(uname -s)
echo "Detected OS: $OS"

if [[ "$OS" == "Linux" ]]; then
  DISTRO=$(lsb_release -i 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
  echo "Detected distro: $DISTRO"
  
  if [[ "$NO_DEPS" == false ]]; then
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    if [[ "$DISTRO" == "Ubuntu" ]] || [[ "$DISTRO" == "Debian" ]]; then
      run_cmd sudo apt-get update
      run_cmd sudo apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        build-essential \
        libgdal-dev libgeos-dev libproj-dev libspatialindex-dev \
        libnetcdf-dev libhdf5-dev libpq-dev \
        libssl-dev libffi-dev \
        curl git wget \
        postgresql-client redis-tools
    elif [[ "$DISTRO" == "CentOS" ]] || [[ "$DISTRO" == "RedHat" ]] || [[ "$DISTRO" == "Fedora" ]]; then
      run_cmd sudo yum groupinstall -y "Development Tools"
      run_cmd sudo yum install -y python3 python3-pip python3-devel \
        gdal-devel geos-devel proj-devel spatialindex-devel \
        netcdf-devel hdf5-devel postgresql-devel \
        openssl-devel libffi-devel \
        curl git wget \
        postgresql redis
    else
      echo -e "${RED}Unsupported Linux distro: $DISTRO. Please install dependencies manually.${NC}"
      exit 1
    fi
  fi
elif [[ "$OS" == "Darwin" ]]; then  # macOS
  if [[ "$NO_DEPS" == false ]]; then
    echo -e "${YELLOW}Installing system dependencies via Homebrew...${NC}"
    if ! command -v brew &> /dev/null; then
      run_cmd /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    run_cmd brew install python@3.12 gdal geos proj postgresql redis
    run_cmd brew install --cask git
  fi
else
  echo -e "${RED}Unsupported OS: $OS${NC}"
  exit 1
fi

# Set up Python virtual environment
GAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$GAM_DIR/venv"

echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
if [[ -d "$VENV_DIR" ]]; then
  echo "Removing existing venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

run_cmd python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install wheel
run_cmd pip install --upgrade pip setuptools wheel

# Install GAM dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
run_cmd pip install -r "$GAM_DIR/requirements.txt"
run_cmd pip install -e .[geophysics,visualization]

# Generate configuration files
echo -e "${YELLOW}Generating configuration files...${NC}"
CONFIG_DIR="$GAM_DIR/config/production"
mkdir -p "$CONFIG_DIR"

if [[ ! -f "$CONFIG_DIR/secrets.yaml" ]]; then
  cp "$CONFIG_DIR/secrets.yaml.template" "$CONFIG_DIR/secrets.yaml"
  echo -e "${YELLOW}Created $CONFIG_DIR/secrets.yaml from template.${NC}"
  echo -e "${YELLOW}Please edit $CONFIG_DIR/secrets.yaml with your credentials and add to .gitignore.${NC}"
else
  echo "secrets.yaml already exists. Skipping."
fi

# Add venv to .gitignore if not present
if ! grep -q "^venv/" "$GAM_DIR/.gitignore" 2>/dev/null; then
  echo "venv/" >> "$GAM_DIR/.gitignore"
fi

echo -e "${GREEN}GAM environment setup completed successfully!${NC}"
echo -e "${GREEN}Virtual environment: $VENV_DIR${NC}"
echo -e "${GREEN}To activate: source $VENV_DIR/bin/activate${NC}"
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Edit config/production/secrets.yaml with your credentials"
echo "  2. For local dev: source venv/bin/activate && gam --help"
echo "  3. For Docker: docker-compose -f deployment/docker/docker-compose.yml up -d"
echo "  4. For K8s: kubectl apply -f deployment/k8s/"