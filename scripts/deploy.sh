#!/bin/bash
# deploy.sh - Main deployment script for GeoAnomalyMapper (GAM)
# Automates deployment to different environments and platforms.
# Usage: ./scripts/deploy.sh [OPTIONS] --environment <env> [--platform <platform>]
# Environments: local, docker, k8s, cloud
# Platforms (for cloud): aws, gcp, azure
# Options:
#   --environment, -e: Target environment
#   --platform, -p: Cloud platform (required for cloud env)
#   --dry-run: Show commands without executing
#   --no-setup: Skip environment setup
#   --help: Show this help

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
ENVIRONMENT=""
PLATFORM=""
DRY_RUN=false
NO_SETUP=false
GAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

show_help() {
  echo "Usage: $0 [OPTIONS] --environment <env> [--platform <platform>]"
  echo "Options:"
  echo "  -e, --environment STRING  Target environment (local|docker|k8s|cloud)"
  echo "  -p, --platform STRING     Cloud platform (aws|gcp|azure) - required for cloud"
  echo "  --dry-run                 Show commands without executing"
  echo "  --no-setup                Skip environment setup"
  echo "  --help                    Show this help"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    -p|--platform)
      PLATFORM="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-setup)
      NO_SETUP=true
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

if [[ -z "$ENVIRONMENT" ]]; then
  echo -e "${RED}Error: --environment is required.${NC}"
  show_help
fi

if [[ "$ENVIRONMENT" == "cloud" && -z "$PLATFORM" ]]; then
  echo -e "${RED}Error: --platform is required for cloud environment.${NC}"
  show_help
fi

run_cmd() {
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: $@"
  else
    "$@"
  fi
}

echo -e "${YELLOW}Starting GAM deployment to environment: $ENVIRONMENT${NC}"
if [[ "$ENVIRONMENT" == "cloud" ]]; then
  echo -e "${YELLOW}Platform: $PLATFORM${NC}"
fi

# Prerequisite checks
echo -e "${YELLOW}Checking prerequisites...${NC}"

case "$ENVIRONMENT" in
  local)
    command -v python3 >/dev/null 2>&1 || { echo -e "${RED}Python3 is required for local deployment.${NC}"; exit 1; }
    ;;
  docker)
    command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required for docker deployment.${NC}"; exit 1; }
    docker compose version >/dev/null 2>&1 || { echo -e "${RED}Docker Compose is required.${NC}"; exit 1; }
    ;;
  k8s)
    command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl is required for k8s deployment.${NC}"; exit 1; }
    kubectl cluster-info >/dev/null 2>&1 || { echo -e "${RED}No Kubernetes cluster accessible.${NC}"; exit 1; }
    ;;
  cloud)
    case "$PLATFORM" in
      aws)
        command -v aws >/dev/null 2>&1 || { echo -e "${RED}AWS CLI is required.${NC}"; exit 1; }
        aws sts get-caller-identity >/dev/null 2>&1 || { echo -e "${RED}Not authenticated to AWS.${NC}"; exit 1; }
        ;;
      gcp)
        command -v gcloud >/dev/null 2>&1 || { echo -e "${RED}gcloud CLI is required.${NC}"; exit 1; }
        gcloud auth list --filter=status:ACTIVE >/dev/null 2>&1 || { echo -e "${RED}Not authenticated to GCP.${NC}"; exit 1; }
        command -v terraform >/dev/null 2>&1 || { echo -e "${RED}Terraform is required.${NC}"; exit 1; }
        ;;
      azure)
        command -v az >/dev/null 2>&1 || { echo -e "${RED}Azure CLI is required.${NC}"; exit 1; }
        az account show >/dev/null 2>&1 || { echo -e "${RED}Not authenticated to Azure.${NC}"; exit 1; }
        ;;
      *)
        echo -e "${RED}Unsupported platform: $PLATFORM${NC}"
        exit 1
        ;;
    esac
    ;;
esac

# Environment setup if not skipped
if [[ "$NO_SETUP" == false ]]; then
  echo -e "${YELLOW}Running environment setup...${NC}"
  run_cmd "$GAM_DIR/scripts/setup_env.sh" --dry-run="$DRY_RUN"
fi

# Deployment logic
case "$ENVIRONMENT" in
  local)
    echo -e "${YELLOW}Deploying to local environment...${NC}"
    VENV_DIR="$GAM_DIR/venv"
    if [[ ! -d "$VENV_DIR" ]]; then
      echo -e "${RED}Virtual environment not found. Run setup_env.sh first.${NC}"
      exit 1
    fi
    run_cmd source "$VENV_DIR/bin/activate"
    run_cmd gam db migrate  # Setup/migrate DB if needed
    run_cmd nohup gam run --config "$GAM_DIR/config/production/production.yaml" --daemon > gam.log 2>&1 &
    echo $! > gam.pid
    echo -e "${GREEN}Local GAM process started (PID: $!). Logs: gam.log${NC}"
    ;;
  docker)
    echo -e "${YELLOW}Deploying to Docker...${NC}"
    run_cmd docker compose -f "$GAM_DIR/deployment/docker/docker-compose.yml" up -d
    run_cmd docker compose ps  # Show status
    # Wait for services to be healthy
    run_cmd timeout 300 sh -c 'until docker compose ps | grep -q "healthy"; do sleep 5; done'
    echo -e "${GREEN}Docker deployment complete. Services: docker compose ps${NC}"
    ;;
  k8s)
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
    run_cmd kubectl apply -f "$GAM_DIR/deployment/k8s/"
    run_cmd kubectl rollout status deployment/gam-deployment -n gam-system --timeout=300s
    run_cmd kubectl get pods -n gam-system
    run_cmd kubectl get svc -n gam-system
    echo -e "${GREEN}K8s deployment complete. Check: kubectl get all -n gam-system${NC}"
    ;;
  cloud)
    case "$PLATFORM" in
      aws)
        echo -e "${YELLOW}Deploying to AWS...${NC}"
        cd "$GAM_DIR/deployment/cloud/aws"
        run_cmd aws cloudformation deploy \
          --template-file cloudformation.yaml \
          --stack-name "gam-${ENVIRONMENT}" \
          --capabilities CAPABILITY_IAM \
          --parameter-overrides Environment=${ENVIRONMENT}
        run_cmd aws cloudformation describe-stacks --stack-name "gam-${ENVIRONMENT}" --query "Stacks[0].Outputs"
        ;;
      gcp)
        echo -e "${YELLOW}Deploying to GCP...${NC}"
        cd "$GAM_DIR/deployment/cloud/gcp/terraform"
        run_cmd terraform init
        run_cmd terraform plan -var="environment=${ENVIRONMENT}" -var="project_id=${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
        run_cmd terraform apply -auto-approve -var="environment=${ENVIRONMENT}" -var="project_id=${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
        run_cmd terraform output
        ;;
      azure)
        echo -e "${YELLOW}Deploying to Azure...${NC}"
        cd "$GAM_DIR/deployment/cloud/azure"
        run_cmd az deployment group create \
          --resource-group "gam-rg-${ENVIRONMENT}" \
          --template-file arm-template.json \
          --parameters environment=${ENVIRONMENT} location=EastUS
        run_cmd az storage account list --query "[?starts_with(name, 'gamstorage')]" -o table
        ;;
    esac
    echo -e "${GREEN}Cloud deployment complete. Check outputs above.${NC}"
    ;;
esac

# Database setup (common)
echo -e "${YELLOW}Setting up database...${NC}"
case "$ENVIRONMENT" in
  local|docker)
    # Assume local/postgres container
    run_cmd gam db create  # If GAM CLI supports
    run_cmd gam db migrate
    ;;
  k8s)
    run_cmd kubectl exec -n gam-system deployment/gam-deployment -- gam db migrate
    ;;
  cloud)
    # Cloud-specific, e.g., for AWS RDS
    echo "Database migration for cloud - run manually via bastion or Lambda"
    ;;
esac

# Health check
echo -e "${YELLOW}Running health check...${NC}"
run_cmd python3 "$GAM_DIR/scripts/health_check.py" --endpoint "http://localhost:8000"  # Adjust endpoint per env

echo -e "${GREEN}Deployment successful!${NC}"
echo -e "${YELLOW}Logs and monitoring:${NC}"
case "$ENVIRONMENT" in
  local) echo "  tail -f gam.log" ;;
  docker) echo "  docker compose logs -f" ;;
  k8s) echo "  kubectl logs -f deployment/gam-deployment -n gam-system" ;;
  cloud) echo "  Check cloud console logs" ;;
esac