# Makefile for GAM Docker Deployment
# Usage: From deployment/ directory, run 'make up' etc.
# Assumes docker-compose.yml in deployment/docker/

DC := docker compose -f docker/docker-compose.yml

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  up          - Build and start services (production)"
	@echo "  dev         - Build and start in development mode"
	@echo "  down        - Stop services"
	@echo "  logs        - Follow logs"
	@echo "  build       - Build images"
	@echo "  clean       - Stop and remove volumes"
	@echo "  rebuild     - Rebuild images without cache"

# Production deployment
.PHONY: up
up:
	$(DC) up -d --build

# Development mode
.PHONY: dev
dev:
	GAM_ENV=development $(DC) up -d --build

# Stop services
.PHONY: down
down:
	$(DC) down

# Follow logs
.PHONY: logs
logs:
	$(DC) logs -f

# Build images
.PHONY: build
build:
	$(DC) build

# Clean (stop and remove volumes - data loss)
.PHONY: clean
clean:
	$(DC) down -v
	docker system prune -f

# Rebuild without cache
.PHONY: rebuild
rebuild:
	$(DC) build --no-cache
	$(DC) up -d