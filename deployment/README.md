# GeoAnomalyMapper Docker Deployment

This directory contains the Docker-based deployment configuration for the GeoAnomalyMapper (GAM) professional interface, consisting of a FastAPI backend for async job management and a Streamlit frontend for interactive 2D/3D visualization. The setup uses containerization for easy setup, scalability, and production readiness.

The deployment supports data persistence via named volumes and internal service communication via Docker networks. No modifications to core GAM modules are required.

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **Hardware**: Minimum 4GB RAM, 2 CPUs; recommended 8GB+ for geospatial processing
- **Disk**: 5GB+ free space for images, data, and results

### Software Requirements
- **Docker**: Version 20.10+ (install from [docker.com](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (included with Docker Desktop; standalone from [docs.docker.com/compose/install](https://docs.docker.com/compose/install/))
- **Git**: For cloning the repository

Verify installation:
```bash
docker --version
docker compose version
```

### GAM Repository
Clone the GAM repository:
```bash
git clone https://github.com/your-org/GeoAnomalyMapper.git
cd GeoAnomalyMapper
```

## Setup

1. **Navigate to Deployment Directory**:
   ```bash
   cd deployment/docker
   ```

2. **Configure Environment Variables**:
   Copy the example environment file and customize as needed:
   ```bash
   cp .env.example .env
   # Edit .env for custom settings (e.g., ports, paths)
   ```
   Key variables:
   - `GAM_ENV`: `production` (default) or `development`
   - `GAM_API_URL`: Internal (`http://gam-api:8000`) for Docker; `http://localhost:8000` for local dev
   - `UVICORN_WORKERS`: Number of API workers (default: 4 for prod)
   - `LOG_LEVEL`: Logging verbosity (default: `INFO`)

3. **Optional: Custom Volume Paths**:
   If using host directories instead of named volumes, set `DATA_PATH` and `RESULTS_PATH` in `.env` and update `docker-compose.yml` volumes accordingly.

## Running the Deployment

### Production Deployment
Build and start the services in detached mode:
```bash
docker compose up -d --build
```

- **FastAPI Backend**: Available at `http://localhost:8000`
  - OpenAPI docs: `http://localhost:8000/docs`
  - Health: `http://localhost:8000` (returns 200 if healthy)
- **Streamlit Dashboard**: Available at `http://localhost:8501`
  - Access the UI to run analyses, view 2D/3D visualizations

Services start in order: API first, then dashboard (waits for API health).

### Development Mode
For hot-reload and debugging:
```bash
# Edit .env: GAM_ENV=development
docker compose up -d --build
# Follow logs: docker compose logs -f gam-dashboard
```

### Stopping and Cleanup
```bash
# Stop services
docker compose down

# Stop and remove volumes (data loss warning)
docker compose down -v

# Rebuild images
docker compose build --no-cache
```

### Makefile Commands (Optional)
See [Makefile](#makefile) for simplified commands.

## Monitoring and Logging

### Service Status
Check container health and status:
```bash
docker compose ps
docker compose logs gam-api    # API logs
docker compose logs gam-dashboard  # Dashboard logs
```

### Health Checks
- API: `curl http://localhost:8000` (OpenAPI response)
- Dashboard: `curl http://localhost:8501/healthz` (Streamlit health)

### Resource Monitoring
Use `docker stats` for CPU/memory usage:
```bash
docker stats
```

For advanced monitoring, integrate with Prometheus/Grafana (see `monitoring/` directory in repo root).

### Logs Rotation
Docker handles log rotation; configure via daemon.json if needed.

## Troubleshooting

### Common Issues

1. **Port Conflicts**:
   - Error: "Port 8000/8501 already in use"
   - Solution: Stop conflicting services or change ports in `.env` and `docker-compose.yml`

2. **API Connectivity from Dashboard**:
   - Symptom: Dashboard shows "Backend API not available"
   - Solution: 
     - Ensure `GAM_API_URL=http://gam-api:8000` in `.env` (internal)
     - Check API health: `curl http://localhost:8000`
     - Verify network: `docker compose logs gam-dashboard`

3. **Permission Errors on Volumes**:
   - Error: "Permission denied" on /app/data or /app/results
   - Solution: Run `chmod -R 755 data results` on host if using bind mounts, or ensure non-root user in Dockerfiles

4. **Build Failures (Geospatial Deps)**:
   - Error: GDAL/GEOS compilation issues
   - Solution: Ensure build-essential and lib*-dev packages are available; use `--no-cache` rebuild

5. **Slow Startup or OOM**:
   - Increase Docker resources in settings (e.g., 8GB RAM)
   - Reduce `UVICORN_WORKERS` or use smaller bbox in analyses

6. **No Anomalies/Empty Results**:
   - Check bbox validity and modalities in dashboard
   - Verify data sources in `config.yaml`
   - Review API job status: `curl http://localhost:8000/analysis/{job_id}/status`

### Debugging Tips
- **Interactive Shell**: `docker compose run --rm gam-api bash` (debug API)
- **Tail Logs**: `docker compose logs -f --tail=100`
- **Inspect Volumes**: `docker volume inspect gam-data`
- **Rebuild Specific Service**: `docker compose up -d --build gam-dashboard`

### Known Limitations
- Startup script patches `dashboard/app.py` for API URL (minimal change; revert on updates)
- No built-in auth; add reverse proxy (e.g., Traefik/Nginx) for production
- Geospatial processing is CPU-intensive; scale with Docker Swarm/K8s for large jobs

## File Structure

```
deployment/docker/
├── docker-compose.yml          # Orchestration for API + Dashboard
├── Dockerfile.api              # FastAPI backend container
├── Dockerfile.dashboard        # Streamlit frontend container
├── start-api.sh                # API startup with uvicorn
├── start-dashboard.sh          # Dashboard startup with Streamlit patching
├── .env.example                # Environment template
└── README.md                   # This guide
```

## Makefile (Optional)

Create `deployment/Makefile` for convenience:
```makefile
# Build and run
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

# Development
dev:
	GAM_ENV=development docker compose up -d --build

# Clean
clean:
	docker compose down -v
	docker system prune -f

.PHONY: up down logs dev clean
```

Run: `make up` from `deployment/docker/`.

## Next Steps

- **Scaling**: Use Docker Swarm or Kubernetes (see `deployment/k8s/`)
- **Cloud Deployment**: AWS/EC2, Azure, GCP (configs in `deployment/cloud/`)
- **CI/CD**: Integrate with GitHub Actions for automated builds
- **Security**: Add HTTPS via Traefik, secrets management with Docker Secrets

For GAM core documentation, see `docs/user/installation.md`. Report issues at [GitHub Issues](https://github.com/your-org/GeoAnomalyMapper/issues).

---
*Last updated: 2025-09-24 | Version: 1.0*