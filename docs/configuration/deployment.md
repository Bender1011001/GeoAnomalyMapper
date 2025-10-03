# Deployment and Scaling Guide

## Overview

This guide covers deploying GeoAnomalyMapper (GAM) for local development, production analysis, and large-scale processing. GAM is Python-based and container-friendly, supporting local runs, cloud environments (AWS, Google Cloud, Azure), and Docker for reproducibility. Scaling focuses on parallel processing with Dask for global workloads.

**Key Concepts**:
- **Local Deployment**: Single machine for development/regional analysis.
- **Cloud Deployment**: Use managed compute for global runs (e.g., EC2, GCE).
- **Containerization**: Docker for consistent environments.
- **Scaling**: Dask clusters for parallelism; monitor with dashboard.

For configuration, see [`config_reference.md`](configuration/config_reference.md). Ensure GAM installed with extras: `pip install geoanomalymapper[geophysics,visualization]`.

## Local Deployment Options

For development or small-scale analysis (e.g., regional bbox).

1. **Basic Setup**:
   - Install as per [`installation.md`](../user/installation.md).
   - Run via CLI: `gam run --config local_config.yaml`.
   - Or Jupyter: Use tutorials in [docs/tutorials/](.. /tutorials/).

2. **Performance Tuning**:
   - Set `parallel_workers: -1` in config for all cores.
   - Cache on SSD: `cache_dir: "/fast/storage/cache"`.
   - Monitor: `dask dashboard` for task graphs.

3. **Example Local Run**:
   ```
   conda activate gam
   gam run --bbox 29.9 30.0 31.1 31.2 --modalities all --output local_results/
   ```

**Hardware Recommendation**: 16GB RAM, 4+ cores, 100GB storage. Run time: 5-30 min for regional.

## Cloud Deployment

For global or high-compute needs, deploy on cloud platforms. Use Dask for distributed scaling.

### AWS (EC2 + EMR)

1. **Launch Instance**:
   - EC2 t3.xlarge (4 vCPU, 16GB) for local; c5.4xlarge for scaling.
   - AMI: Deep Learning AMI (Ubuntu) with Python/Conda pre-installed.
   - Storage: EBS 100GB+ for cache.

2. **Install GAM**:
   ```
   # SSH to instance
   conda create -n gam python=3.12
   conda activate gam
   conda install -c conda-forge gdal obspy simpeg
   pip install geoanomalymapper[geophysics,visualization]
   ```

3. **Dask Cluster on EMR** (for global):
   - Create EMR cluster (m5.xlarge master, 4x m5.xlarge core).
   - Install GAM on bootstrap script:
     ```
     #!/bin/bash
     conda install -c conda-forge gdal obspy
     pip install geoanomalymapper[all]
     ```
   - Run: `dask-worker` on nodes; client on master.
   - Submit global job: Use `run_pipeline` with `global: true`.

4. **Storage**: S3 for cache/outputs (`cache_dir: "s3://gam-bucket/cache"` with s3fs).
   - Example: `pip install s3fs; config['data']['cache_dir'] = 's3://my-bucket/gam-cache'`.

**Cost Estimate**: EC2 t3.xlarge ~$0.2/hr; EMR global run ~$10-50 depending on tiles.

### Google Cloud (GCE + Dataproc)

1. **Launch VM**:
   - n1-standard-4 (4 vCPU, 15GB).
   - Image: Deep Learning VM (with Conda).

2. **Install**:
   Similar to AWS; use `gcloud compute ssh`.

3. **Dataproc Cluster** (Dask equivalent):
   - Create cluster: `gcloud dataproc clusters create gam-cluster --num-workers 4 --worker-machine-type n1-standard-4`.
   - Install GAM via initialization action (script URL on GCS).
   - Run distributed: Use dask-yarn or dask-kubernetes.

4. **Storage**: GCS (`cache_dir: "gs://gam-bucket/cache"` with gcsfs).

**Integration**: Use Google Earth Engine for auxiliary data if extending.

### Azure (VM + Batch)

1. **Launch VM**:
   - Standard_D4s_v3 (4 vCPU, 16GB).
   - Image: Ubuntu with Python.

2. **Install**:
   ```
   az vm ssh
   # Conda/pip as above
   ```

3. **Azure Batch for Scaling**:
   - Pool: 4 nodes, Standard_D4s_v3.
   - Job: Script to run `gam run --global`.
   - Install GAM in task script.

4. **Storage**: Azure Blob (`cache_dir: "az://gam-container/cache"` with adlfs).

**Cost**: Similar to AWS; use spot instances for savings.

## Docker Containerization

Docker for reproducible environments, especially geospatial deps. The provided Docker Compose stack includes an NGINX reverse proxy in front of the Streamlit dashboard and FastAPI API.

See [`docker-compose.yml`](deployment/docker/docker-compose.yml) and [`nginx.conf`](deployment/docker/nginx.conf).

### Routing
- `/` → dashboard
- `/analysis` → API
- `/tiles` → API

### Prerequisites
- Docker and Docker Compose installed.
- A Cesium Ion token (optional for initial load, required for terrain). Reference:
  - [`installation.md`](../user/installation.md)
  - [`quickstart.md`](../user/quickstart.md)

### Quick Start: Docker Compose
- Environment file setup (verify variable name CESIUM_TOKEN is used in compose):
  ```
  cp GeoAnomalyMapper/deployment/docker/.env.example .env
  # Edit .env and set:
  # CESIUM_TOKEN=your_token_here
  ```
- Bring up the stack (ensure the compose path is correct):
  ```
  docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml up -d
  ```
- Access URLs:
  - Dashboard: http://localhost:8080/
  - API docs (proxied): http://localhost:8080/analysis/docs
  - Tiles (proxied): http://localhost:8080/tiles
- Verification examples:
  - Tileset manifest (replace myset with an actual tileset):
    - URL: http://localhost:8080/tiles/myset/tileset.json
    - Curl:
      ```
      curl -I http://localhost:8080/tiles/myset/tileset.json
      ```
  - Scene artifact (replace your analysis id):
    - URL: http://localhost:8080/analysis/<analysis_id>/scene.json
    - Curl:
      ```
      curl -s http://localhost:8080/analysis/<analysis_id>/scene.json | jq .
      ```

  - **New Lightweight Scene Endpoint**: `GET /api/scene/{analysis_id}`
    Returns the scene configuration JSON directly.
    - **Response**: Raw JSON object from `data/outputs/state/{analysis_id}/scene.json`.
    - **Status**: 200 OK, 404 if not found.

    **Example URL** (API server at `localhost:8000`):
    - Scene JSON: `http://localhost:8000/api/scene/voids_carlsbad`

    **cURL example**:
      ```
      curl -s http://localhost:8000/api/scene/voids_carlsbad
      ```

### Service Responsibilities
- Dashboard container runs Streamlit app from [`app.py`](dashboard/app.py) and consumes CESIUM_TOKEN from environment.
- API container runs FastAPI app from [`main.py`](gam/api/main.py). Tiles are served under /tiles. Scenes are under /analysis/<id>/scene.json, per [`api_reference.md`](../developer/api_reference.md).

### Reverse Proxy Details (NGINX)
Key locations from [`nginx.conf`](deployment/docker/nginx.conf):
- `/` → dashboard upstream
- `/analysis` → API upstream
- `/api/scene` → API upstream (new lightweight endpoint)
- `/tiles` → API upstream

### Configuration and Environment
- Passing CESIUM_TOKEN:
  - Preferred via .env used by Docker Compose, or by exporting in shell before running compose.
  - Example:
    - Linux/macOS:
      ```
      export CESIUM_TOKEN="your_token_here"
      docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml up -d
      ```
    - Windows PowerShell:
      ```
      $Env:CESIUM_TOKEN = "your_token_here"
      docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml up -d
      ```
- Optional: Mention you can set additional variables found in [`docker-compose.yml`](deployment/docker/docker-compose.yml) (ports, volumes) and that tiles live under data/outputs/tilesets by default.

### Common Operations
- Rebuild images after code changes (verify flags):
  ```
  docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml build --no-cache
  docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml up -d
  ```
- Tail logs:
  ```
  docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml logs -f
  ```
- Stop stack:
  ```
  docker compose -f GeoAnomalyMapper/deployment/docker/docker-compose.yml down
  ```

### Troubleshooting
- 404 for /tiles/... → Confirm tileset exists under data/outputs/tilesets and API container mounts that path. See [`main.py`](gam/api/main.py) and [`api_reference.md`](../developer/api_reference.md).
- 404 for /api/scene/<id> or /analysis/<id>/scene.json → Confirm analysis artifacts are present per [`artifacts.py`](gam/core/artifacts.py).
- Missing CESIUM_TOKEN → Globe loads without terrain; set token via .env or environment. See [`globe_viewer.md`](../user/globe_viewer.md).
- Reverse proxy mismatch → Inspect [`nginx.conf`](deployment/docker/nginx.conf).

1. **Dockerfile** (create in root):
   ```
   FROM condaforge/mambaforge:latest

   COPY environment.yml /tmp/environment.yml
   RUN mamba env create -f /tmp/environment.yml && mamba clean --all

   COPY . /app
   WORKDIR /app
   RUN pip install -e .[geophysics,visualization,dev]

   ENTRYPOINT ["gam"]
   CMD ["run", "--help"]
   ```

2. **environment.yml**:
   ```
   name: gam
   channels:
     - conda-forge
   dependencies:
     - python=3.12
     - gdal
     - obspy
     - simpeg
     - pygimli
     - pip
     - pip:
       - geoanomalymapper[all]
   ```

3. **Build and Run**:
   ```
   docker build -t gam .
   docker run -v $(pwd)/data:/app/data gam run --config config.yaml --output /app/data/output
   ```

4. **For Global/Dask**:
   - Use dask-gateway for multi-container clusters.
   - Docker Compose for local Dask: Add dask-scheduler, dask-worker services.

**Benefits**: Handles GDAL binaries; share images on Docker Hub.

## Scaling and Performance Tuning

- **Parallelism**: `parallel_workers: -1` for local; scale to 100+ on cloud.
- **Memory Management**: Chunk data (Dask chunks=1000); spill to disk (`memory_limit='4GB'` in Client).
- **Optimization**:
  - Coarse grid for scouting (grid_res=1.0).
  - Batch modalities (run gravity first).
  - Checkpoint tiles: Save intermediate HDF5.
- **Monitoring**: Dask dashboard, Prometheus for cloud.
- **Cost/Perf Tradeoffs**: Use spot/preemptible instances; auto-scale clusters.

**Example Global Docker Run**:
```
docker run -v cache:/app/data/cache gam run --global --modalities gravity --tile-size 30
```

For HPC, integrate with SLURM/PBS via dask-jobqueue.

---
*Last Updated: 2025-10-03 | GAM v1.0.0*

### Local API Runbook

For direct API testing (bypassing Docker):

```bash
uvicorn GeoAnomalyMapper.gam.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Verification cURL Examples**:

- Scene endpoint:
  ```bash
  curl -s http://127.0.0.1:8000/api/scene/voids_carlsbad
  ```

- Tiles head (replace `your/tileset` with actual path):
  ```bash
  curl -I http://127.0.0.1:8000/tiles/your/tileset/tileset.json
  ```

### Reverse Proxy Alignment Note

When using a reverse proxy (e.g., NGINX), ensure `/api/scene` and `/tiles` are routed to the FastAPI service, mapping `/tiles` to the `data/outputs/tilesets` directory. Align with the provided [`nginx.conf`](deployment/docker/nginx.conf) snippet for Docker deployments, but no changes to Docker files are needed.