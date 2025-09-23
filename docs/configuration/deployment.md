# Deployment and Scaling Guide

## Overview

This guide covers deploying GeoAnomalyMapper (GAM) for local development, production analysis, and large-scale processing. GAM is Python-based and container-friendly, supporting local runs, cloud environments (AWS, Google Cloud, Azure), and Docker for reproducibility. Scaling focuses on parallel processing with Dask for global workloads.

**Key Concepts**:
- **Local Deployment**: Single machine for development/regional analysis.
- **Cloud Deployment**: Use managed compute for global runs (e.g., EC2, GCE).
- **Containerization**: Docker for consistent environments.
- **Scaling**: Dask clusters for parallelism; monitor with dashboard.

For configuration, see [Config Reference](config_reference.md). Ensure GAM installed with extras: `pip install geoanomalymapper[geophysics,visualization]`.

## Local Deployment Options

For development or small-scale analysis (e.g., regional bbox).

1. **Basic Setup**:
   - Install as per [Installation Guide](../user/installation.md).
   - Run via CLI: `gam run --config local_config.yaml`.
   - Or Jupyter: Use tutorials in [docs/tutorials/](.. /tutorials/).

2. **Performance Tuning**:
   - Set `parallel_workers: -1` in config for all cores.
   - Cache on SSD: `cache_dir: "/fast/storage/cache"`.
   - Monitor: `dask dashboard` for task graphs.

3. **Example Local Run**:
   ```bash
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
   ```bash
   # SSH to instance
   conda create -n gam python=3.12
   conda activate gam
   conda install -c conda-forge gdal obspy simpeg
   pip install geoanomalymapper[geophysics,visualization]
   ```

3. **Dask Cluster on EMR** (for global):
   - Create EMR cluster (m5.xlarge master, 4x m5.xlarge core).
   - Install GAM on bootstrap script:
     ```bash
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
   ```bash
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

Docker for reproducible environments, especially geospatial deps.

1. **Dockerfile** (create in root):
   ```dockerfile
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
   ```yaml
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
   ```bash
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
```bash
docker run -v cache:/app/data/cache gam run --global --modalities gravity --tile-size 30
```

For HPC, integrate with SLURM/PBS via dask-jobqueue.

---

*Last Updated: 2025-09-23 | GAM v1.0.0*