from fastapi import FastAPI, BackgroundTasks, HTTPException, APIRouter
from pydantic import BaseModel, conlist, validator
from typing import Dict, List, Optional, Any, Tuple
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uuid
import asyncio
import json
import os
from datetime import datetime
from .job_store import job_store
from gam.core.pipeline import GAMPipeline
from gam.core.config import GAMConfig
from gam.core.exceptions import (
    PipelineError, IngestionError, PreprocessingError,
    ModelingError, VisualizationError
)
from gam.core.artifacts import load_scene_config, scene_json_path
from pathlib import Path

app = FastAPI(title="GeoAnomalyMapper API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}
 
JOB_STATES = {"QUEUED", "RUNNING", "COMPLETED", "FAILED"}
 
# Tiles configuration
tiles_root = Path("data/outputs/tilesets")
tiles_root.mkdir(parents=True, exist_ok=True)
# Reverse proxies should pass through /tiles/* to this service
app.mount("/tiles", StaticFiles(directory=str(tiles_root)), name="tiles")
 
router = APIRouter()

@router.get("/tilesets/{name}/tileset.json")
def get_tileset_json(name: str):
    ts_path = tiles_root / name / "tileset.json"
    if not ts_path.is_file():
        raise HTTPException(status_code=404, detail=f"Tileset '{name}' not found")
    try:
        with ts_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid tileset.json in '{name}'")
    return JSONResponse(content=data)

@router.get("/tiles/tilesets.json")
def list_tilesets():
    tilesets = []
    if tiles_root.exists():
        for entry in os.listdir(tiles_root):
            full = tiles_root / entry
            if full.is_dir() and (full / "tileset.json").is_file():
                tilesets.append({"name": entry, "url": f"/tiles/{entry}/tileset.json"})
    tilesets.sort(key=lambda x: x["name"])
    return {"tilesets": tilesets}

@router.get("/analysis/{analysis_id}/scene.json")
def get_scene_config(analysis_id: str):
    path = scene_json_path(analysis_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Scene not found")
    try:
        text = load_scene_config(analysis_id)
        data = json.loads(text)
    except json.JSONDecodeError:
        # Persisted by our utility should be valid JSON; if not, surface server error.
        raise HTTPException(status_code=500, detail="Invalid scene.json content")
    return JSONResponse(content=data)

app.include_router(router)

class AnalysisRequest(BaseModel):
    bbox: conlist(float, min_length=4, max_length=4)
    modalities: List[str]
    output_dir: str
    config_path: Optional[str] = None
    verbose: bool = False

    @validator("bbox")
    def validate_bbox(cls, v):
        # v is guaranteed length 4 by conlist; enforce numeric and reasonable ranges
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("bbox must contain only numeric values")
        # enforce ordering: [min_lon, min_lat, max_lon, max_lat]
        min_lon, min_lat, max_lon, max_lat = v
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat] with min < max")
        return v

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    message: Optional[str] = None

class ResultsResponse(BaseModel):
    job_id: str
    results: Dict[str, Any]
    output_files: Dict[str, str]

async def run_analysis_job(job_id: str, request: AnalysisRequest):
    """Background task to run the analysis pipeline asynchronously."""
    def progress_callback(stage: str, progress: float):
        """Callback to update job store with progress."""
        job_store.update_job(job_id, {"stage": stage, "progress": progress})

    job_store.update_job(job_id, {
        "status": "RUNNING",
        "start_time": datetime.now(),
        "progress": 0.0,
        "stage": "Starting"
    })

    pipeline = None
    try:
        # Create unique output directory for this job
        base_output_dir = Path(request.output_dir)
        job_output_dir = base_output_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        config = GAMConfig.from_yaml(request.config_path) if request.config_path else GAMConfig()

        # Create pipeline instance
        pipeline = GAMPipeline(
            config=config,
            use_dask=config.parallel.backend == 'dask',
            n_workers=config.parallel.n_workers,
            cache_dir=Path(config.cache_dir)
        )

        # Convert bbox to tuple
        bbox_tuple: Tuple[float, float, float, float] = tuple(request.bbox)

        # Run the full pipeline
        results = pipeline.run_analysis(
            bbox=bbox_tuple,
            modalities=request.modalities,
            output_dir=str(job_output_dir),
            use_cache=True,
            progress_callback=progress_callback
        )

        # Store results
        job_store.update_job(job_id, {
            "status": "COMPLETED",
            "progress": 1.0,
            "stage": "Finished",
            "results": {
                "raw_data": results.raw_data,
                "processed_data": results.processed_data,
                "inversion_results": results.inversion_results,
                "anomalies": results.anomalies.to_dict(orient='records'),
                "visualizations": {k: str(v) for k, v in results.visualizations.items()}
            },
            "output_files": {k: str(v) for k, v in results.visualizations.items()},
            "message": f"Analysis completed successfully. Found {len(results.anomalies)} anomalies."
        })

    except IngestionError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Ingestion failed: {str(e)}",
            "stage": "Ingestion"
        })
    except PreprocessingError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Preprocessing failed: {str(e)}",
            "stage": "Preprocessing"
        })
    except ModelingError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Modeling failed: {str(e)}",
            "stage": "Modeling"
        })
    except VisualizationError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Visualization failed: {str(e)}",
            "stage": "Visualization"
        })
    except PipelineError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Pipeline error: {str(e)}",
            "stage": "Pipeline"
        })
    except Exception as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Unexpected error: {str(e)}",
            "stage": "Unknown"
        })
    finally:
        if pipeline:
            pipeline.close()

@app.post("/analysis", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = job_store.create_job()
    background_tasks.add_task(run_analysis_job, job_id, request)
    return AnalysisResponse(
        job_id=job_id,
        status="QUEUED",
        message="Analysis job queued successfully"
    )

@app.get("/analysis/{job_id}/status", response_model=StatusResponse)
async def get_status(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        message=job.get("message") or job.get("error_message")
    )

@app.get("/analysis/{job_id}/results", response_model=ResultsResponse)
async def get_results(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=425, detail="Job not completed yet")
    if job["results"] is None:
        raise HTTPException(status_code=500, detail="Results not available")
    return ResultsResponse(
        job_id=job_id,
        results=job["results"],
        output_files=job["output_files"]
    )